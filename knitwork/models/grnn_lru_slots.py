"""Addressable key/value slot banks in a peripheral column of the LRU grid.

Column 0 stays the controller and the only output. One peripheral column per layer becomes S
independent slots. Per slot: key, value, occupancy (binary once written) and age (steps since last write).
The hub reads the old memory first (normalized query, bounded temperature, null address); the write
decision (gate) and the address are separate; a slot with zero address weight is preserved exactly.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from knitwork.models.grnn_lru_experimental import GridRnn as ExperimentalGridRnn
from knitwork.models.grnn_lru_sparse import entmax15


SLOT_MODES = ('slot_alloc', 'slot_soft', 'slot_entmax')
AGE_STEP = 0.01  # age saturates at 1 after 100 steps


class GridRnn(ExperimentalGridRnn):
    def __init__(
            self, *, mode, hidden_size, n_layers, n_columns, horizon, dtype, device,
            fb_norm=True, n_inputs=1, n_outputs=1, hub_query_size=8, n_slots=8,
            slot_key_size=16, slot_value_size=32, memory_column=None, slot_write_bias=0.0, read_scale=0.3,
    ):
        if mode not in SLOT_MODES:
            raise ValueError(f'Unknown slot mode: {mode}')
        for name, value in (('n_slots', n_slots), ('slot_key_size', slot_key_size), ('slot_value_size', slot_value_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        # The parent's 'associative' plumbing places a memory in one peripheral column; we override its math.
        super().__init__(
            mechanism='associative', hidden_size=hidden_size, n_layers=n_layers, n_columns=n_columns,
            horizon=horizon, dtype=dtype, device=device, fb_norm=fb_norm, n_inputs=n_inputs,
            n_outputs=n_outputs, hub_query_size=hub_query_size, memory_column=memory_column,
            bounded_feedback=True, residual_scale=read_scale, memory_key_size=slot_key_size, memory_value_size=slot_value_size,
        )
        self.mode, self.n_slots = mode, n_slots
        self.slot_key_size, self.slot_value_size = slot_key_size, slot_value_size
        del self.memory_writes, self.memory_queries, self.memory_outputs
        with torch.random.fork_rng(devices=[]):
            # write input: [routed message of the slot column ; hub context] -> key, value, gate (decision)
            self.slot_writes = nn.ModuleList([
                nn.Linear(2 * hidden_size, slot_key_size + slot_value_size + 1) for _ in range(n_layers)
            ])
            self.slot_queries = nn.ModuleList([nn.Linear(hidden_size, slot_key_size, bias=False) for _ in range(n_layers)])
            self.slot_outputs = nn.ModuleList([nn.Linear(slot_value_size, hidden_size, bias=False) for _ in range(n_layers)])
            self.raw_read_beta = nn.Parameter(torch.zeros(n_layers))  # beta = 1 + 15 * sigmoid(raw), 8.5 at init
            self.raw_write_beta = nn.Parameter(torch.zeros(n_layers))
            self.null_logit = nn.Parameter(torch.zeros(n_layers))  # "read nothing" address
            self.alloc_logit = nn.Parameter(torch.full((n_layers,), 4.0))  # pull toward the allocated slot
            for write in self.slot_writes:
                with torch.no_grad():
                    write.bias[slot_key_size + slot_value_size].fill_(slot_write_bias)

    @staticmethod
    def _beta(raw):
        return 1 + 15 * raw.sigmoid()

    def _split(self, memory):
        return torch.split(memory, (self.slot_key_size, self.slot_value_size, 1, 1), dim=-1)  # key, value, occ, age

    def _memory_read(self, layer, memory, context):
        keys, values, occ, _ = self._split(memory)  # [B, S, *]
        query = F.normalize(self.slot_queries[layer](context), dim=-1, eps=1e-6)  # [B, K]
        cosine = torch.einsum('bk,bsk->bs', query, F.normalize(keys, dim=-1, eps=1e-6))
        logits = self._beta(self.raw_read_beta[layer]) * cosine - 8.0 * (1 - occ.squeeze(-1))  # empty slots masked
        null = self.null_logit[layer].expand(logits.shape[0], 1)
        weights = torch.cat((logits, null), dim=-1).softmax(-1)[:, :-1]  # null address reads zeros
        read = self.slot_outputs[layer](torch.einsum('bs,bsv->bv', weights, values))
        return self.log_residual_scale[layer].exp() * F.rms_norm(read, (self.hidden_size,))  # bounded read

    def _allocation(self, occ, age):
        # first free slot; if none, the oldest one. Fixed rule, one-hot.
        free = (occ.squeeze(-1) < 0.5).to(age.dtype)  # [B, S]
        order = 0.01 * torch.arange(self.n_slots, device=age.device, dtype=age.dtype)
        score = free * (2 - order) + (1 - free.amax(-1, keepdim=True)) * (1 - free) * age.squeeze(-1)
        return F.one_hot(score.argmax(-1), self.n_slots).to(age.dtype)

    def _associative_step(self, layer, message, hidden, memory, context):
        lru_message = message.index_select(0, self.lru_columns)
        next_h = self._advance_cells(layer, lru_message, hidden)
        lru_out = self._project_cells(layer, lru_message, next_h)
        slot_message = message[self.memory_column]  # [B, H]
        # read the OLD memory for this step's output, then write
        slot_out = F.silu(self._memory_read(layer, memory, context)) + slot_message
        next_memory, gate = self._write(layer, memory, torch.cat((slot_message, context), dim=-1))
        out = lru_out.new_zeros(lru_out.shape[0], self.n_columns, self.hidden_size)
        out = out.index_copy(1, self.lru_columns, lru_out)
        out = out.index_copy(1, self.lru_columns.new_tensor([self.memory_column]), slot_out.unsqueeze(1))
        return out, next_h, next_memory, gate

    def _write(self, layer, memory, control):
        K, V = self.slot_key_size, self.slot_value_size
        keys, values, occ, age = self._split(memory)
        new_key, new_value, gate = torch.split(self.slot_writes[layer](control), (K, V, 1), dim=-1)
        new_key, new_value, gate = F.normalize(new_key, dim=-1, eps=1e-6), new_value.tanh(), gate.sigmoid()  # gate [B, 1]
        target = self._allocation(occ, age)  # [B, S]
        if self.mode == 'slot_alloc':
            address = target
        else:
            # match an occupied slot with similar key, else go to the allocated slot
            cosine = torch.einsum('bk,bsk->bs', new_key, F.normalize(keys, dim=-1, eps=1e-6))
            logits = self._beta(self.raw_write_beta[layer]) * cosine * occ.squeeze(-1) + self.alloc_logit[layer] * target
            address = logits.softmax(-1) if self.mode == 'slot_soft' else entmax15(logits)
        step = (gate * address).unsqueeze(-1)  # [B, S, 1]: zero address weight keeps a slot exactly
        keys = torch.lerp(keys, new_key.unsqueeze(1).expand_as(keys), step)
        values = torch.lerp(values, new_value.unsqueeze(1).expand_as(values), step)
        written = (step > 0.3).to(occ.dtype)
        occ = torch.maximum(occ, written)
        age = torch.minimum(age + AGE_STEP, torch.ones_like(age)) * (1 - written)
        return torch.cat((keys, values, occ, age), dim=-1), gate.detach()

    def init_state(self, bsz):
        state = super().init_state(bsz)
        state['memory'] = state['h'].new_zeros(self.n_layers, bsz, self.n_slots, self.slot_key_size + self.slot_value_size + 2)
        return state

    def carried_state_floats(self):
        slots = self.n_layers * self.n_slots * (self.slot_key_size + self.slot_value_size + 2)
        return super().carried_state_floats() - self.n_layers * self.memory_key_size * self.memory_value_size + slots
