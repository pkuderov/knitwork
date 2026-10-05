"""Five memory/access experiments with a single column-zero output bottleneck.

Each mechanism starts from the same star-connected, zero-query dynamic hub.
No auxiliary losses or discrete routing estimators are used.
"""

import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from knitwork.models.grnn_lru_core import LruBank
from knitwork.models.grnn_lru_sparse import GridRnn as SparseGridRnn


MECHANISMS = ('write_gate', 'query_read', 'associative', 'rotations', 'two_reads')


def delta_write(memory, key, value, rate):
    """Correct a key's current value; key is unit-normalized by the caller."""
    prediction = torch.einsum('bk,bkv->bv', key, memory)
    correction = rate * (value - prediction)
    return memory + key.unsqueeze(-1) * correction.unsqueeze(-2)


class CompactLruBank(LruBank):
    """Keep only real LRU columns when one column is replaced by a memory matrix."""

    def __init__(self, original, columns):
        nn.Module.__init__(self)
        for attribute in ('n_layers', 'hidden_size', 'horizon', 'hz_min', 'hz_max'):
            setattr(self, attribute, getattr(original, attribute))
        self.n_columns = len(columns)
        for name in ('log_r', 'theta', 'weight_in', 'weight_out'):
            self.register_parameter(name, nn.Parameter(getattr(original, name)[:, columns].detach().clone()))


class GridRnn(SparseGridRnn):
    def __init__(
            self, *, mechanism, hidden_size, n_layers, n_columns, horizon,
            dtype, device, fb_norm=True, n_inputs=1, n_outputs=1,
            hub_query_size=8, write_gate_bias=2.0, read_rank=16,
            memory_key_size=16, memory_value_size=32, memory_column=None,
            memory_write_bias=-2.0, rotation_groups=4, bounded_feedback=False, residual_scale=0.1,
    ):
        if mechanism not in MECHANISMS:
            raise ValueError(f'Unknown LRU mechanism: {mechanism}')
        if n_inputs != 1 or n_outputs != 1:
            raise ValueError('These experiments require one input and the column-zero output')
        if type(hub_query_size) is not int or hub_query_size < 1:
            raise ValueError('hub_query_size must be a positive integer')
        for name, value in (
            ('read_rank', read_rank), ('memory_key_size', memory_key_size),
            ('memory_value_size', memory_value_size), ('rotation_groups', rotation_groups),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if rotation_groups > hidden_size:
            raise ValueError('rotation_groups cannot exceed hidden_size')
        if not math.isfinite(write_gate_bias) or not math.isfinite(memory_write_bias):
            raise ValueError('Gate biases must be finite')
        memory_column = n_columns - 1 if memory_column is None else memory_column
        if type(memory_column) is not int or not 0 < memory_column < n_columns:
            raise ValueError('memory_column must be a peripheral column')
        super().__init__(
            hidden_size=hidden_size, n_layers=n_layers, n_columns=n_columns,
            horizon=horizon, dtype=dtype, device=device, fb_norm=fb_norm,
            n_inputs=n_inputs, n_outputs=n_outputs, topology='star',
            hub_query_size=hub_query_size, hub_query_init='zero',
        )
        self.mechanism = mechanism
        self.bounded_feedback = bounded_feedback
        self.read_rank = read_rank
        # small residual scale keeps the original message dominant at initialization
        self.log_residual_scale = nn.Parameter(torch.full((n_layers,), math.log(residual_scale))) if bounded_feedback else None
        self.memory_key_size = memory_key_size
        self.memory_value_size = memory_value_size
        self.memory_column = memory_column
        self.register_buffer('hub_column', torch.tensor([0]), persistent=False)
        # New modules must not change the same-seed initialization of embedding/head.
        with torch.random.fork_rng(devices=[]):
            if mechanism == 'write_gate':
                self.write_gates = nn.ModuleList([
                    nn.Linear(2 * hidden_size, n_columns - 1) for _ in range(n_layers)
                ])
                for gate in self.write_gates:
                    nn.init.normal_(gate.weight, std=0.01 / math.sqrt(2 * hidden_size))
                    nn.init.constant_(gate.bias, write_gate_bias)
            elif mechanism == 'query_read':
                self.read_queries = nn.ModuleList([
                    nn.Linear(hidden_size, read_rank, bias=False) for _ in range(n_layers)
                ])
                self.read_down = nn.Parameter(torch.empty(n_layers, n_columns - 1, 2 * hidden_size, read_rank))
                self.read_up = nn.Parameter(torch.empty(n_layers, n_columns - 1, read_rank, hidden_size))
                nn.init.uniform_(self.read_down, -1 / math.sqrt(2 * hidden_size), 1 / math.sqrt(2 * hidden_size))
                nn.init.uniform_(self.read_up, -1 / math.sqrt(read_rank), 1 / math.sqrt(read_rank))
            elif mechanism == 'associative':
                retained = [column for column in range(n_columns) if column != memory_column]
                self.register_buffer('lru_columns', torch.tensor(retained), persistent=False)
                self.cells = CompactLruBank(self.cells, retained)
                self.memory_writes = nn.ModuleList([
                    nn.Linear(hidden_size, memory_key_size + memory_value_size + 1)
                    for _ in range(n_layers)
                ])
                self.memory_queries = nn.ModuleList([
                    nn.Linear(hidden_size, memory_key_size, bias=False) for _ in range(n_layers)
                ])
                self.memory_outputs = nn.ModuleList([
                    nn.Linear(memory_value_size, hidden_size, bias=False) for _ in range(n_layers)
                ])
                for write in self.memory_writes:
                    with torch.no_grad():
                        write.bias[-1].fill_(memory_write_bias)
            elif mechanism == 'rotations':
                self.rotation_angles = nn.Parameter(torch.zeros(n_layers, n_columns - 1, rotation_groups))
                indices = torch.arange(hidden_size) * rotation_groups // hidden_size
                self.register_buffer('rotation_indices', indices, persistent=False)

    def _fresh(self, sources):
        return sources[:, self.n_columns:].mean(1) if sources.shape[1] > self.n_columns else sources[:, 0]

    def _hub_weights(self, layer, sources, context):
        comm = self.comm[layer]
        logits = comm.pi_route_logits[0].masked_fill(~comm.allowed[0], float('-inf'))
        query, keys = comm.query(context), comm.key(sources)
        scores = (keys * query.unsqueeze(1)).sum(-1) / math.sqrt(comm.hub_query_size)
        return (logits.unsqueeze(0) + scores).softmax(-1)

    def _query_values(self, layer, hidden, context):
        # Only the hub receives these narrow, query-conditioned private-memory reads.
        private = torch.bmm(hidden[1:], self.read_down[layer])
        gate = 2 * self.read_queries[layer](context).sigmoid()
        return torch.bmm(private * gate.unsqueeze(0), self.read_up[layer]).transpose(0, 1)

    def _memory_read(self, layer, memory, context):
        query = F.normalize(self.memory_queries[layer](context), dim=-1, eps=1e-6)
        retrieved = torch.einsum('bk,bkv->bv', query, memory)
        read = self.memory_outputs[layer](retrieved)
        return self.log_residual_scale[layer].exp() * F.rms_norm(read, (self.hidden_size,)) if self.bounded_feedback else read

    def _hub_message(self, layer, sources, context, hidden, memory=None):
        weights = self._hub_weights(layer, sources, context)
        values = sources
        if self.mechanism == 'query_read':
            read = self._query_values(layer, hidden, context)
            if self.bounded_feedback:  # normalized private read added to the original message
                read = sources[:, 1:self.n_columns] + self.log_residual_scale[layer].exp() * F.rms_norm(read, (self.hidden_size,))
            values = torch.cat((sources[:, :1], read, sources[:, self.n_columns:]), dim=1)
        elif self.mechanism == 'associative':
            column = self.memory_column
            read = self._memory_read(layer, memory, context).unsqueeze(1)
            if self.bounded_feedback:  # keep the original column message, add the bounded read
                read = sources[:, column:column + 1] + read
            values = torch.cat((sources[:, :column], read, sources[:, column + 1:]), dim=1)
        return torch.bmm(weights.unsqueeze(1), values).transpose(0, 1), weights

    def _messages(self, layer, sources, previous_out, hidden, memory, capture):
        context = previous_out[:, 0] + self._fresh(sources)
        hub, hub_weights = self._hub_message(layer, sources, context, hidden, memory)
        comm = self.comm[layer]
        logits = comm.pi_route_logits[1:].masked_fill(~comm.allowed[1:], float('-inf'))
        peripheral_weights = logits.softmax(-1).unsqueeze(0).expand(sources.shape[0], -1, -1)
        peripheral = torch.bmm(peripheral_weights, sources).transpose(0, 1)
        info = {}
        if capture:
            info['attn_weights'] = torch.cat((hub_weights.unsqueeze(1), peripheral_weights), dim=1).detach().mean(0)
        return torch.cat((hub, peripheral), dim=0), context, info

    def _rotate(self, layer, hidden):
        rows = list(hidden.unbind(0))
        angles = self.rotation_angles[layer].index_select(-1, self.rotation_indices)
        angles = torch.cat((angles, angles), dim=-1)
        for column in range(1, self.n_columns):
            cosine, sine = angles[column - 1].cos(), angles[column - 1].sin()
            hub, peripheral = rows[0], rows[column]
            rows[0] = cosine * hub + sine * peripheral
            rows[column] = -sine * hub + cosine * peripheral
        return torch.stack(rows)

    def _associative_step(self, layer, message, hidden, memory, context):
        lru_message = message.index_select(0, self.lru_columns)
        next_h = self._advance_cells(layer, lru_message, hidden)
        lru_out = self._project_cells(layer, lru_message, next_h)
        write = self.memory_writes[layer](message[self.memory_column])
        key, value, rate = torch.split(write, (self.memory_key_size, self.memory_value_size, 1), dim=-1)
        key = F.normalize(key, dim=-1, eps=1e-6)
        next_memory = delta_write(memory, key, value.tanh(), rate.sigmoid())
        memory_out = F.silu(self._memory_read(layer, next_memory, context)) + message[self.memory_column]
        out = lru_out.new_zeros(lru_out.shape[0], self.n_columns, self.hidden_size)
        out = out.index_copy(1, self.lru_columns, lru_out)
        out = out.index_copy(1, self.lru_columns.new_tensor([self.memory_column]), memory_out.unsqueeze(1))
        return out, next_h, next_memory, rate.sigmoid()

    def forward(self, x, state, *, capture=False, **_):
        if x.shape[1] != 1:
            raise ValueError('Expected batch-first input with one token per step')
        sources = torch.cat((state['out'], x), dim=1)
        hidden, outputs, memories, info = [], [], [], defaultdict(list)
        for layer in range(self.n_layers):
            old_h = state['h'][layer]
            memory = state['memory'][layer] if self.mechanism == 'associative' else None
            message, context, comm_info = self._messages(layer, sources, state['outs'][layer], old_h, memory, capture)
            if self.mechanism == 'associative':
                out, next_h, next_memory, rate = self._associative_step(layer, message, old_h, memory, context)
                memories.append(next_memory)
                if capture:
                    info['memory_write_rate'].append(rate.detach().mean(0))
            else:
                recurrence_h = self._rotate(layer, old_h) if self.mechanism == 'rotations' else old_h
                next_h = self._advance_cells(layer, message, recurrence_h)
                if self.mechanism == 'write_gate':
                    control = torch.cat((state['outs'][layer, :, 0], self._fresh(sources)), dim=-1)
                    peripheral_gate = self.write_gates[layer](control).sigmoid().transpose(0, 1).unsqueeze(-1)
                    gate = torch.cat((torch.ones_like(peripheral_gate[:1]), peripheral_gate), dim=0)
                    next_h = torch.lerp(old_h, next_h, gate)
                    if capture:
                        info['write_gate'].append(gate.detach().mean(1))
                out = self._project_cells(layer, message, next_h)
                if self.mechanism == 'two_reads':
                    # Write peripherals once. Refine the hub from its original previous state.
                    fresh = self._fresh(sources)
                    second_sources = torch.cat((out, sources[:, self.n_columns:]), dim=1)
                    second_message, second_weights = self._hub_message(layer, second_sources, out[:, 0] + fresh, old_h)
                    hub_h = self._advance_cells(layer, second_message, old_h[:1], self.hub_column)
                    hub_out = self._project_cells(layer, second_message, hub_h, self.hub_column)
                    next_h = torch.cat((hub_h, next_h[1:]), dim=0)
                    out = torch.cat((hub_out, out[:, 1:]), dim=1)
                    if capture:
                        info['second_attn_weights'].append(second_weights.detach().mean(0))
            hidden.append(next_h)
            outputs.append(out)
            for key, value in comm_info.items():
                info[key].append(value)
            sources = F.rms_norm(out, (self.hidden_size,)) if self.bounded_feedback else out
        hidden, outputs = torch.stack(hidden), torch.stack(outputs)
        feedback = F.rms_norm(outputs[-1], (self.hidden_size,)) if self.fb_norm else outputs[-1]
        new_state = {'h': hidden, 'outs': outputs, 'out': feedback}
        if self.mechanism == 'associative':
            new_state['memory'] = torch.stack(memories)
        return outputs[-1][:, 0], new_state, info

    def init_state(self, bsz):
        if self.mechanism != 'associative':
            return super().init_state(bsz)
        small = 0.01 / math.sqrt(self.hidden_size)
        # Allocate only the remaining LRUs; the replaced column has no hidden-state placeholder.
        h = torch.empty(self.n_layers, self.n_columns - 1, bsz, 2 * self.hidden_size, device=self.device, dtype=self.dtype).normal_(0, small)
        outs = torch.empty(self.n_layers, bsz, self.n_columns, self.hidden_size, device=self.device, dtype=self.dtype).normal_(0, small)
        memory = h.new_zeros(self.n_layers, bsz, self.memory_key_size, self.memory_value_size)
        return {'h': h, 'outs': outs, 'out': outs[-1], 'memory': memory}

    def reset_state(self, state=None, reset_mask=None, *, bsz=None):
        if state is None:
            return self.init_state(reset_mask.shape[0] if reset_mask is not None else bsz)
        keep = ~reset_mask.flatten()
        h = state['h'] * keep[None, None, :, None]
        outs = state['outs'] * keep[None, :, None, None]
        feedback = F.rms_norm(outs[-1], (self.hidden_size,)) if self.fb_norm else outs[-1]
        new_state = {'h': h, 'outs': outs, 'out': feedback}
        if self.mechanism == 'associative':
            new_state['memory'] = state['memory'] * keep[None, :, None, None]
        return new_state

    def carried_state_floats(self):
        recurrent_columns = self.n_columns - (self.mechanism == 'associative')
        hidden = 2 * self.n_layers * recurrent_columns * self.hidden_size
        feedback = self.n_columns * self.hidden_size
        hub_cache = (self.n_layers if self.fb_norm else self.n_layers - 1) * self.hidden_size
        memory = self.n_layers * self.memory_key_size * self.memory_value_size if self.mechanism == 'associative' else 0
        return hidden + feedback + hub_cache + memory

    def inspection_hidden(self, state):
        # A matrix memory has no H-wide LRU hidden state: compare all public messages.
        if self.mechanism == 'associative':
            return state['outs'].transpose(1, 2)
        return state['h']
