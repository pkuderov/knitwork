from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from knitwork.models.grnn_core import StochasticMessagePassingLayer
from knitwork.models.grnn_lru_core import LruBank


class ManyRnn(nn.Module):
    """One-layer grid of many cheap LRU columns (shared weights, per-column horizon bands) with structured communication."""
    batch_first = True  # step input is (B, 1, H)

    def __init__(
            self, *,
            hidden_size, n_columns, horizon, comm='full', n_attn_heads=4,
            active_k=0, active_balance=1e-3, route_key_size=32,
            n_slots=4, slot_topk=4, group_size=4,
            dtype, device,
    ):
        super().__init__()
        # comm='full': every column attends to all columns and the input (M1)
        # comm='workspace': columns compete for n_slots shared slots (top slot_topk sources each), slots are broadcast (M3)
        # comm='groups': full attention inside groups of group_size, the first column of each group is a hub that
        # also attends to the other hubs (M4)
        # active_k=k > 0: per step only column 0 and the k best-matching other columns update, the rest hold their
        # state without decay (M2); selection bias is balanced toward even load without an aux loss
        assert comm in ('full', 'workspace', 'groups')
        assert comm != 'groups' or n_columns % group_size == 0
        assert 0 <= active_k < n_columns
        self.hidden_size, self.n_columns = hidden_size, n_columns
        self.comm_mode, self.active_k, self.active_balance = comm, active_k, active_balance
        self.n_slots, self.group_size = n_slots, group_size
        self.dtype, self.device = dtype, device
        H, C = hidden_size, n_columns

        self.cells = LruBank(
            n_layers=1, n_columns=C, hidden_size=H, horizon=horizon, column_horizons=True, share_columns=True,
        )
        msg = lambda n_q, n_kv, top_k=None: StochasticMessagePassingLayer(
            H, num_heads=n_attn_heads, ln_msg=True, n_q=n_q, n_kv=n_kv, top_k=top_k,
        )
        if comm == 'full':
            self.comm = msg(C, C + 1)
        elif comm == 'workspace':
            self.ws_write = msg(n_slots, C + 1, top_k=slot_topk)
            # identity table sized for the C queries (StochasticMessagePassingLayer indexes q ids by the kv table)
            self.ws_read = msg(C, max(C, n_slots + 1))
        else:
            G = C // group_size
            self.comm = msg(group_size, group_size + 1)
            self.hub = msg(G, G)
        if active_k:
            K = route_key_size
            self.route_q = nn.Linear(H, K)
            # zero key projection: selection starts from the column ids and the balancing bias
            self.route_k = nn.Linear(H, K)
            nn.init.zeros_(self.route_k.weight)
            self.route_ids = nn.Parameter(torch.randn(C, 1, K) * 0.1)
            self.register_buffer('route_bias', torch.zeros(C, 1, 1))
            self.register_buffer('is_first', (torch.arange(C) == 0).view(C, 1, 1))
        self.to(device=device, dtype=dtype)

    def forward(self, x, state, **_):
        # x: (B, 1, H)
        xe = x[:, 0]
        B, C, H = xe.shape[0], self.n_columns, self.hidden_size
        # (C, B, H) queries from the previous outputs, sources from the normalised fed-back outputs
        q = state['outs'].transpose(0, 1)
        kv = state['out'].transpose(0, 1)
        xt = xe[None]
        new = {}
        if self.comm_mode == 'full':
            src = torch.cat([kv, xt])
            msg, _ = self.comm(q, src, src)
        elif self.comm_mode == 'workspace':
            # slots ask with their previous content, read the top sources: (S, B, H)
            src = torch.cat([kv, xt])
            ws, _ = self.ws_write(state['ws'], src, src)
            src = torch.cat([ws, xt])
            msg, _ = self.ws_read(q, src, src)
            new['ws'] = ws
        else:
            g = self.group_size
            G = C // g
            # (g, G * B, H): groups folded into the batch, the input joins every group
            fold = lambda t: t.view(G, g, B, H).transpose(0, 1).reshape(g, G * B, H)
            src = torch.cat([fold(kv), xt.repeat(1, G, 1)])
            msg, _ = self.comm(fold(q), src, src)
            msg = msg.view(g, G, B, H).transpose(0, 1)  # (G, g, B, H)
            # hubs (first column of each group) exchange across groups: (G, B, H)
            hmsg, _ = self.hub(q[::g], kv[::g], kv[::g])
            msg = torch.cat([msg[:, :1] + hmsg[:, None], msg[:, 1:]], dim=1).reshape(C, B, H)

        cell_in = msg + xt
        write = None
        if self.active_k:
            # (C, B, 1) match of the input against what each column last emitted
            k = self.route_k(q) + self.route_ids
            score = (self.route_q(xe)[None] * k).sum(-1, keepdim=True) / math.sqrt(k.shape[-1])
            # column 0 (the readout) always updates; the others compete for active_k places
            sel = torch.where(self.is_first, torch.full_like(score, 1e4), score.detach() + self.route_bias)
            mask = sel >= sel.topk(self.active_k + 1, dim=0).values[-1:]
            write = torch.where(self.is_first, torch.ones_like(score), torch.sigmoid(score + 2.0)) * mask
            # balance only in the update replay (grad on), so collection and replay see the same selection
            if self.training and self.active_balance and torch.is_grad_enabled():
                with torch.no_grad():
                    load = mask.float().mean(dim=(1, 2), keepdim=True)
                    self.route_bias += self.active_balance * torch.sign(self.active_k / (C - 1) - load)
        out, h = self.cells(0, cell_in, state['h'][0], write=write)
        y = out[:, 0]
        # key order must match init_state: CUDA graphs rebuild the state from a flat leaf list
        state = {'h': h[None], 'outs': out, 'out': F.rms_norm(out, (H,))} | new
        return y, state, {}

    def init_state(self, bsz):
        B, C, H = bsz, self.n_columns, self.hidden_size
        h = torch.zeros(1, C, B, 2 * H, device=self.device, dtype=self.dtype)
        state = {'h': h, 'outs': h.new_zeros(B, C, H), 'out': h.new_zeros(B, C, H)}
        if self.comm_mode == 'workspace':
            state['ws'] = h.new_zeros(self.n_slots, B, H)
        return state

    def reset_state(self, state=None, reset_mask=None, *, bsz=None):
        if state is None:
            bsz = reset_mask.shape[0] if reset_mask is not None else bsz
            return self.init_state(bsz)
        keep = ~reset_mask.flatten()
        new = {
            'h': state['h'] * keep[None, None, :, None],
            'outs': state['outs'] * keep[:, None, None], 'out': state['out'] * keep[:, None, None],
        }
        if 'ws' in state:
            new['ws'] = state['ws'] * keep[None, :, None]
        return new

    def detach_state(self, state):
        if state is None:
            return state
        return {key: value.detach() for key, value in state.items()}
