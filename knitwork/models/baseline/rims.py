"""RIMs / BRIMs baselines (Goyal et al., ICLR 2021, arXiv 1909.10893; Mittal et al., ICML 2020, arXiv 2006.16981).

Compact step-wise reimplementation following the official RIMCell (dido1998/Recurrent-Independent-Mechanisms):
modules attend to [inputs, null]; the top-k modules by non-null attention update their LSTM state,
the others keep it; active modules then communicate via attention over all modules (residual).
BRIMs stack such layers and additionally feed each layer with the previous-step modules of the layer above.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class _GroupLinear(nn.Module):
    """Independent linear map per module: [B, n, i] -> [B, n, o]."""

    def __init__(self, n, i, o):
        super().__init__()
        bound = 1.0 / math.sqrt(i)
        self.w = nn.Parameter(torch.empty(n, i, o).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.zeros(n, o))

    def forward(self, x):
        return torch.einsum('bni,nio->bno', x, self.w) + self.b


class _RimLayer(nn.Module):
    def __init__(self, *, src_dims, n, h, k, d_key, d_val, n_comm_heads, d_comm):
        super().__init__()
        self.n, self.h, self.k, self.d_key = n, h, k, d_key
        self.n_comm_heads, self.d_comm = n_comm_heads, d_comm

        # input attention: modules (queries) over all sources and a null source
        self.key = nn.ModuleList([nn.Linear(d, d_key) for d in src_dims])
        self.val = nn.ModuleList([nn.Linear(d, d_val) for d in src_dims])
        self.null_k = nn.Parameter(torch.zeros(d_key))
        self.null_v = nn.Parameter(torch.zeros(d_val))
        self.q = _GroupLinear(n, h, d_key)

        # block-diagonal LSTM
        self.w_ih = _GroupLinear(n, d_val, 4 * h)
        self.w_hh = _GroupLinear(n, h, 4 * h)
        with torch.no_grad():
            self.w_ih.b[:, h:2 * h] = 1.0  # forget gate bias

        # communication attention among modules
        self.q2 = _GroupLinear(n, h, n_comm_heads * d_comm)
        self.k2 = _GroupLinear(n, h, n_comm_heads * d_comm)
        self.v2 = _GroupLinear(n, h, n_comm_heads * d_comm)
        self.out = _GroupLinear(n, n_comm_heads * d_comm, h)

    def forward(self, srcs, h, c):
        # srcs: list of [B, S_i, d_i]; h, c: [B, n, h]
        B = h.shape[0]
        keys = torch.cat([lin(s) for lin, s in zip(self.key, srcs)] + [self.null_k.expand(B, 1, -1)], dim=1)  # [B, S+1, dk]
        vals = torch.cat([lin(s) for lin, s in zip(self.val, srcs)] + [self.null_v.expand(B, 1, -1)], dim=1)  # [B, S+1, dv]
        att = (self.q(h) @ keys.transpose(1, 2) / math.sqrt(self.d_key)).softmax(-1)  # [B, n, S+1]

        # active modules: top-k by attention mass on real (non-null) sources
        idx = (1.0 - att[..., -1]).topk(self.k, dim=1).indices  # [B, k]
        mask = torch.zeros_like(h[..., 0]).scatter(1, idx, 1.0).unsqueeze(-1)  # [B, n, 1]
        inp = (att @ vals) * mask  # [B, n, dv]

        i, f, g, o = (self.w_ih(inp) + self.w_hh(h)).chunk(4, dim=-1)
        c_rnn = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h_rnn = o.sigmoid() * c_rnn.tanh()
        # gradients flow only into active modules
        h_g = mask * h_rnn + (1 - mask) * h_rnn.detach()

        nh, dc = self.n_comm_heads, self.d_comm
        split = lambda t: t.view(B, self.n, nh, dc).transpose(1, 2)  # [B, nh, n, dc]
        a = (split(self.q2(h_g)) @ split(self.k2(h_g)).transpose(-1, -2) / math.sqrt(dc)).softmax(-1)
        ctx = (a @ split(self.v2(h_g))).transpose(1, 2).reshape(B, self.n, nh * dc)
        h_comm = self.out(ctx) + h_g

        h_new = mask * h_comm + (1 - mask) * h
        c_new = mask * c_rnn + (1 - mask) * c
        return h_new, c_new


class RimsCore(nn.Module):
    """Feature-level RIMs core; hidden_size = n_modules * module_size."""
    has_attn = False
    top_down = False

    def __init__(
            self, *,
            n_modules, module_size, n_active, n_layers=1,
            d_key=32, d_comm=32, n_comm_heads=2,
            dtype, device,
    ):
        super().__init__()
        self.n_modules, self.module_size, self.n_layers = n_modules, module_size, n_layers
        self.hidden_size = n_modules * module_size
        self.dtype, self.device = dtype, device

        self.layers = nn.ModuleList()
        for li in range(n_layers):
            src_dims = [self.hidden_size if li == 0 else module_size]
            if self.top_down and li < n_layers - 1:
                src_dims.append(module_size)
            self.layers.append(_RimLayer(
                src_dims=src_dims, n=n_modules, h=module_size, k=n_active,
                d_key=d_key, d_val=module_size, n_comm_heads=n_comm_heads, d_comm=d_comm,
            ))
        print(
            f'{type(self).__name__[:-4].upper()} {n_layers}L, {n_modules} modules x {module_size}'
            f' ({n_active} active), hidden {self.hidden_size}'
        )

    def forward(self, x: torch.Tensor, state: dict, **_):
        assert x.shape[0] == 1
        x = x.squeeze(0)  # [B, H]
        if state is None:
            state = self.init_state(x.shape[0])
        h_old, c_old = state['h'], state['c']  # [L, B, n, h]

        hs, cs, below = [], [], x.unsqueeze(1)  # [B, 1, H]
        for li, layer in enumerate(self.layers):
            srcs = [below]
            if self.top_down and li < self.n_layers - 1:
                srcs.append(h_old[li + 1])  # previous-step modules of the layer above
            h, c = layer(srcs, h_old[li], c_old[li])
            hs.append(h)
            cs.append(c)
            below = h

        y = hs[-1].flatten(1)  # [B, n*h]
        return y, {'h': torch.stack(hs), 'c': torch.stack(cs)}, {}

    def reset_state(self, state=None, reset_mask=None, *, bsz=None):
        if state is None:
            bsz = reset_mask.shape[0] if reset_mask is not None else bsz
            return self.init_state(bsz)

        keep = (~reset_mask.flatten())[None, :, None, None]
        return {k: v * keep for k, v in state.items()}

    def detach_state(self, state):
        if state is None:
            return state
        return {k: v.detach() for k, v in state.items()}

    def init_state(self, bsz):
        z = torch.zeros(
            self.n_layers, bsz, self.n_modules, self.module_size,
            device=self.device, dtype=self.dtype,
        )
        return {'h': z, 'c': z.clone()}


class BrimsCore(RimsCore):
    """RIMs layers with bottom-up and top-down (previous-step) signals."""
    top_down = True
