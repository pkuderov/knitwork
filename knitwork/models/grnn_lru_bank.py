from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from knitwork.models.grnn_lru_core import LruBank


class BankRnn(nn.Module):
    """Multi-timescale LRU columns without recurrent routing, read by attention outside the recurrence."""
    batch_first = True  # step input is (B, 1, H)

    def __init__(
            self, *,
            hidden_size, n_layers, n_columns, horizon, n_attn_heads=4,
            column_horizons=True, write='all', write_key_size=32, write_heads=1, write_feedback=False,
            dtype, device,
    ):
        super().__init__()
        # write='all': every column receives the input; 'attn': columns compete for it by content (soft slots)
        assert write in ('all', 'attn')
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_columns = n_columns
        self.write = write
        # write_heads=n > 1: n write heads, each with its own value projection and allocation over columns
        # write_feedback: the write query also sees the previous read output (closes the loop through the state)
        assert write == 'attn' or (write_heads == 1 and not write_feedback)
        self.write_heads, self.write_feedback = write_heads, write_feedback
        self.dtype, self.device = dtype, device
        H, C = hidden_size, n_columns

        # column-local stacks: layer l of column c reads layer l - 1 of the same column only
        self.cells = LruBank(
            n_layers=n_layers, n_columns=n_columns, hidden_size=hidden_size, horizon=horizon,
            column_horizons=column_horizons,
        )
        if write == 'attn':
            n, K = write_heads, write_key_size
            self.write_q = nn.Linear(H, n * K)
            # zero key projection: uniform allocation at init; column ids break the symmetry later
            self.write_k = nn.Linear(H, n * K)
            nn.init.zeros_(self.write_k.weight)
            self.col_ids = nn.Parameter(torch.randn(C, 1, n * K) * 0.1)
            if write_feedback:
                self.write_qy = nn.Linear(H, n * K, bias=False)
                nn.init.zeros_(self.write_qy.weight)
            if write_heads > 1:
                self.write_v = nn.Linear(H, n * H)
        # read: the current input queries the input token and the top column outputs
        self.col_emb = nn.Parameter(torch.randn(C + 1, 1, H) * 0.02)
        self.read = nn.MultiheadAttention(H, n_attn_heads)
        self.ln = nn.LayerNorm(H)
        self.to(device=device, dtype=dtype)

    def forward(self, x, state, **_):
        # x: (B, 1, H)
        x = x[:, 0]
        h_tp, outs_tp, y_tp = state['h'], state['outs'], state['y']
        C = self.n_columns
        # (C, B, H) layer-0 input
        inp = x[None].expand(C, -1, -1)
        info = {}
        if self.write == 'attn':
            n, B = self.write_heads, x.shape[0]
            # (C, B, n, K) keys from what each column last emitted
            k = (self.write_k(outs_tp[-1].transpose(0, 1)) + self.col_ids).view(C, B, n, -1)
            q = self.write_q(x) + (self.write_qy(y_tp) if self.write_feedback else 0.0)
            q = q.view(1, B, n, -1)
            # (C, B, n) allocation over columns per head, scaled so the uniform split keeps the input scale
            w = torch.softmax((q * k).sum(-1) / math.sqrt(k.shape[-1]), dim=0) * C
            if n == 1:
                inp = inp * w
            else:
                # (B, n, H) per-head values, each routed to columns by its own allocation
                v = self.write_v(x).view(B, n, -1)
                inp = torch.einsum('cbn,bnh->cbh', w, v) / n
            info['write_entropy'] = [-(w / C * (w / C).clamp_min(1e-9).log()).sum(0).mean() / math.log(C)]

        h_t, outs_t = [], []
        for layer in range(self.n_layers):
            # out: (B, C, H), h: (C, B, 2H)
            out, hl = self.cells(layer, inp, h_tp[layer])
            h_t.append(hl)
            outs_t.append(out)
            inp = out.transpose(0, 1)

        # (C + 1, B, H) tokens: input then top columns, with position embeddings
        tokens = torch.cat([x[None], outs_t[-1].transpose(0, 1)]) + self.col_emb
        read, _ = self.read(tokens[:1], tokens, tokens, need_weights=False)
        y = self.ln(x + read[0])
        state = {'h': torch.stack(h_t), 'outs': torch.stack(outs_t), 'y': y}
        return y, state, info

    def init_state(self, bsz):
        L, B, C, H = self.n_layers, bsz, self.n_columns, self.hidden_size
        h = torch.zeros(L, C, B, 2 * H, device=self.device, dtype=self.dtype)
        outs = h.new_zeros(L, B, C, H)
        return {'h': h, 'outs': outs, 'y': h.new_zeros(B, H)}

    def reset_state(self, state=None, reset_mask=None, *, bsz=None):
        if state is None:
            bsz = reset_mask.shape[0] if reset_mask is not None else bsz
            return self.init_state(bsz)
        keep = ~reset_mask.flatten()
        return {
            'h': state['h'] * keep[None, None, :, None], 'outs': state['outs'] * keep[None, :, None, None],
            'y': state['y'] * keep[:, None],
        }

    def detach_state(self, state):
        if state is None:
            return state
        return {key: value.detach() for key, value in state.items()}
