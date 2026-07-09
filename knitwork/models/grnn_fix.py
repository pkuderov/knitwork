from __future__ import annotations

import math

import torch
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch


class GridRnnFix(nn.Module):
    # GridRnn with fixed column attention: additive gated message (closed at init),
    # no post-norm, learnable beta, all-column inputs, protected recurrent state,
    # concat readout over top-layer columns. See architecture_analysis.md §7.1.
    def __init__(
            self, *,
            input_size, embedding_size, output_size,
            hidden_size,
            n_layers: int, n_columns: int,
            n_attn_heads,
            use_bias = True, dropout = 0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.embedding = nn.Embedding(input_size, embedding_size)

        self.n_layers = n_layers
        assert n_columns > 1
        self.n_columns = n_columns
        self.n_attn_heads = n_attn_heads
        self.hidden_size = hidden_size - hidden_size % n_attn_heads

        print(
            f'GridRnnFix {n_layers}L x {n_columns}C GRU'
            f' hidden={self.hidden_size}'
        )

        # symmetry breaking: every column gets the input through its own projection
        self.col_input_projs = nn.ModuleList(
            nn.Linear(embedding_size, embedding_size, bias=False)
            for _ in range(n_columns)
        )
        for proj in self.col_input_projs:
            nn.init.orthogonal_(proj.weight)

        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = embedding_size if layer == 0 else self.hidden_size
            self.cells.append(nn.ModuleList(
                nn.GRUCell(in_dim, self.hidden_size, bias=use_bias)
                for _ in range(n_columns)
            ))
            self.attn.append(ColumnAttention(self.hidden_size, num_heads=n_attn_heads))
            gate = nn.Linear(2 * self.hidden_size, 1)
            # closed gate at init: model opts INTO attention as it becomes useful
            nn.init.constant_(gate.bias, -3.0)
            self.attn_gates.append(gate)

        # concat readout over top-layer columns
        self.head = nn.Linear(self.n_columns * self.hidden_size, output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(self, tokens: torch.Tensor, h=None, return_attn=False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2
        x = self.embedding(tokens.view(-1))                      # [B, E]

        h_n, o_top, extras = self.grid_step(x, h=h, return_attn=return_attn)

        y = self.head(o_top)
        if return_attn:
            return y, h_n, extras
        return y, h_n

    def grid_step(self, x, *, h, return_attn=False):
        h_n, attn_list, gate_list = [], [], []
        # all columns receive the projected input at layer 0
        x = torch.stack([proj(x) for proj in self.col_input_projs], dim=0)  # [C, B, E]

        o = None
        for cells, attn, attn_gate, hl in zip(self.cells, self.attn, self.attn_gates, h):
            hl_n = torch.stack([
                cells[ic](x[ic], hl[ic]) for ic in range(self.n_columns)
            ], dim=0)                                            # [C, B, H]

            msg, attn_w = attn(hl_n, return_weights=return_attn)
            g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))
            # additive message; recurrent state stays unmixed (memory protection)
            o = hl_n + g * msg                                   # [C, B, H]

            h_n.append(hl_n)
            attn_list.append(attn_w)
            gate_list.append(g)
            x = o

        h_n = torch.stack(h_n, dim=0)
        o_top = o.permute(1, 0, 2).reshape(o.shape[1], -1)       # [B, C*H]
        return h_n, o_top, {"attn_weights": attn_list, "gates": gate_list}

    def reset_state(self, state, reset_mask):
        if state is None:
            return self.init_state(reset_mask.shape[0])
        ixs = torch.nonzero(reset_mask).flatten()
        if ixs.numel() == 0:
            return state
        state = state.clone()
        state[:, :, ixs, :] *= 0.0
        return state

    def detach_state(self, state):
        if state is None:
            return state
        return state.detach()

    def init_state(self, bsz):
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.head.weight.device, dtype=self.head.weight.dtype
        )


class ColumnAttention(nn.Module):
    # Hopfield-style column mixing: learnable beta per head, tiny out_proj, NO post-norm
    def __init__(self, dim, num_heads, beta_scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        # init = beta_scale/sqrt(d_k); beta_scale>1 starts sharper than standard attention
        self.log_beta = nn.Parameter(
            torch.full((num_heads,), math.log(beta_scale / math.sqrt(self.head_dim)))
        )
        # message really is negligible at init: no norm to re-inflate it
        nn.init.normal_(self.out_proj.weight, 0.0, 0.001)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h, return_weights: bool = False):
        # h: [C, B, D]
        C, B, D = h.shape
        q = self.W_q(h).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        k = self.W_k(h).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        v = self.W_v(h).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)

        beta = self.log_beta.exp().view(self.num_heads, 1, 1, 1)
        attn = torch.softmax(beta * torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        out = torch.matmul(attn, v)                              # [heads, B, C, hd]
        out = out.permute(2, 1, 0, 3).contiguous().view(C, B, D)
        attn_w = attn.mean(dim=(0, 1)) if return_weights else None
        return self.out_proj(out), attn_w
