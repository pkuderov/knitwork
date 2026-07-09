from __future__ import annotations

import torch
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch
from knitwork.models.grnn_fix import ColumnAttention


class HopfieldGridRnnFix(nn.Module):
    # HopfieldGridRnn with fixed attention: additive gated message (closed at init),
    # no post-norm, all-column inputs, concat readout. LSTM cell state c stays
    # protected from messages. See architecture_analysis.md §7.1.
    def __init__(
        self, *,
        input_size: int,
        embedding_size: int,
        output_size: int,
        hidden_size: int,
        n_layers: int,
        n_columns: int,
        n_attn_heads: int,
        use_bias: bool = True,
        dropout: float = 0.0,
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
            f'HopfieldGridRnnFix {n_layers}L x {n_columns}C LSTM'
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
                nn.LSTMCell(in_dim, self.hidden_size, bias=use_bias)
                for _ in range(n_columns)
            ))
            self.attn.append(ColumnAttention(self.hidden_size, num_heads=n_attn_heads))
            gate = nn.Linear(2 * self.hidden_size, 1)
            nn.init.constant_(gate.bias, -3.0)
            self.attn_gates.append(gate)

        self.head = nn.Linear(self.n_columns * self.hidden_size, output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(self, tokens: torch.Tensor, state=None, return_attn: bool = False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2
        x = self.embedding(tokens.view(-1))                      # [B, E]
        h, c = state

        h_n, c_n = [], []
        x = torch.stack([proj(x) for proj in self.col_input_projs], dim=0)  # [C, B, E]

        hl_mix = None
        for cells, attn, attn_gate, hl, cl in zip(
            self.cells, self.attn, self.attn_gates, h, c
        ):
            hl_cols, cl_cols = [], []
            for ic in range(self.n_columns):
                h_ic, c_ic = cells[ic](x[ic], (hl[ic], cl[ic]))
                hl_cols.append(h_ic)
                cl_cols.append(c_ic)
            hl_new = torch.stack(hl_cols, dim=0)                 # [C, B, H]
            cl_new = torch.stack(cl_cols, dim=0)

            msg, _ = attn(hl_new)
            g = torch.sigmoid(attn_gate(torch.cat([hl_new, msg], dim=-1)))
            # additive message into working state h; memory c stays untouched
            hl_mix = hl_new + g * msg

            h_n.append(hl_mix)
            c_n.append(cl_new)
            x = hl_mix

        h_n = torch.stack(h_n, dim=0)
        c_n = torch.stack(c_n, dim=0)

        z = hl_mix.permute(1, 0, 2).reshape(hl_mix.shape[1], -1)  # [B, C*H]
        y = self.head(z)
        if return_attn:
            return y, (h_n, c_n), {}
        return y, (h_n, c_n)

    def reset_state(self, state, reset_mask):
        if state is None:
            return self.init_state(reset_mask.shape[0])
        h, c = state
        keep = (~reset_mask.bool()).to(dtype=h.dtype, device=h.device)  # [B]
        keep = keep[None, None, :, None]
        return (h * keep, c * keep)

    def detach_state(self, state):
        if state is None:
            return state
        h, c = state
        return (h.detach(), c.detach())

    def init_state(self, bsz: int):
        shape = (self.n_layers, self.n_columns, bsz, self.hidden_size)
        device = self.head.weight.device
        dtype = self.head.weight.dtype
        return (
            torch.zeros(*shape, device=device, dtype=dtype),
            torch.zeros(*shape, device=device, dtype=dtype),
        )
