from __future__ import annotations

import math
import torch
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch


class HopfieldMessageLayer(nn.Module):
    """Modern Hopfield message passing between columns (Ramsauer et al. 2020)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        # learnable beta per head; init equivalent to standard 1/sqrt(d_k) scaling
        init_log_beta = math.log(1.0 / math.sqrt(self.head_dim))
        self.log_beta = nn.Parameter(torch.full((num_heads,), init_log_beta))
        self.norm = nn.LayerNorm(dim)

        # small init so message is negligible at start of training
        nn.init.normal_(self.out_proj.weight, 0.0, 0.001)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [cols, B, dim]
        C, B, D = h.shape
        # projections: [heads, B, cols, head_dim]
        q = self.W_q(h).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        k = self.W_k(h).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        v = self.W_v(h).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)

        beta = self.log_beta.exp().view(self.num_heads, 1, 1, 1)
        attn = torch.softmax(beta * torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        out = torch.matmul(attn, v)                                    # [heads, B, cols, head_dim]
        out = out.permute(2, 1, 0, 3).contiguous().view(C, B, D)      # [cols, B, dim]
        return self.norm(self.out_proj(out))


class HopfieldGridRnn(nn.Module):
    """Grid RNN with LSTM cells and Modern Hopfield message passing."""

    def __init__(
        self,
        *,
        input_size: int,
        embedding_size: int,
        output_size: int,
        hidden_size: int,
        n_layers: int,
        n_columns: int,
        n_attn_heads: int,
        messaging: str = "post",
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
        self.use_postmsg = (messaging == "post")

        print(
            f'HopfieldGridRNN {n_layers}L x {n_columns}C LSTM'
            f' hidden={self.hidden_size}'
        )

        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.attn_gates = nn.ModuleList()

        for layer in range(n_layers):
            self.cells.append(nn.ModuleList([
                nn.LSTMCell(
                    input_size=self._cell_input_dim(layer, icol),
                    hidden_size=self.hidden_size,
                    bias=use_bias,
                )
                for icol in range(n_columns)
            ]))
            self.attn.append(HopfieldMessageLayer(self.hidden_size, num_heads=n_attn_heads))
            if self.use_postmsg:
                self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

        self.head = nn.Linear(self.hidden_size, output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(self, tokens: torch.Tensor, state=None):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2

        x = self.embedding(tokens.view(-1))   # [B, embedding_size]
        h, c = state

        if self.use_postmsg:
            h, c = self.grid_step_postmsg(x, h=h, c=c)
        else:
            h, c = self.grid_step_premsg(x, h=h, c=c)

        y = self.head(h[-1][0])   # top layer, first col
        return y, (h, c)

    def grid_step_postmsg(self, x, *, h, c):
        h_n, c_n = [], []
        x_list = self._prepare_grid_input(x)

        for cells, attn, attn_gate, hl, cl in zip(
            self.cells, self.attn, self.attn_gates, h, c
        ):
            hl_cols, cl_cols = [], []
            for ic in range(self.n_columns):
                h_ic, c_ic = cells[ic](x_list[ic], (hl[ic], cl[ic]))
                hl_cols.append(h_ic)
                cl_cols.append(c_ic)

            hl_new = torch.stack(hl_cols, dim=0)   # [cols, B, H]
            cl_new = torch.stack(cl_cols, dim=0)

            msg = attn(hl_new)
            g = torch.sigmoid(attn_gate(torch.cat([hl_new, msg], dim=-1)))
            hl_new = (1 - g) * hl_new + g * msg

            h_n.append(hl_new)
            c_n.append(cl_new)
            x_list = hl_new

        return torch.stack(h_n, dim=0), torch.stack(c_n, dim=0)

    def grid_step_premsg(self, x, *, h, c):
        h_n, c_n = [], []
        x_list = self._prepare_grid_input(x)
        first_row = True

        for cells, attn, hl, cl in zip(self.cells, self.attn, h, c):
            msg = attn(hl)
            if first_row:
                x_list = [torch.cat([xc, msgc], dim=-1) for xc, msgc in zip(x_list, msg)]
            else:
                x_list = torch.cat([x_list, msg], dim=-1)

            hl_cols, cl_cols = [], []
            for ic in range(self.n_columns):
                h_ic, c_ic = cells[ic](x_list[ic], (hl[ic], cl[ic]))
                hl_cols.append(h_ic)
                cl_cols.append(c_ic)

            hl_new = torch.stack(hl_cols, dim=0)
            cl_new = torch.stack(cl_cols, dim=0)

            h_n.append(hl_new)
            c_n.append(cl_new)
            x_list = hl_new
            first_row = False

        return torch.stack(h_n, dim=0), torch.stack(c_n, dim=0)

    def _cell_input_dim(self, ix_layer: int, ix_col: int) -> int:
        if ix_layer == 0:
            return self.embedding_size if ix_col == 0 else 1
        hsz = self.hidden_size
        if not self.use_postmsg:
            hsz += self.hidden_size
        return hsz

    def _prepare_grid_input(self, x: torch.Tensor) -> list:
        bsz, _ = x.shape
        dummy = torch.zeros(bsz, self._cell_input_dim(0, 1), device=x.device, dtype=x.dtype)
        return [x] + [dummy] * (self.n_columns - 1)

    def reset_state(self, state, reset_mask):
        if state is None:
            return self.init_state(reset_mask.shape[0])
        ixs = torch.nonzero(reset_mask).flatten()
        if ixs.numel() == 0:
            return state
        h, c = state

        def _reset(t):
            t = t.clone()
            t[:, :, ixs, :] *= 0.0
            return t

        return (_reset(h), _reset(c))

    def detach_state(self, state):
        if state is None:
            return state
        h, c = state
        return (h.detach(), c.detach())

    def init_state(self, bsz: int):
        # [layers, cols, batch, hidden]
        shape = (self.n_layers, self.n_columns, bsz, self.hidden_size)
        device = self.head.weight.device
        dtype = self.head.weight.dtype
        return (
            torch.zeros(*shape, device=device, dtype=dtype),
            torch.zeros(*shape, device=device, dtype=dtype),
        )
