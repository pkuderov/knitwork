from __future__ import annotations

import math
import torch
from torch import nn

from knitwork.common.utils import convert_hidden_size, format_readable_num, to_torch


class HopfieldMessageLayer(nn.Module):
    """
    I test variant with change Attention layer on Modern Hopfield layer from article
    https://arxiv.org/pdf/2008.02217
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        init_log_beta = math.log(1.0 / math.sqrt(self.head_dim))
        self.log_beta = nn.Parameter(torch.full((num_heads,), init_log_beta)) # one b for one head

        self.norm = nn.LayerNorm(dim)

        nn.init.normal_(self.out_proj.weight, 0.0, 0.001)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (n_columns, batch, dim) — hidden states of all columns in one layer
        Returns:
            msg: (n_columns, batch, dim) — Hopfield-retrieved messages
        """
        C, B, D = h.shape

        q = self.W_q(h) 
        k = self.W_k(h)
        v = self.W_v(h)

        # Reshape for multi-head: (C, B, D) → (C, B, heads, d_k) → (heads, B, C, d_k)
        q = q.view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        k = k.view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        v = v.view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)

        # β-scaled attention scores
        # β = exp(log_beta), shape: (heads,) → (heads, 1, 1, 1)
        beta = self.log_beta.exp().view(self.num_heads, 1, 1, 1)
        scores = beta * torch.matmul(q, k.transpose(-2, -1))  # (heads, B, C, C)

        attn = torch.softmax(scores, dim=-1)  # (heads, B, C, C)

        out = torch.matmul(attn, v)  # (heads, B, C, d_k)

        out = out.permute(2, 1, 0, 3).contiguous().view(C, B, D)

        out = self.out_proj(out)
        return self.norm(out)


class HopfieldGridRnn(nn.Module):
    """
    All logic stand recent, but change gru to lstm cells so was in original article and
    change MessageLayer on new.
    """

    def __init__(
            self, *,
            input_size, embedding_size, output_size,
            hidden_size=None, base_hidden_size=None,
            n_layers: int, n_columns: int,
            n_attn_heads, messaging: str = "post",
            use_bias=True, dropout=0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.embedding = nn.Embedding(input_size, self.embedding_size)

        self.n_layers = n_layers
        assert n_columns > 1
        self.n_columns = n_columns
        self.n_attn_heads = n_attn_heads
        self.base_hidden_size = base_hidden_size

        if hidden_size is not None:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = convert_hidden_size(
                base_hid_dim=self.base_hidden_size,
                in_dim=self.embedding_size, out_dim=self.output_size,
                n_layers=self.n_layers, n_columns=self.n_columns,
                cell='lstm', type='grnn'
            )

        self.hidden_size -= self.hidden_size % self.n_attn_heads
        print(
            f'HopfieldGridRNN of {self.n_layers}L x {self.n_columns}C LSTM cells'
            f' w/ {self.hidden_size} hidden units'
        )

        self.use_postmsg = messaging == "post"

        
        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.attn_gates = nn.ModuleList()

        for layer in range(self.n_layers):
            row = nn.ModuleList([
                nn.LSTMCell(
                    input_size=self._cell_input_dim(layer, icol),
                    hidden_size=self.hidden_size, bias=use_bias,
                )
                for icol in range(self.n_columns)
            ])
            self.cells.append(row)
            self.attn.append(
                HopfieldMessageLayer(self.hidden_size, num_heads=self.n_attn_heads)
            )
            if self.use_postmsg:
                self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

        self.head = nn.Linear(self.hidden_size, self.output_size)

        #count of parameters
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(self, tokens: torch.Tensor, state=None):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2
        bsz, n_features = tokens.shape

        x = self.embedding(tokens.view(-1))

        h, c = state

        if self.use_postmsg:
            h, c = self.grid_step_postmsg(x, h=h, c=c)
        else:
            h, c = self.grid_step_premsg(x, h=h, c=c)

        z = h[-1][0]
        y = self.head(z)
        return y, (h, c)

    def grid_step_postmsg(self, x, *, h, c):
        h_n = []
        c_n = []
        x_list = self._prepare_grid_input(x)

        for cells, attn, attn_gate, hl, cl in zip(
            self.cells, self.attn, self.attn_gates, h, c
        ):
            #Independent LSTMCell step per column
            hl_new_cols = []
            cl_new_cols = []
            for ic in range(self.n_columns):
                h_ic, c_ic = cells[ic](x_list[ic], (hl[ic], cl[ic]))
                hl_new_cols.append(h_ic)
                cl_new_cols.append(c_ic)

            hl_new = torch.stack(hl_new_cols, dim=0)  # (cols, batch, hidden)
            cl_new = torch.stack(cl_new_cols, dim=0)

            msg = attn(hl_new)

            g = torch.sigmoid(attn_gate(
                torch.cat([hl_new, msg], dim=-1)
            ))
            hl_new = (1 - g) * hl_new + g * msg

            h_n.append(hl_new)
            c_n.append(cl_new)
            # Next layer input = h output of this layer
            x_list = hl_new

        return torch.stack(h_n, dim=0), torch.stack(c_n, dim=0)

    def grid_step_premsg(self, x, *, h, c):
        h_n = []
        c_n = []
        x_list = self._prepare_grid_input(x)
        first_row = True

        for cells, attn, hl, cl in zip(self.cells, self.attn, h, c):
            msg = attn(hl)
            if first_row:
                x_list = [
                    torch.cat([xc, msgc], -1)
                    for xc, msgc in zip(x_list, msg)
                ]
            else:
                x_list = torch.cat([x_list, msg], dim=-1)

            hl_new_cols = []
            cl_new_cols = []
            for ic in range(self.n_columns):
                h_ic, c_ic = cells[ic](x_list[ic], (hl[ic], cl[ic]))
                hl_new_cols.append(h_ic)
                cl_new_cols.append(c_ic)

            hl_new = torch.stack(hl_new_cols, dim=0)
            cl_new = torch.stack(cl_new_cols, dim=0)

            h_n.append(hl_new)
            c_n.append(cl_new)
            x_list = hl_new
            first_row = False

        return torch.stack(h_n, dim=0), torch.stack(c_n, dim=0)

    def cell_forward(self, cells, x, h, c, *, ix_col):
        return cells[ix_col](x[ix_col], (h[ix_col], c[ix_col]))

    def _cell_input_dim(self, ix_layer: int, ix_col) -> int:
        if ix_layer == 0:
            return self.embedding_size if ix_col == 0 else 1
        hsz = self.hidden_size
        if not self.use_postmsg:
            hsz += self.hidden_size
        return hsz

    def _prepare_grid_input(self, x):
        xl = [x]
        bsz, n_features = x.shape
        in_dim = self._cell_input_dim(ix_layer=0, ix_col=1)
        dummy_input = torch.zeros(bsz, in_dim, device=x.device, dtype=x.dtype)
        for _ in range(1, self.n_columns):
            xl.append(dummy_input)
        return xl

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

    def init_state(self, bsz):
        shape = (self.n_layers, self.n_columns, bsz, self.hidden_size)
        device = self.head.weight.device
        dtype = self.head.weight.dtype
        return (
            torch.zeros(*shape, device=device, dtype=dtype),
            torch.zeros(*shape, device=device, dtype=dtype),
        )