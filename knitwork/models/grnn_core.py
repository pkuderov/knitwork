from __future__ import annotations

import torch
from torch import nn

from knitwork.common.utils import format_readable_num


class GridRnn(nn.Module):
    def __init__(
            self, *,
            hidden_size, n_layers: int, n_columns: int,
            n_inputs: int = 1, n_outputs: int = 1,
            n_attn_heads, use_bias = True,
            self_feeding: bool = False,
            dtype, device
    ):
        super().__init__()
        assert n_columns > 1
        assert 0 < n_inputs <= n_columns
        assert 0 < n_outputs <= n_columns

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_columns = n_columns
        self.n_attn_heads = n_attn_heads
        self.self_feeding = self_feeding

        self.dtype = dtype
        self.device = device

        self.input_dim = (n_inputs, hidden_size)
        self.output_dim = (n_outputs, hidden_size)

        # Hidden size should be a multiply of the n_attn_heads
        self.hidden_size -= self.hidden_size % self.n_attn_heads
        print(
            f'GridRNN of {self.n_layers}L x {self.n_columns}C GRU cells'
            f' w/ {self.hidden_size} hidden units'
        )

        # Build a grid of cells: layers x columns
        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
        # used only for the post-messaging
        self.attn_gates = nn.ModuleList()
        for layer in range(self.n_layers):
            row = nn.ModuleList(
                nn.GRUCell(
                    input_size=self.hidden_size, hidden_size=self.hidden_size,
                    bias=use_bias, dtype=dtype
                )
                for icol in range(self.n_columns)
            )
            comm = MessagePassingLayer(
                self.hidden_size, num_heads=self.n_attn_heads, n_participants=self.n_columns
            )
            self.cells.append(row)
            self.attn.append(comm)
            
            self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(self, x: torch.Tensor, state: dict, *, out_attn=False, **_):
        # x shape: (n_inputs, batch, hidden_size)
        assert x.shape[0] == self.n_inputs
        # h shape: (layers, cols, batch, hidden_size)
        h = state['h']

        h_new, attn_ws, gate_vs = [], [], []
        x = self._prepare_grid_input(x, h)

        for layer in range(self.n_layers):
            hl_n = [
                self.cell_forward(self.cells[layer], x, h[layer], ix_col=ix_col)
                for ix_col in range(self.n_columns)
            ]
            hl_n = torch.stack(hl_n, dim=0)

            msg, attn_w = self.attn[layer](hl_n, return_weights=out_attn)
            g = torch.sigmoid(self.attn_gates[layer](
                torch.cat([hl_n, msg], dim=-1)
            ))
            hl_n = torch.lerp(hl_n, msg, g)

            h_new.append(hl_n)
            attn_ws.append(attn_w)
            gate_vs.append(g.detach())
            x = hl_n

        h_new = torch.stack(h_new, dim=0)

        # top (=last) layer, first col as grid output
        y = h[-1][0]
        state = {'h': h_new}
        info = {"attn_weights": attn_ws, "gates": gate_vs} 

        return y, state, info

    def grid_step(self, x, *, h: torch.Tensor, return_attn=True):
        h_n, attn_ws, gate_vs = [], [], []
        # it is a list of inputs, each input is [batch, col_in_dim]
        x = self._prepare_grid_input(x)

        for layer in range(self.n_layers):
            hl_n = [
                self.cell_forward(self.cells[layer], x, h[layer], ix_col=ix_col)
                for ix_col in range(self.n_columns)
            ]
            hl_n = torch.stack(hl_n, dim=0)

            msg, attn_w = self.attn[layer](hl_n, return_weights=return_attn)
            g = torch.sigmoid(self.attn_gates[layer](
                torch.cat([hl_n, msg], dim=-1)
            ))
            hl_n = torch.lerp(hl_n, msg, g)

            h_n.append(hl_n)
            attn_ws.append(attn_w)
            gate_vs.append(g.detach())
            # starting from there, x is a contiguous tensor [cols, batch, hidden_size]
            x = hl_n

        h_n = torch.stack(h_n, dim=0)
        info = {"attn_weights": attn_ws, "gates": gate_vs} 

        return h_n, info

    def cell_forward(self, cells, x, h, *, ix_col):
        cells, x, h = cells[ix_col], x[ix_col], h[ix_col]
        return cells(x, h)

    def _cell_input_dim(self, ix_layer: int, ix_col) -> int:
        if ix_layer == 0:
            # only the first col gets non-empty external input, 
            # the others get dummy 1-dim zero tensor
            return self.embedding_size if ix_col == 0 else 1

        hsz = self.hidden_size
        if not self.use_postmsg:
            # RNN input: [x; h_mix]
            hsz += self.hidden_size
        return hsz

    def _prepare_grid_input(self, x: torch.Tensor, h: torch.Tensor):
        if self.self_feeding:
            internal_input = h[-1, :, self.n_inputs:]
        else:
            n_internal_cols = self.n_columns - self.n_inputs
            internal_input = x.new_zeros(n_internal_cols, *x.shape[1:])

        # (col, batch, features), cat over cols
        x = torch.cat([x, internal_input], dim=0)
        return x

    def reset_state(self, state=None, *, reset_mask=None, bsz=None):
        if state is None:
            bsz = reset_mask.shape[0] if reset_mask is not None else bsz
            return self.init_state(bsz)

        keep = (~reset_mask.flatten())[None, None, :, None]
        h = state['h'] * keep
        return {'h': h}

    def detach_state(self, state):
        if state is None:
            return state
        return {'h': state['h'].detach()}

    def init_state(self, bsz):
        h = torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.device, dtype=self.dtype
        )
        return {'h': h}


class MessagePassingLayer(nn.Module):
    def __init__(self, dim, num_heads, n_participants=None):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=False)
        self.norm = nn.LayerNorm(dim)

        xavier_alpha = (1 / dim) ** 0.5
        # learnable identities "bias" to distinguish self-attention participants
        self.ids = None
        if n_participants is not None:
            # (col, batch, dim)
            self.ids = nn.Parameter(torch.empty(n_participants, 1, dim))
            # init them with different near-zero vectors
            nn.init.normal_(self.ids, 0.0, 0.01 * xavier_alpha)

        # Set very small out_proj to make the initial "message" negligible
        nn.init.normal_(self.mha.out_proj.weight, 0.0, 0.01 * xavier_alpha)
        nn.init.zeros_(self.mha.out_proj.bias)

    def forward(self, h, return_weights: bool = False):
        # h: (cols, batch, dim)
        qh, kh, vh = h, h, h
        if self.ids is not None:
            qh = kh = qh + self.ids

        h_mixed, attn_w = self.mha(qh, kh, vh, need_weights=return_weights, average_attn_weights=True)

        # Layer norm ensures we are in a good range
        if return_weights:
            attn_w = attn_w.detach().mean(dim=0)
        return self.norm(h_mixed), attn_w
