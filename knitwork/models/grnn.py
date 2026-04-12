from __future__ import annotations

import torch
from torch import nn

from knitwork.common.utils import convert_hidden_size, format_readable_num, to_torch


class GridRnn(nn.Module):
    def __init__(
            self, *,
            input_size, embedding_size, output_size,
            hidden_size = None, base_hidden_size = None,
            n_layers: int, n_columns: int,
            n_attn_heads, messaging: str = "post", col_identities,
            
            use_bias = True, dropout = 0.0
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
                n_layers=self.n_layers, n_columns=self.n_columns, type='grnn'
            )
        # Hidden size should be a multiply of the n_attn_heads
        self.hidden_size -= self.hidden_size % self.n_attn_heads
        print(
            f'GridRNN of {self.n_layers}L x {self.n_columns}C GRU cells'
            f' w/ {self.hidden_size} hidden units'
        )

        # pre- or post- messaging, i.e. when attention is applied
        self.use_postmsg = messaging == "post"

        # Build a grid of cells: layers x columns
        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
        # used only for the post-messaging
        self.attn_gates = nn.ModuleList()
        for layer in range(self.n_layers):
            row = (
                nn.GRUCell(
                    input_size=self._cell_input_dim(layer, icol), 
                    hidden_size=self.hidden_size, bias=use_bias,
                    dtype=torch.float64
                )
                for icol in range(self.n_columns)
            )
            self.cells.append(nn.ModuleList(row))

            n_participants = self.n_columns if col_identities else None
            self.attn.append(MessagePassingLayer(
                self.hidden_size, num_heads=self.n_attn_heads, n_participants=n_participants
            ))
            
            if self.use_postmsg:
                self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

        # Head reads from the top layer, 0-th column (the external column)
        self.head = nn.Linear(self.hidden_size, self.output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(self, tokens: torch.Tensor, h=None, return_attn=False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2
        bsz, n_features = tokens.shape
        assert n_features == 1, "expected input features dimension = 1 (token ids)"

        x = self.embedding(tokens.view(-1))

        # shape: (layers, cols, batch, hidden_size)
        #h = self.grid_step_postmsg(x, h=h) if self.use_postmsg else self.grid_step_premsg(x, h=h)
        # top (=last) layer, first col as grid output
        
        if self.use_postmsg:
            h, extras = self.grid_step_postmsg(x, h=h, return_attn=return_attn)
        else:
            h, extras = self.grid_step_premsg(x, h=h), {}
                      
        z = h[-1][0]

        y = self.head(z)
        if return_attn:
            return y, h, extras
        return y, h

    def grid_step_postmsg(self, x, *, h: torch.Tensor, return_attn=True):
        h_n, attn_list, gate_list = [], [], []
        # it is a list of inputs, each input is [batch, col_in_dim]
        x = self._prepare_grid_input(x)

        for cells, attn, attn_gate, hl in zip(self.cells, self.attn, self.attn_gates, h):
            hl_n = [
                self.cell_forward(cells, x, hl, ix_col=ix_col)
                for ix_col in range(self.n_columns)
            ]
            hl_n = torch.stack(hl_n, dim=0)

            msg, attn_w = attn(hl_n, return_weights=return_attn)
            g = torch.sigmoid(attn_gate(
                torch.cat([hl_n, msg], dim=-1)
            ))
            hl_n = (1 - g) * hl_n + g * msg

            h_n.append(hl_n)
            attn_list.append(attn_w)
            gate_list.append(g)   
            # starting from there, x is a contiguous tensor [cols, batch, hidden_size]
            x = hl_n

        h_n = torch.stack(h_n, dim=0)
        return h_n, {"attn_weights": attn_list, "gates": gate_list} 

    def grid_step_premsg(self, x, *, h: torch.Tensor):
        h_n = []
        # it is a list of inputs, each input is [batch, col_in_dim]
        x = self._prepare_grid_input(x)
        first_row = True

        for cells, attn, hl in zip(self.cells, self.attn, h):
            msg, _ = attn(hl, return_weights=False)
            if first_row:
                # a list, not a contiguous tensor
                x = [
                    torch.cat([xc, msgc], -1)
                    for xc, msgc in zip(x, msg)
                ]
            else:
                # a contiguous tensor
                x = torch.cat([x, msg], dim=-1) # type: ignore

            hl_n = [
                self.cell_forward(cells, x, hl, ix_col=ix_col)
                for ix_col in range(self.n_columns)
            ]
            hl_n = torch.stack(hl_n, dim=0)

            h_n.append(hl_n)
            # starting from there, x is a contiguous tensor [cols, batch, hidden_size]
            x = hl_n
            first_row = False

        h_n = torch.stack(h_n, dim=0)
        return h_n

    def cell_forward(self, cells, x, h, *, ix_col):
        cells, x, h = cells[ix_col], x[ix_col], h[ix_col]
        return cells(x, h)

    def reset_state(self, state, reset_mask):
        if state is None:
            return self.init_state(reset_mask.shape[0])

        ixs = torch.nonzero(reset_mask).flatten()
        if ixs.numel() == 0:
            return state

        def _reset(h):
            # shape: (layers, cols, batch, hidden_size)
            h = h.clone()
            h[:, :, ixs, :] *= 0.0
            return h

        state = _reset(state)
        return state

    def detach_state(self, state):
        if state is None:
            return state
        return state.detach()

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

    def _prepare_grid_input(self, x):
        # the first "layer" (=row) may have input of different size,
        # therefore it is stored as list, not as contiguous stacked array [for the later rows]
        xl = [x]
        bsz, n_features = x.shape
        in_dim = self._cell_input_dim(ix_layer=0, ix_col=1)
        dummy_input = torch.zeros(bsz, in_dim, device=x.device, dtype=x.dtype)
        for _ in range(1, self.n_columns):
            xl.append(dummy_input)

        return xl

    def init_state(self, bsz):
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.head.weight.device, dtype=self.head.weight.dtype
        )


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

        h_mixed, attn_w  = self.mha(qh, kh, vh, average_attn_weights=True)

        # Layer norm ensures we are in a good range
        if return_weights and attn_w is not None:
            attn_w = attn_w.mean(dim=0)
        return self.norm(h_mixed), attn_w 