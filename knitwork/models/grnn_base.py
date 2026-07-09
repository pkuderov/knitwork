
"""GridRNN base — pure cascade grid without attention (Kalchbrenner et al., 2015).

Each column receives the previous column's output directly as input (no attention, no gating).
Within each layer columns are chained: col0 → col1 → col2 → ...
Between layers the last column feeds into col0 of the next layer.
All other inter-column columns within upper layers also cascade through the chain.
This matches the original Grid LSTM paper: each cell processes inputs from
the time dimension (h_prev same col) and the depth dimension (h_new prev col).
"""
from __future__ import annotations

import torch
from torch import nn

from knitwork.common.utils import format_readable_num


class GridRnnBase(nn.Module):
    """Grid RNN without inter-column attention.

    State shape: [n_layers, n_columns, B, hidden_size].
    Column cascade within each layer: col_i input = output of col_{i-1}.
    """

    def __init__(
        self,
        *,
        input_size: int,
        embedding_size: int,
        output_size: int,
        hidden_size: int,
        n_layers: int,
        n_columns: int,
        use_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_columns >= 1

        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.n_columns   = n_columns

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.drop      = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # grid cells: layers × columns
        # col 0 of layer 0: input_size = embedding_size
        # all other cells: input_size = hidden_size (receives prev col output)
        self.cells = nn.ModuleList()
        for layer in range(n_layers):
            row = nn.ModuleList()
            for col in range(n_columns):
                in_dim = embedding_size if (layer == 0 and col == 0) else hidden_size
                row.append(nn.GRUCell(in_dim, hidden_size, bias=use_bias))
            self.cells.append(row)

        self.head = nn.Linear(hidden_size, output_size)

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'GridRNN-Base {n_layers}L x {n_columns}C  H={hidden_size}  params={format_readable_num(n)}')

    def forward(self, tokens: torch.Tensor, h=None, return_attn: bool = False):
        # tokens: [B, 1]
        B = tokens.shape[0]
        x = self.drop(self.embedding(tokens.view(-1)))  # [B, embed]

        if h is None:
            h = self.init_state(B, tokens.device)

        layer_outputs: list[torch.Tensor] = []

        for layer in range(self.n_layers):
            col_input = x if layer == 0 else layer_outputs[-1][-1]
            col_outputs: list[torch.Tensor] = []
            for col in range(self.n_columns):
                h_prev = h[layer, col]                              # [B, H]
                h_col = self.cells[layer][col](col_input, h_prev)   # [B, H]
                col_outputs.append(h_col)
                col_input = h_col                                    # cascade
            layer_outputs.append(col_outputs)

        # stack into [n_layers, n_columns, B, H]
        h_new = torch.stack([torch.stack(cols, dim=0) for cols in layer_outputs], dim=0)

        out = self.head(layer_outputs[-1][-1])  # top layer, last col
        if return_attn:
            return out, h_new, {}
        return out, h_new

    def init_state(self, batch_size: int, device):
        return torch.zeros(
            self.n_layers, self.n_columns, batch_size, self.hidden_size,
            device=device,
        )

    def reset_state(self, h, mask: torch.Tensor):
        if h is None:
            return self.init_state(mask.shape[0], mask.device)
        ixs = torch.nonzero(mask).flatten()
        if ixs.numel() == 0:
            return h
        h = h.clone()
        h[:, :, ixs, :] = 0.0
        return h

    def detach_state(self, h):
        return h.detach() if h is not None else None
