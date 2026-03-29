from __future__ import annotations

import torch

from knitwork.common.utils import to_torch
from knitwork.models.grnn import GridRnn as BaseGridRnn


class GridRnn(BaseGridRnn):
    def __init__(
            self, *,
            input_size, embedding_size, output_size,
            hidden_size = None, base_hidden_size = None,
            n_layers: int, n_columns: int,
            n_attn_heads, messaging: str = "post", col_identities,
            
            use_bias = True, dropout = 0.0
    ):
        super().__init__(
            input_size=input_size,
            embedding_size=embedding_size,
            output_size=output_size,
            hidden_size=hidden_size,
            base_hidden_size=base_hidden_size,
            n_layers=n_layers,
            n_columns=n_columns,
            n_attn_heads=n_attn_heads,
            messaging=messaging,
            col_identities=col_identities,
            use_bias=use_bias,
            dropout=dropout
        )
        self._y_last = None

    def forward(self, tokens: torch.Tensor, h=None):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2
        bsz, n_features = tokens.shape
        assert n_features == 1, "expected input features dimension = 1 (token ids)"

        x_true = self.embedding(tokens.view(-1))

        if self._y_last is not None:
            t_pred = self._y_last
            probs = torch.softmax(t_pred, dim=-1)
            x_pred = torch.matmul(probs, self.embedding.weight)
        else:
            x_pred = torch.zeros_like(x_true)

        # shape: (layers, cols, batch, hidden_size)
        h = self.grid_step_postmsg(x_true=x_true, x_pred=x_pred, h=h)
        # top (=last) layer, first col as grid output
        z = h[-1][0]

        y = self.head(z)
        return y, h

    def grid_step_postmsg(self, *, x_true, x_pred, h: torch.Tensor):
        h_n = []
        # it is a list of inputs, each input is [batch, col_in_dim]
        x = self._prepare_grid_input(x_true=x_true, x_pred=x_pred)

        for cells, attn, attn_gate, hl in zip(self.cells, self.attn, self.attn_gates, h):
            hl_n = [
                self.cell_forward(cells, x, hl, ix_col=ix_col)
                for ix_col in range(self.n_columns)
            ]
            hl_n = torch.stack(hl_n, dim=0)

            msg = attn(hl_n)
            g = torch.sigmoid(attn_gate(
                torch.cat([hl_n, msg], dim=-1)
            ))
            hl_n = (1 - g) * hl_n + g * msg

            h_n.append(hl_n)
            # starting from there, x is a contiguous tensor [cols, batch, hidden_size]
            x = hl_n

        h_n = torch.stack(h_n, dim=0)
        return h_n

    def reset_state(self, state, reset_mask):
        if state is None:
            return self.init_state(reset_mask.shape[0])

        ixs = torch.nonzero(reset_mask).flatten()
        if ixs.numel() == 0:
            return state

        # shape: (layers, cols, batch, hidden_size)
        h = state.clone()
        h[:, :, ixs, :] *= 0.0

        if self._y_last is not None:
            self._y_last = self._y_last.clone()
            self._y_last[:, ixs, :] *= 0.0

        return h

    def detach_state(self, state):
        if state is None:
            return state

        if self._y_last is not None:
            self._y_last = self._y_last.detach()
        return state.detach()

    def _cell_input_dim(self, ix_layer: int, ix_col) -> int:
        if ix_layer == 0:
            # only the first col gets non-empty external input, 
            # the others get dummy 1-dim zero tensor
            return self.embedding_size if ix_col < 2 else 1

        hsz = self.hidden_size
        if not self.use_postmsg:
            # RNN input: [x; h_mix]
            hsz += self.hidden_size
        return hsz

    def _prepare_grid_input(self, x_true, x_pred):
        xl = [x_pred, x_true - x_pred]

        if self.n_columns > 2:
            bsz, n_features = x_pred.shape
            in_dim = self._cell_input_dim(ix_layer=0, ix_col=2)
            dummy_input = torch.zeros(bsz, in_dim, device=x_true.device, dtype=x_true.dtype)
            for _ in range(1, self.n_columns):
                xl.append(dummy_input)

        return xl
