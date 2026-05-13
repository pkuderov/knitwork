from __future__ import annotations

import torch
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch
from knitwork.models.grnn import MessagePassingLayer
from knitwork.models.lru import LRUCell, LRUBlock


class GridLRU(nn.Module):
    """Grid RNN with LRU cells instead of GRU.

    State shape: [layers, cols, batch, 2*hidden_size] (complex LRU state).
    Output activations [batch, hidden_size] are separate from the state.
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
        n_attn_heads: int,
        messaging: str = "post",
        col_identities: bool = False,
        use_bias: bool = True,
        dropout: float = 0.0,
        ff_mult: int = 2,
        r_min: float = 0.0,
        r_max: float = 0.999,
        lru_r_per_col: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.embedding = nn.Embedding(input_size, embedding_size)

        self.n_layers = n_layers
        self.n_columns = n_columns
        self.n_attn_heads = n_attn_heads
        assert n_columns > 1

        self.hidden_size = hidden_size - hidden_size % n_attn_heads
        self.use_postmsg = messaging == "post"
        self.lru_r_per_col = lru_r_per_col

        print(f'GridLRU {n_layers}L x {n_columns}C  hidden={self.hidden_size}')

        # grid of cells: layers x columns
        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.attn_gates = nn.ModuleList()  # post-messaging only

        for layer in range(n_layers):
            row = nn.ModuleList()
            for icol in range(n_columns):
                col_r_max = (
                    r_min + (r_max - r_min) * (icol + 1) / n_columns
                    if lru_r_per_col else r_max
                )
                row.append(LRUBlock(
                    input_size=self._cell_input_dim(layer, icol),
                    hidden_size=self.hidden_size,
                    ff_mult=ff_mult,
                    r_min=r_min,
                    r_max=col_r_max,
                    dropout=dropout,
                ))
            self.cells.append(row)

            n_part = n_columns if col_identities else None
            self.attn.append(MessagePassingLayer(
                self.hidden_size, num_heads=n_attn_heads, n_participants=n_part
            ))
            if self.use_postmsg:
                self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

        self.head = nn.Linear(self.hidden_size, output_size)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(n_params)}')

    def forward(self, tokens: torch.Tensor, h=None, return_attn: bool = False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2
        assert tokens.shape[1] == 1, "expected 1 token per step"

        x = self.embedding(tokens.view(-1))  # [batch, emb_size]

        if self.use_postmsg:
            h, last_out, extras = self.grid_step_postmsg(x, h=h, return_attn=return_attn)
        else:
            h, last_out, extras = self.grid_step_premsg(x, h=h)

        z = last_out[0]   # [batch, H] — top layer, col 0
        y = self.head(z)  # [batch, output_size]

        if return_attn:
            return y, h, extras
        return y, h

    def grid_step_postmsg(
        self, x: torch.Tensor, *, h: torch.Tensor, return_attn: bool = True
    ):
        """
        Returns:
            h_new    : [layers, cols, batch, 2H]
            last_out : [cols, batch, H]
            extras   : dict(attn_weights, gates)
        """
        h_new_list, attn_list, gate_list = [], [], []
        x_list = self._prepare_grid_input(x)  # list[cols] of [batch, dim]

        for cells, attn, attn_gate, hl in zip(self.cells, self.attn, self.attn_gates, h):
            out_cols, state_cols = [], []
            for icol in range(self.n_columns):
                y_col, h_col_n = cells[icol](x_list[icol], hl[icol])
                out_cols.append(y_col)
                state_cols.append(h_col_n)

            out_t   = torch.stack(out_cols,   dim=0)  # [cols, batch, H]
            state_t = torch.stack(state_cols, dim=0)  # [cols, batch, 2H]

            msg, attn_w = attn(out_t, return_weights=return_attn)
            g      = torch.sigmoid(attn_gate(torch.cat([out_t, msg], dim=-1)))  # [cols, batch, 1]
            merged = (1 - g) * out_t + g * msg                                  # [cols, batch, H]

            h_new_list.append(state_t)
            attn_list.append(attn_w)
            gate_list.append(g)
            x_list = [merged[ic] for ic in range(self.n_columns)]

        last_out = torch.stack(x_list,     dim=0)  # [cols, batch, H]
        h_new    = torch.stack(h_new_list, dim=0)  # [layers, cols, batch, 2H]
        return h_new, last_out, {"attn_weights": attn_list, "gates": gate_list}

    def grid_step_premsg(self, x: torch.Tensor, *, h: torch.Tensor):
        h_new_list = []
        x_list = self._prepare_grid_input(x)

        for cells, attn, hl in zip(self.cells, self.attn, h):
            # pre-messaging: aggregate from previous activations
            prev_out = torch.stack(
                [hl[ic][:, :self.hidden_size] for ic in range(self.n_columns)], dim=0
            )  # [cols, batch, H]
            msg, _ = attn(prev_out, return_weights=False)

            x_aug = [torch.cat([x_list[ic], msg[ic]], dim=-1) for ic in range(self.n_columns)]

            out_cols, state_cols = [], []
            for icol in range(self.n_columns):
                y_col, h_col_n = cells[icol](x_aug[icol], hl[icol])
                out_cols.append(y_col)
                state_cols.append(h_col_n)

            h_new_list.append(torch.stack(state_cols, dim=0))
            x_list = out_cols

        last_out = torch.stack(x_list,     dim=0)  # [cols, batch, H]
        h_new    = torch.stack(h_new_list, dim=0)  # [layers, cols, batch, 2H]
        return h_new, last_out, {}

    def init_state(self, bsz: int) -> torch.Tensor:
        # [layers, cols, batch, 2*H] — complex LRU state
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, 2 * self.hidden_size,
            device=self.head.weight.device,
            dtype=self.head.weight.dtype,
        )

    def reset_state(self, state, reset_mask) -> torch.Tensor:
        if state is None:
            return self.init_state(reset_mask.shape[0])
        ixs = torch.nonzero(reset_mask).flatten()
        if ixs.numel() == 0:
            return state
        state = state.clone()
        state[:, :, ixs, :] = 0.0
        return state

    def detach_state(self, state):
        return state.detach() if state is not None else None

    def _cell_input_dim(self, ix_layer: int, ix_col: int) -> int:
        if ix_layer == 0:
            return self.embedding_size if ix_col == 0 else 1
        return 2 * self.hidden_size if not self.use_postmsg else self.hidden_size

    def _prepare_grid_input(self, x: torch.Tensor) -> list[torch.Tensor]:
        # col 0 gets embedding, col 1..N get zeros
        bsz = x.shape[0]
        dummy_dim = self._cell_input_dim(ix_layer=0, ix_col=1)
        dummy = torch.zeros(bsz, dummy_dim, device=x.device, dtype=x.dtype)
        return [x] + [dummy] * (self.n_columns - 1)
