from __future__ import annotations

import torch
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch
from knitwork.models.grnn import MessagePassingLayer


def _scale_to_spectral_radius(weight: torch.Tensor, target_radius: float) -> torch.Tensor:
    """Scale square weight matrix to the given spectral radius in-place."""
    with torch.no_grad():
        if weight.shape[0] <= 512:
            current_radius = torch.linalg.eigvals(weight).abs().max().item()
        else:
            # power iteration: O(n^2), fast for large matrices
            v = torch.randn(weight.shape[0], 1, device=weight.device, dtype=weight.dtype)
            for _ in range(20):
                v = weight @ v
                norm = v.norm()
                if norm < 1e-10:
                    break
                v = v / norm
            current_radius = (weight @ v).norm().item() / (v.norm().item() + 1e-10)
        if current_radius > 1e-10:
            weight.mul_(target_radius / current_radius)
    return weight


class GridRnnReservoir(nn.Module):
    """Grid RNN with frozen reservoir columns (Echo State Network style)."""

    def __init__(
        self, *,
        input_size: int,
        embedding_size: int,
        output_size: int,
        hidden_size: int,
        n_layers: int,
        n_columns: int,
        n_attn_heads: int,
        messaging: str = "post",
        col_identities: bool,
        n_reservoir_cols: int = 1,
        spectral_radius: float = 0.9,
        reservoir_scale: float = 0.1,
        use_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert 0 < n_reservoir_cols < n_columns, (
            f"n_reservoir_cols={n_reservoir_cols} must be in (0, {n_columns})"
        )

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.n_columns = n_columns
        self.n_attn_heads = n_attn_heads
        self.n_reservoir_cols = n_reservoir_cols
        self.n_trainable_cols = n_columns - n_reservoir_cols
        self.spectral_radius = spectral_radius

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.hidden_size = hidden_size - hidden_size % n_attn_heads
        self.use_postmsg = messaging == "post"

        print(
            f'GridRNN-Reservoir: {n_layers}L x {n_columns}C '
            f'({self.n_trainable_cols} trainable + {n_reservoir_cols} reservoir) '
            f'| hidden={self.hidden_size} | SR={spectral_radius}'
        )

        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.attn_gates = nn.ModuleList()

        for layer in range(n_layers):
            row = nn.ModuleList([
                nn.GRUCell(
                    input_size=self._cell_input_dim(layer, icol),
                    hidden_size=self.hidden_size,
                    bias=use_bias,
                    dtype=torch.float64,
                )
                for icol in range(n_columns)
            ])
            self.cells.append(row)

            # init and freeze reservoir columns (last n_reservoir_cols)
            for icol in range(self.n_trainable_cols, n_columns):
                cell = self.cells[layer][icol]
                self._init_reservoir_cell(cell, spectral_radius, reservoir_scale)
                for param in cell.parameters():
                    param.requires_grad = False

            n_participants = n_columns if col_identities else None
            self.attn.append(MessagePassingLayer(
                self.hidden_size, num_heads=n_attn_heads, n_participants=n_participants,
            ))
            if self.use_postmsg:
                self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

        self.head = nn.Linear(self.hidden_size, output_size)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f'Params: total={format_readable_num(total)}'
            f' | trainable={format_readable_num(trainable)}'
            f' | frozen={format_readable_num(total - trainable)}'
        )

    @staticmethod
    def _init_reservoir_cell(cell: nn.GRUCell, spectral_radius: float, scale: float):
        """Scale weight_hh to target SR per gate; randomize weight_ih; zero biases."""
        with torch.no_grad():
            hid = cell.hidden_size
            # GRUCell.weight_hh: [3*H, H] — three gate blocks
            for gate_idx in range(3):
                block = cell.weight_hh.data[gate_idx * hid:(gate_idx + 1) * hid]
                _scale_to_spectral_radius(block, spectral_radius)
            nn.init.uniform_(cell.weight_ih, -scale, scale)
            if cell.bias_ih is not None:
                nn.init.zeros_(cell.bias_ih)
            if cell.bias_hh is not None:
                nn.init.zeros_(cell.bias_hh)

    def forward(self, tokens: torch.Tensor, h=None, return_attn: bool = False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2 and tokens.shape[1] == 1

        x = self.embedding(tokens.view(-1))

        if self.use_postmsg:
            h, extras = self.grid_step_postmsg(x, h=h, return_attn=return_attn)
        else:
            h, extras = self.grid_step_premsg(x, h=h), {}

        z = h[-1][0]  # top layer, col 0
        y = self.head(z)

        if return_attn:
            return y, h, extras
        return y, h

    def grid_step_postmsg(self, x, *, h: torch.Tensor, return_attn: bool = False):
        h_n, attn_list, gate_list = [], [], []
        x = self._prepare_grid_input(x)

        for cells, attn_mod, attn_gate, hl in zip(self.cells, self.attn, self.attn_gates, h):
            hl_n = torch.stack(
                [self.cell_forward(cells, x, hl, ix_col=ic) for ic in range(self.n_columns)],
                dim=0,
            )  # [cols, batch, H]

            msg, attn_w = attn_mod(hl_n, return_weights=return_attn)
            g    = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))
            hl_n = (1 - g) * hl_n + g * msg

            h_n.append(hl_n)
            attn_list.append(attn_w)
            gate_list.append(g)
            x = hl_n

        h_n = torch.stack(h_n, dim=0)
        return h_n, {"attn_weights": attn_list, "gates": gate_list}

    def grid_step_premsg(self, x, *, h: torch.Tensor):
        h_n = []
        x = self._prepare_grid_input(x)
        first_row = True

        for cells, attn_mod, hl in zip(self.cells, self.attn, h):
            msg, _ = attn_mod(hl, return_weights=False)
            if first_row:
                x = [torch.cat([xc, msgc], -1) for xc, msgc in zip(x, msg)]
            else:
                x = torch.cat([x, msg], dim=-1)  # type: ignore

            hl_n = torch.stack(
                [self.cell_forward(cells, x, hl, ix_col=ic) for ic in range(self.n_columns)],
                dim=0,
            )
            h_n.append(hl_n)
            x = hl_n
            first_row = False

        return torch.stack(h_n, dim=0)

    def cell_forward(self, cells, x, h, *, ix_col):
        return cells[ix_col](x[ix_col], h[ix_col])

    def reset_state(self, state, reset_mask):
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

    def init_state(self, bsz: int):
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.head.weight.device,
            dtype=self.head.weight.dtype,
        )

    def _cell_input_dim(self, ix_layer: int, ix_col: int) -> int:
        if ix_layer == 0:
            return self.embedding_size if ix_col == 0 else 1
        return 2 * self.hidden_size if not self.use_postmsg else self.hidden_size

    def _prepare_grid_input(self, x):
        bsz = x.shape[0]
        in_dim = self._cell_input_dim(ix_layer=0, ix_col=1)
        dummy = torch.zeros(bsz, in_dim, device=x.device, dtype=x.dtype)
        return [x] + [dummy] * (self.n_columns - 1)

    def reservoir_info(self) -> dict:
        """Return spectral radii of reservoir cell gates (for monitoring)."""
        info = {}
        for layer in range(self.n_layers):
            for icol in range(self.n_trainable_cols, self.n_columns):
                cell = self.cells[layer][icol]
                hid = self.hidden_size
                radii = [
                    torch.linalg.eigvals(
                        cell.weight_hh.data[g * hid:(g + 1) * hid].float()
                    ).abs().max().item()
                    for g in range(3)
                ]
                info[f"layer{layer}_col{icol}_sr"] = radii
        return info
