from __future__ import annotations

import torch
from torch import nn

from knitwork.common.utils import convert_hidden_size, format_readable_num, to_torch
from knitwork.models.grnn import MessagePassingLayer


class GridRnnMultimodal(nn.Module):
    """
    GridRNN variant for the Multimodal Digit-Sum benchmark.

    Unlike the reference `grnn.GridRnn` (where only column 0 ever receives real
    external input, the rest get a dummy zero scalar), here EVERY column at
    layer 0 gets its own real, per-column-modality input: two "signal" columns
    (image, audio digit features) plus `n_columns - 2` "buffer" columns (pure
    noise). `col_identities` must stay True — it is the only way attention can
    learn "this column is buffer, ignore it".
    """

    def __init__(
            self, *,
            image_feat_dim, audio_feat_dim, output_size,
            embedding_size, buffer_feat_dim=None,
            hidden_size=None, base_hidden_size=None,
            n_layers: int, n_columns: int,
            n_attn_heads, messaging: str = "post", col_identities: bool = True,
            signal_columns: tuple = (0, 1),

            use_bias=True, dropout=0.0,
    ):
        super().__init__()
        assert col_identities, "grnn_multimodal requires col_identities=True"
        assert n_columns >= 2

        self.image_feat_dim = image_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.signal_columns = tuple(signal_columns)
        assert len(self.signal_columns) == 2

        self.buffer_columns = [c for c in range(n_columns) if c not in self.signal_columns]
        self.n_buffer_columns = len(self.buffer_columns)
        self.buffer_feat_dim = buffer_feat_dim or max(image_feat_dim, audio_feat_dim)

        self.image_proj = nn.Linear(image_feat_dim, embedding_size)
        self.audio_proj = nn.Linear(audio_feat_dim, embedding_size)
        self.buffer_proj = (
            nn.Linear(self.buffer_feat_dim, embedding_size) if self.n_buffer_columns > 0 else None
        )

        self.n_layers = n_layers
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
            f'GridRnnMultimodal of {self.n_layers}L x {self.n_columns}C GRU cells'
            f' w/ {self.hidden_size} hidden units, signal_columns={self.signal_columns}'
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
                )
                for icol in range(self.n_columns)
            )
            self.cells.append(nn.ModuleList(row))

            n_participants = self.n_columns  # col_identities is required for this model
            self.attn.append(MessagePassingLayer(
                self.hidden_size, num_heads=self.n_attn_heads, n_participants=n_participants
            ))

            if self.use_postmsg:
                self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

        # Head reads from the top layer, pooled over the signal columns
        # (the "answer" isn't tied to a single fixed column here, unlike GridRnn)
        self.head = nn.Linear(self.hidden_size, self.output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(self, image_feat, audio_feat, buffer_feat, h=None, return_attn=False):
        image_feat = to_torch(image_feat)
        audio_feat = to_torch(audio_feat)
        buffer_feat = to_torch(buffer_feat)
        assert buffer_feat.shape[1] == self.n_buffer_columns, (
            f"generator emits {buffer_feat.shape[1]} buffer columns, "
            f"model expects {self.n_buffer_columns} (n_columns - len(signal_columns))"
        )

        x = self._prepare_grid_input(image_feat, audio_feat, buffer_feat)

        if self.use_postmsg:
            h, extras = self.grid_step_postmsg(x, h=h, return_attn=return_attn)
        else:
            h, extras = self.grid_step_premsg(x, h=h), {}

        # mean-pool the top layer's signal columns as the grid output
        z = h[-1, self.signal_columns, :, :].mean(dim=0)

        y = self.head(z)
        if return_attn:
            return y, h, extras
        return y, h

    def grid_step_postmsg(self, x, *, h: torch.Tensor, return_attn=True):
        h_n, attn_list, gate_list = [], [], []

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
                x = torch.cat([x, msg], dim=-1)  # type: ignore

            hl_n = [
                self.cell_forward(cells, x, hl, ix_col=ix_col)
                for ix_col in range(self.n_columns)
            ]
            hl_n = torch.stack(hl_n, dim=0)

            h_n.append(hl_n)
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
            h = h.clone()
            h[:, :, ixs, :] *= 0.0
            return h

        return _reset(state)

    def detach_state(self, state):
        if state is None:
            return state
        return state.detach()

    def _cell_input_dim(self, ix_layer: int, ix_col) -> int:
        if ix_layer == 0:
            # every column gets a real, per-column-modality projected input
            return self.embedding_size

        hsz = self.hidden_size
        if not self.use_postmsg:
            # RNN input: [x; h_mix]
            hsz += self.hidden_size
        return hsz

    def _prepare_grid_input(self, image_feat, audio_feat, buffer_feat):
        # returns a list of per-column inputs, each [batch, embedding_size]
        xl = [None] * self.n_columns
        if self.signal_columns[0] == self.signal_columns[1]:
            # "naive concat" baseline: both modalities merged into one column
            xl[self.signal_columns[0]] = self.image_proj(image_feat) + self.audio_proj(audio_feat)
        else:
            xl[self.signal_columns[0]] = self.image_proj(image_feat)
            xl[self.signal_columns[1]] = self.audio_proj(audio_feat)
        for j, ix_col in enumerate(self.buffer_columns):
            xl[ix_col] = self.buffer_proj(buffer_feat[:, j])
        return xl

    def init_state(self, bsz):
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.head.weight.device, dtype=self.head.weight.dtype
        )
