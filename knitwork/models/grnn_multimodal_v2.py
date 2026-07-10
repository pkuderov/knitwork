from __future__ import annotations

import torch
from torch import nn

from knitwork.common.utils import convert_hidden_size, format_readable_num, to_torch


class MaskedMessagePassingLayer(nn.Module):
    """Like grnn.MessagePassingLayer, but signal columns are structurally
    forbidden from attending to buffer columns (hard-gated isolation, rather
    than relying on attention weights learning to go to ~0 on their own)."""

    def __init__(self, dim, num_heads, n_participants: int, attn_mask: torch.Tensor):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=False)
        self.norm = nn.LayerNorm(dim)

        xavier_alpha = (1 / dim) ** 0.5
        self.ids = nn.Parameter(torch.empty(n_participants, 1, dim))
        nn.init.normal_(self.ids, 0.0, 0.01 * xavier_alpha)

        nn.init.normal_(self.mha.out_proj.weight, 0.0, 0.01 * xavier_alpha)
        nn.init.zeros_(self.mha.out_proj.bias)

        # [n_cols, n_cols] bool, True = query col forbidden from attending to key col
        self.register_buffer('attn_mask', attn_mask, persistent=False)

    def forward(self, h, return_weights: bool = False):
        # h: (cols, batch, dim)
        qh = kh = h + self.ids
        vh = h
        h_mixed, attn_w = self.mha(
            qh, kh, vh, attn_mask=self.attn_mask, average_attn_weights=True
        )
        if return_weights and attn_w is not None:
            attn_w = attn_w.mean(dim=0)
        return self.norm(h_mixed), attn_w


class GridRnnMultimodalV2(nn.Module):
    """
    v2 fixes over `grnn_multimodal.GridRnnMultimodal`, informed by diagnostics
    from a real training run (col_sim, attention heatmap, ablation):

    1. Hard-gated buffer isolation: signal columns are attention-masked so they
       CANNOT query buffer columns at any layer (grnn_multimodal only relied on
       attention weights learning to go to ~0, but the run showed layer-1's
       output-facing column still attending ~0.45 to a buffer column).
    2. Concat (not mean-pool) of the two signal columns at the head: since the
       two signal columns' representations diverge over training (col_sim went
       negative), mean-pooling them risks destructive cancellation.
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
        assert col_identities, "grnn_multimodal_v2 requires col_identities=True"
        assert n_columns >= 2
        assert messaging == "post", "v2 only implements post-messaging"

        self.image_feat_dim = image_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.signal_columns = tuple(signal_columns)
        assert len(self.signal_columns) == 2 and self.signal_columns[0] != self.signal_columns[1], (
            "v2 requires two distinct signal columns (no naive-concat degenerate mode)"
        )

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
        self.hidden_size -= self.hidden_size % self.n_attn_heads
        print(
            f'GridRnnMultimodalV2 of {self.n_layers}L x {self.n_columns}C GRU cells'
            f' w/ {self.hidden_size} hidden units, signal_columns={self.signal_columns}'
            f' (hard-isolated buffer cols={self.buffer_columns})'
        )

        # signal columns may never attend to buffer columns, at any layer
        attn_mask = torch.zeros(n_columns, n_columns, dtype=torch.bool)
        for sc in self.signal_columns:
            for bc in self.buffer_columns:
                attn_mask[sc, bc] = True

        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
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
            self.attn.append(MaskedMessagePassingLayer(
                self.hidden_size, num_heads=self.n_attn_heads,
                n_participants=self.n_columns, attn_mask=attn_mask,
            ))
            self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

        # concat (not mean-pool) the two signal columns — avoids cancellation
        # if their representations become (anti-)correlated
        self.head = nn.Linear(2 * self.hidden_size, self.output_size)

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
        h, extras = self.grid_step_postmsg(x, h=h, return_attn=return_attn)

        sc0, sc1 = self.signal_columns
        z = torch.cat([h[-1, sc0], h[-1, sc1]], dim=-1)

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
            x = hl_n

        h_n = torch.stack(h_n, dim=0)
        return h_n, {"attn_weights": attn_list, "gates": gate_list}

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
            return self.embedding_size
        return self.hidden_size  # post-messaging only

    def _prepare_grid_input(self, image_feat, audio_feat, buffer_feat):
        xl = [None] * self.n_columns
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
