from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch
from knitwork.models.diversity import ColumnSpecializationLoss


class MessagePassingLayer(nn.Module):
    """MHA message passing with enhanced per-column identity anchors."""

    def __init__(self, dim: int, num_heads: int, n_participants: Optional[int] = None):
        super().__init__()
        self.mha  = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=False)
        self.norm = nn.LayerNorm(dim)
        xa = (1 / dim) ** 0.5

        self.ids = None
        if n_participants is not None:
            self.ids = nn.Parameter(torch.empty(n_participants, 1, dim))
            nn.init.normal_(self.ids, 0.0, 0.1 * xa)  # larger std than base grnn

        # per-column low-rank Q/K projection  [C, dim, proj_dim]
        self.col_proj = self.col_proj_out = None
        if n_participants is not None:
            proj_dim = max(dim // 4, num_heads)
            proj_dim -= proj_dim % num_heads
            self.col_proj = nn.Parameter(torch.empty(n_participants, dim, proj_dim))
            nn.init.orthogonal_(self.col_proj.view(-1, proj_dim))
            self.col_proj_out = nn.Parameter(torch.empty(n_participants, proj_dim, dim))
            nn.init.orthogonal_(self.col_proj_out.view(-1, dim))

        # per-column post-attention nonlinearity
        self.post_proj = None
        if n_participants is not None:
            self.post_proj = nn.ModuleList([
                nn.Sequential(nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, dim))
                for _ in range(n_participants)
            ])
            for seq in self.post_proj:
                nn.init.normal_(seq[-1].weight, 0.0, 0.01 * xa)
                nn.init.zeros_(seq[-1].bias)

        nn.init.normal_(self.mha.out_proj.weight, 0.0, 0.01 * xa)
        nn.init.zeros_(self.mha.out_proj.bias)

    def forward(self, h: torch.Tensor, return_weights: bool = False):
        # h: [cols, batch, dim]
        qh = kh = h
        if self.ids is not None:
            qh = kh = h + self.ids

        if self.col_proj is not None:
            proj = torch.einsum('cbd,cdp->cbp', qh, self.col_proj)
            proj = torch.einsum('cbp,cpd->cbd', proj, self.col_proj_out)
            qh = kh = qh + 0.1 * proj

        h_mixed, attn_w = self.mha(qh, kh, h, average_attn_weights=True)

        if self.post_proj is not None:
            h_mixed = torch.stack(
                [h_mixed[c] + self.post_proj[c](h_mixed[c]) for c in range(h_mixed.shape[0])],
                dim=0,
            )

        if return_weights and attn_w is not None:
            attn_w = attn_w.mean(dim=0)
        return self.norm(h_mixed), attn_w


class GridRnn(nn.Module):
    """GridRNN with ColumnSpecializationLoss to prevent column collapse."""

    def __init__(
        self, *,
        input_size, embedding_size, output_size,
        hidden_size: int,
        n_layers: int, n_columns: int, n_attn_heads,
        messaging: str = "post",
        col_identities,
        use_bias=True, dropout: float = 0.0,
        spec_lambda_decorr: float = 1.0,
        spec_lambda_var:    float = 0.5,
        spec_lambda_cosine: float = 0.3,
        spec_lambda_whiten: float = 0.1,
        spec_target_layers: Optional[List[int]] = None,
        spec_loss_weight:   float = 0.1,
    ):
        super().__init__()
        assert n_columns > 1
        self.input_size      = input_size
        self.embedding_size  = embedding_size
        self.output_size     = output_size
        self.n_layers        = n_layers
        self.n_columns       = n_columns
        self.n_attn_heads    = n_attn_heads
        self.spec_loss_weight = spec_loss_weight
        self.use_postmsg     = messaging == "post"

        self.embedding = nn.Embedding(input_size, embedding_size)

        self.hidden_size = hidden_size - hidden_size % n_attn_heads
        print(f'GridRNN of {n_layers}L x {n_columns}C GRU cells w/ {self.hidden_size} hidden units')

        self.spec_loss = ColumnSpecializationLoss(
            hidden_size=self.hidden_size,
            n_columns=n_columns, n_layers=n_layers,
            lambda_decorr=spec_lambda_decorr, lambda_var=spec_lambda_var,
            lambda_cosine=spec_lambda_cosine, lambda_whiten=spec_lambda_whiten,
            target_layers=spec_target_layers or list(range(n_layers)),
        )

        self.cells      = nn.ModuleList()
        self.attn       = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        for layer in range(n_layers):
            self.cells.append(nn.ModuleList([
                nn.GRUCell(
                    input_size=self._cell_input_dim(layer, ic),
                    hidden_size=self.hidden_size, bias=use_bias, dtype=torch.float64,
                )
                for ic in range(n_columns)
            ]))
            n_part = n_columns if col_identities else None
            self.attn.append(MessagePassingLayer(self.hidden_size, n_attn_heads, n_part))
            if self.use_postmsg:
                self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

        self.head = nn.Linear(self.hidden_size, output_size)
        print(f'Param count: {format_readable_num(sum(p.numel() for p in self.parameters() if p.requires_grad))}')

    def forward(
        self,
        tokens: torch.Tensor,
        h=None,
        return_attn: bool = False,
        collect_states: bool = False,
        compute_spec_loss: bool = True,
    ):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2 and tokens.shape[1] == 1
        x = self.embedding(tokens.view(-1))

        if self.use_postmsg:
            h_new, extras = self.grid_step_postmsg(x, h=h, return_attn=return_attn)
        else:
            h_new = self.grid_step_premsg(x, h=h)
            extras = {}

        z = h_new[-1][0]  # top layer, first col
        y = self.head(z)

        if compute_spec_loss:
            spec_loss, spec_details = self.spec_loss(h_new)
            extras["spec_loss"]    = spec_loss * self.spec_loss_weight
            extras["spec_details"] = spec_details

        if collect_states:
            extras["hidden_states"] = h_new  # [L, C, B, D]

        return y, h_new, extras

    def grid_step_postmsg(self, x, *, h, return_attn=True):
        h_n, attn_list, gate_list = [], [], []
        x = self._prepare_grid_input(x)
        for cells, attn, attn_gate, hl in zip(self.cells, self.attn, self.attn_gates, h):
            hl_n = torch.stack(
                [cells[ic](x[ic], hl[ic]) for ic in range(self.n_columns)], dim=0
            )
            msg, attn_w = attn(hl_n, return_weights=return_attn)
            g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))
            hl_n = (1 - g) * hl_n + g * msg
            h_n.append(hl_n)
            attn_list.append(attn_w)
            gate_list.append(g)
            x = hl_n
        return torch.stack(h_n, dim=0), {"attn_weights": attn_list, "gates": gate_list}

    def grid_step_premsg(self, x, *, h):
        h_n = []
        x = self._prepare_grid_input(x)
        first_row = True
        for cells, attn, hl in zip(self.cells, self.attn, h):
            msg, _ = attn(hl, return_weights=False)
            if first_row:
                x = [torch.cat([xc, mc], -1) for xc, mc in zip(x, msg)]
            else:
                x = torch.cat([x, msg], dim=-1)  # type: ignore
            hl_n = torch.stack(
                [cells[ic](x[ic], hl[ic]) for ic in range(self.n_columns)], dim=0
            )
            h_n.append(hl_n)
            x = hl_n
            first_row = False
        return torch.stack(h_n, dim=0)

    def reset_state(self, state, reset_mask):
        if state is None:
            return self.init_state(reset_mask.shape[0])
        ixs = torch.nonzero(reset_mask).flatten()
        if ixs.numel() == 0:
            return state
        state = state.clone()
        state[:, :, ixs, :] *= 0.0
        return state

    def detach_state(self, state):
        return state.detach() if state is not None else state

    def _cell_input_dim(self, ix_layer: int, ix_col: int) -> int:
        if ix_layer == 0:
            return self.embedding_size if ix_col == 0 else 1
        return self.hidden_size * (1 if self.use_postmsg else 2)

    def _prepare_grid_input(self, x):
        bsz, _ = x.shape
        dummy = torch.zeros(bsz, self._cell_input_dim(0, 1), device=x.device, dtype=x.dtype)
        return [x] + [dummy] * (self.n_columns - 1)

    def init_state(self, bsz: int) -> torch.Tensor:
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.head.weight.device, dtype=self.head.weight.dtype,
        )
