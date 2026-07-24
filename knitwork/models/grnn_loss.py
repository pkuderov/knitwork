from __future__ import annotations

import torch
from typing import Optional

from knitwork.models.grnn import GridRnn
from knitwork.models.diversity import ColumnDiversityLoss, DiversityLossConfig


class GridRnnLoss(GridRnn):
    """GridRNN with column diversity losses."""

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
        col_identities: bool,
        use_bias: bool = True,
        dropout: float = 0.0,
        diversity_cfg: Optional[DiversityLossConfig] = None,
    ):
        super().__init__(
            input_size=input_size,
            embedding_size=embedding_size,
            output_size=output_size,
            hidden_size=hidden_size,
            base_hidden_size=None,
            n_layers=n_layers,
            n_columns=n_columns,
            n_attn_heads=n_attn_heads,
            messaging=messaging,
            col_identities=col_identities,
            use_bias=use_bias,
            dropout=dropout,
        )
        if diversity_cfg is None:
            # layer weights grow linearly: 0.5 (first) .. 2.0 (last)
            layer_w = [0.5 + 1.5 * i / max(n_layers - 1, 1) for i in range(n_layers)]
            diversity_cfg = DiversityLossConfig(layer_weights=layer_w)
        self.diversity_cfg = diversity_cfg
        self.diversity_loss_fn = ColumnDiversityLoss(diversity_cfg, n_layers)

    def compute_diversity_loss(self, extras: dict) -> dict[str, torch.Tensor]:
        h_layers = extras.get('h_layers', [])
        gates = extras.get('gates', [])
        if not h_layers:
            zero = torch.tensor(0.0)
            return {k: zero for k in ('cosine', 'covariance', 'variance', 'gate_entropy', 'total')}
        return self.diversity_loss_fn(h_layers, gates)

    def grid_step(self, x: torch.Tensor, *, h: torch.Tensor, return_attn: bool = True):
        h_n, attn_list, gate_list, h_layer_list = [], [], [], []
        x = self._prepare_grid_input(x)

        for cells, attn, attn_gate, hl in zip(self.cells, self.attn, self.attn_gates, h):
            hl_n = [self.cell_forward(cells, x, hl, ix_col=ic) for ic in range(self.n_columns)]
            hl_n = torch.stack(hl_n, dim=0)  # [cols, batch, H]

            msg, attn_w = attn(hl_n, return_weights=return_attn)
            g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))
            hl_n = (1.0 - g) * hl_n + g * msg

            h_n.append(hl_n)
            attn_list.append(attn_w)
            gate_list.append(g)
            h_layer_list.append(hl_n)
            x = hl_n

        h_n = torch.stack(h_n, dim=0)
        return h_n, {'attn_weights': attn_list, 'gates': gate_list, 'h_layers': h_layer_list}
