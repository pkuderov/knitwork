from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch


class MessagePassingLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, n_participants: int | None = None):
        super().__init__()
        self.mha  = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=False)
        self.norm = nn.LayerNorm(dim)
        xa = (1 / dim) ** 0.5
        self.ids = None
        if n_participants is not None:
            self.ids = nn.Parameter(torch.empty(n_participants, 1, dim))
            nn.init.normal_(self.ids, 0.0, 0.01 * xa)
        nn.init.normal_(self.mha.out_proj.weight, 0.0, 0.01 * xa)
        nn.init.zeros_(self.mha.out_proj.bias)

    def forward(self, h: torch.Tensor, return_weights: bool = False):
        # h: (cols, batch, dim)
        qh = kh = h + self.ids if self.ids is not None else h
        h_mixed, attn_w = self.mha(qh, kh, h, average_attn_weights=True)
        if return_weights and attn_w is not None:
            attn_w = attn_w.mean(0)
        return self.norm(h + h_mixed), attn_w


class HaltingUnit(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, -1.0)

    def forward(self, h_pool: torch.Tensor) -> torch.Tensor:
        # h_pool: (batch, hidden) -> (batch,)
        return torch.sigmoid(self.proj(h_pool).squeeze(-1))


class ChainOfThoughtGRU(nn.Module):
    def __init__(self, hidden_size: int, thought_size: int):
        super().__init__()
        self.cell = nn.GRUCell(hidden_size, thought_size)
        self.norm = nn.LayerNorm(thought_size)

    def forward(self, h_top: torch.Tensor, thought: torch.Tensor) -> torch.Tensor:
        return self.norm(self.cell(h_top, thought))


class EquilibriumGridRnnCoT(nn.Module):
    """Equilibrium-GridRNN with ACT and Chain-of-Thought."""

    def __init__(
        self, *,
        input_size:               int,
        embedding_size:           int,
        output_size:              int,
        hidden_size:              int,
        n_layers:                 int,
        n_columns:                int,
        n_attn_heads:             int,
        col_identities:           bool,
        thought_size:             int | None = None,
        use_bias:                 bool  = True,
        dropout:                  float = 0.0,
        max_eq_iters:             int   = 12,
        eq_tol:                   float = 1e-2,
        act_eps:                  float = 0.01,
        act_loss_weight:          float = 1e-3,
        eq_residual_weight:       float = 1e-2,
        col_participation_weight: float = 1e-3,
        anderson_beta:            float = 0.0,
        attn_every:               int   = 1,
    ):
        super().__init__()
        assert n_columns > 1

        self.input_size               = input_size
        self.embedding_size           = embedding_size
        self.output_size              = output_size
        self.n_layers                 = n_layers
        self.n_columns                = n_columns
        self.n_attn_heads             = n_attn_heads
        self.act_eps                  = act_eps
        self.act_loss_weight          = act_loss_weight
        self.eq_residual_weight       = eq_residual_weight
        self.col_participation_weight = col_participation_weight
        self.max_eq_iters             = max_eq_iters
        self.eq_tol                   = eq_tol
        self.anderson_beta            = anderson_beta
        self.attn_every               = max(1, attn_every)

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.drop      = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.hidden_size  = hidden_size - hidden_size % n_attn_heads
        self.thought_size = thought_size or self.hidden_size

        print(
            f"EqGridRNN-CoT | {n_layers}L x {n_columns}C"
            f" | hidden={self.hidden_size} thought={self.thought_size}"
            f" | max_eq_iters={max_eq_iters} attn_every={attn_every}"
        )

        self.cells   = nn.ModuleList()
        self.attn    = nn.ModuleList()
        self.gates   = nn.ModuleList()
        self.halters = nn.ModuleList()

        for layer in range(n_layers):
            row = nn.ModuleList([
                nn.GRUCell(
                    input_size  = self._cell_input_dim(layer, ic),
                    hidden_size = self.hidden_size,
                    bias        = use_bias,
                )
                for ic in range(n_columns)
            ])
            self.cells.append(row)

            n_part = n_columns if col_identities else None
            self.attn.append(MessagePassingLayer(
                self.hidden_size, num_heads=n_attn_heads, n_participants=n_part
            ))

            # gate starts open: sigmoid(1.0) ≈ 0.73
            gate = nn.Linear(2 * self.hidden_size, 1)
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, 1.0)
            self.gates.append(gate)

            self.halters.append(HaltingUnit(self.hidden_size))

        self.cot  = ChainOfThoughtGRU(self.hidden_size, self.thought_size)
        self.head = nn.Linear(self.hidden_size + self.thought_size, output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Param count: {format_readable_num(param_count)}")

    # --- public API ---

    def forward(self, tokens: torch.Tensor, state=None, return_attn: bool = False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2 and tokens.shape[1] == 1
        bsz = tokens.shape[0]

        if state is None:
            h       = self.init_hidden(bsz)
            thought = self.init_thought(bsz)
        else:
            h, thought = state

        x = self.drop(self.embedding(tokens.view(-1)))   # (batch, emb)

        h_new, eq_metrics = self._grid_step(x, h, collect_metrics=return_attn)

        h_top   = h_new[-1, 0]   # top layer, col 0  (batch, hidden)
        thought = self.cot(h_top, thought)
        y       = self.head(torch.cat([h_top, thought], dim=-1))

        new_state = (h_new, thought)
        if return_attn:
            return y, new_state, eq_metrics
        return y, new_state

    def total_eq_loss(self, eq_metrics: dict) -> torch.Tensor:
        """Weighted sum of ponder cost, equilibrium residual, and participation losses."""
        device = self.head.weight.device
        dtype  = self.head.weight.dtype
        loss   = torch.zeros(1, device=device, dtype=dtype).squeeze()

        ponder        = eq_metrics.get("eq_ponder_cost_tensor")
        residual      = eq_metrics.get("eq_residual_loss")
        participation = eq_metrics.get("eq_participation_loss")

        if ponder is not None:
            loss = loss + self.act_loss_weight * ponder
        if residual is not None:
            loss = loss + self.eq_residual_weight * residual
        if participation is not None:
            loss = loss + self.col_participation_weight * participation
        return loss

    # --- grid step ---

    def _grid_step(
        self,
        x:               torch.Tensor,   # (batch, emb)
        h:               torch.Tensor,   # (layers, cols, batch, hidden)
        collect_metrics: bool = False,
    ) -> tuple[torch.Tensor, dict]:

        bsz    = x.shape[0]
        device = x.device
        dtype  = x.dtype

        x_in = self._prepare_grid_input(x, bsz, device, dtype)

        h_n:     list[torch.Tensor] = []
        metrics: dict = {
            "attn_weights":        [],
            "gates":               [],
            "act_iters":           [],
            "eq_delta_norms":      [],
            "eq_convergence_rate": [],
            "eq_halt_probs":       [],
        }

        total_ponder_tensor        = torch.zeros(1, device=device, dtype=dtype)
        total_residual_tensor      = torch.zeros(1, device=device, dtype=dtype)
        total_participation_tensor = torch.zeros(1, device=device, dtype=dtype)

        for layer_idx, (cells_row, attn_mod, gate_mod, halter) in enumerate(
            zip(self.cells, self.attn, self.gates, self.halters)
        ):
            hl = h[layer_idx]   # (cols, batch, hidden)

            # --- ACT state ---
            halt_acc  = torch.zeros(bsz, device=device, dtype=dtype)
            halt_mask = torch.zeros(bsz, device=device, dtype=torch.bool)
            h_acc     = torch.zeros_like(hl)   # (cols, batch, hidden)
            n_iters   = torch.zeros(bsz, device=device, dtype=dtype)

            h_layer      = hl
            h_layer_prev = hl
            last_delta   = torch.zeros(bsz, device=device, dtype=dtype)
            last_attn_w  = None
            last_gate_raw = None
            attn_w_accum: list[torch.Tensor] = []

            for act_step in range(self.max_eq_iters):
                h_new_cols = [cell(x_in[ic], h_layer[ic]) for ic, cell in enumerate(cells_row)]
                h_layer_new = torch.stack(h_new_cols, dim=0)   # (cols, batch, hidden)

                # Anderson mixing
                if self.anderson_beta > 0.0 and act_step >= 1:
                    h_layer_new = h_layer_new + self.anderson_beta * (
                        h_layer_new - h_layer_prev
                    )

                # cross-column attention
                is_attn_step = (
                    (act_step % self.attn_every == 0)
                    or (act_step == self.max_eq_iters - 1)
                )
                if is_attn_step:
                    msg, attn_w = attn_mod(h_layer_new, return_weights=True)
                    g_raw = gate_mod(torch.cat([h_layer_new, msg], dim=-1))  # (cols, batch, 1)
                    g     = torch.sigmoid(g_raw)
                    h_layer_new = (1.0 - g) * h_layer_new + g * msg
                    last_attn_w   = attn_w if collect_metrics else None
                    last_gate_raw = g_raw
                    if attn_w is not None:
                        attn_w_accum.append(attn_w.detach())

                # residual norm  [batch,]
                delta     = (h_layer_new - h_layer).norm(dim=-1).mean(dim=0)
                last_delta = delta.detach()

                # ACT halting
                h_pool = h_layer_new.mean(dim=0)   # (batch, hidden)
                p_halt = halter(h_pool)             # (batch,)

                is_last = (act_step == self.max_eq_iters - 1)
                p_use   = torch.where(
                    is_last | (halt_acc + p_halt >= 1.0 - self.act_eps),
                    (1.0 - halt_acc).clamp(min=0.0),
                    p_halt,
                )

                w     = p_use.unsqueeze(0).unsqueeze(-1)   # (1, batch, 1)
                h_acc = h_acc + w * h_layer_new

                halt_acc  = halt_acc + p_use.detach()
                n_iters  += (~halt_mask).float()
                halt_mask = halt_mask | (halt_acc >= 1.0 - self.act_eps)

                h_layer_prev = h_layer
                h_layer      = h_layer_new

                if halt_mask.all():
                    break

            # equilibrium residual loss
            h_acc_det = h_acc.detach()
            h_check   = torch.stack(
                [cell(x_in[ic], h_acc_det[ic]) for ic, cell in enumerate(cells_row)], dim=0
            )
            total_residual_tensor = total_residual_tensor + F.mse_loss(h_check, h_acc_det)

            # ponder cost: normalized iteration count  [1/max .. 1]
            total_ponder_tensor = total_ponder_tensor + n_iters.mean() / self.max_eq_iters

            # column participation loss: maximize entropy of attention weights
            if attn_w_accum:
                attn_w_mean = torch.stack(attn_w_accum).mean(0)   # (cols, cols)
                col_entropy = -(
                    attn_w_mean * (attn_w_mean + 1e-8).log()
                ).sum(-1).mean()
                total_participation_tensor = total_participation_tensor + (-col_entropy)

            h_n.append(h_acc)
            metrics["act_iters"].append(n_iters)
            metrics["attn_weights"].append(last_attn_w)
            metrics["gates"].append(
                last_gate_raw if last_gate_raw is not None
                else torch.zeros(1, device=device)
            )

            if collect_metrics:
                conv_rate = (last_delta < self.eq_tol).float().mean().item()
                metrics["eq_delta_norms"].append(last_delta.mean().item())
                metrics["eq_convergence_rate"].append(conv_rate)
                metrics["eq_halt_probs"].append(p_halt.mean().item())

            x_in = h_acc   # pass to next layer

        h_n_tensor = torch.stack(h_n, dim=0)   # (layers, cols, batch, hidden)

        metrics["eq_ponder_cost_tensor"]  = total_ponder_tensor.squeeze()
        metrics["eq_residual_loss"]       = total_residual_tensor.squeeze()
        metrics["eq_participation_loss"]  = total_participation_tensor.squeeze()
        metrics["eq_ponder_cost"]         = total_ponder_tensor.item()

        return h_n_tensor, metrics

    # --- utilities ---

    def _cell_input_dim(self, ix_layer: int, ix_col: int) -> int:
        if ix_layer == 0:
            return self.embedding_size if ix_col == 0 else 1
        return self.hidden_size

    def _prepare_grid_input(
        self, x: torch.Tensor, bsz: int, device: torch.device, dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        dummy = torch.zeros(bsz, 1, device=device, dtype=dtype)
        return [x] + [dummy] * (self.n_columns - 1)

    def init_hidden(self, bsz: int) -> torch.Tensor:
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.head.weight.device, dtype=self.head.weight.dtype,
        )

    def init_thought(self, bsz: int) -> torch.Tensor:
        return torch.zeros(
            bsz, self.thought_size,
            device=self.head.weight.device, dtype=self.head.weight.dtype,
        )

    def reset_state(self, state, reset_mask):
        if state is None:
            bsz = reset_mask.shape[0]
            return self.init_hidden(bsz), self.init_thought(bsz)
        h, thought = state
        ixs = torch.nonzero(reset_mask).flatten()
        if ixs.numel() == 0:
            return h, thought
        h       = h.clone();       h[:, :, ixs, :] = 0.0
        thought = thought.clone(); thought[ixs, :]  = 0.0
        return h, thought

    def detach_state(self, state):
        if state is None:
            return None
        h, thought = state
        return h.detach(), thought.detach()
