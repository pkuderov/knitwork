"""Harmonic Grid RNN — four semantic blocks per layer:

1. Spectral LRU:        2D r_max hierarchy (fast/slow per column × per layer)
2. Surprise-Delta Mem:  EMA-gated delta-rule KV memory (no Hebbian interference)
3. Frozen Reservoir:    read-only long-range context (helps text BPC)
4. Hopfield Integration: sharp cross-column (+ reservoir) associative retrieval

Plus embedding residual skip at each layer for gradient flow.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch
from knitwork.models.hgrnn_lru import HopfieldMessageLayer, LRUCell, PositionwiseFFN


class HarmonicState(NamedTuple):
    h:      torch.Tensor  # [L, C, B, 2H]         — LRU complex states (Re | Im)
    h_res:  torch.Tensor  # [L, C_res, B, H_res]   — frozen reservoir states
    W:      torch.Tensor  # [L, B, dk, dv]          — surprise-delta memory
    m:      torch.Tensor  # [L, B]                   — EMA velocity-surprise per layer
    v2:     torch.Tensor  # [L, B, dk]               — per-batch Adam preconditioner
    y_prev: torch.Tensor  # [L, C, B, H]             — previous LRU outputs for velocity


class SurpriseDeltaMemory(nn.Module):
    """EMA-gated delta-rule KV memory shared across columns per layer.

    Combines:
    - EMA surprise write-gate (grnn_ema_mem): write ∝ prediction error
    - Parallel delta rule (grnn_prec_delta): error-corrective, no Hebbian interference
    """

    def __init__(
        self, *,
        hidden_size: int,
        n_cols: int,
        dk: int,
        dv: int,
        ema_beta: float = 0.9,
        delta_decay: float = 0.99,    # max decay (learned gate interpolates to this)
        decay_min: float = 0.90,      # min decay (strong forgetting)
        lam_base: float = 0.01,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.dk        = dk
        self.dv        = dv
        self.ema_beta  = ema_beta
        self.decay_max = delta_decay
        self.decay_min = decay_min
        self.lam_base  = lam_base
        self.layer_idx = layer_idx

        self.proj_k   = nn.Linear(hidden_size, dk,         bias=False)
        self.proj_v   = nn.Linear(hidden_size, dv,         bias=False)
        self.proj_q   = nn.Linear(hidden_size, dk,         bias=False)
        self.proj_out = nn.Linear(dv,          hidden_size, bias=False)
        self.col_ids  = nn.Parameter(torch.zeros(n_cols, dk))
        self.norm     = nn.LayerNorm(hidden_size)

        # learned forget gate: decay = decay_min + (decay_max - decay_min) * sigmoid(forget_proj)
        self.forget_proj = nn.Linear(hidden_size, 1, bias=True)
        nn.init.constant_(self.forget_proj.bias, 2.0)  # init: sigmoid(2)≈0.88 → decay≈0.98

        # Adam-style preconditioning: v2 passed in from HarmonicState (per-batch, avoids RL contamination)
        self.beta2    = 0.999
        self.prec_eps = 1e-6

        for proj in (self.proj_k, self.proj_v, self.proj_q):
            nn.init.normal_(proj.weight, 0.0, 0.01)
        nn.init.normal_(self.proj_out.weight, 0.0, 0.001)

    def forward(
        self,
        y: torch.Tensor,        # [C, B, H]
        W: torch.Tensor,        # [B, dk, dv]
        m_prev: torch.Tensor,   # [B]
        v2_prev: torch.Tensor,  # [B, dk] — per-batch Adam preconditioner
        y_prev: torch.Tensor,   # [C, B, H] — previous step LRU output for velocity
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        C, B, _ = y.shape
        y_flat = y.reshape(C * B, -1)

        k = self.proj_k(y_flat).view(C, B, self.dk)
        v_raw = self.proj_v(y_flat).view(C, B, self.dv)
        v = F.normalize(v_raw, dim=-1)
        q = self.proj_q(y_flat).view(C, B, self.dk)

        k = F.normalize(k + self.col_ids.unsqueeze(1), dim=-1)  # [C, B, dk]
        q = F.normalize(q + self.col_ids.unsqueeze(1), dim=-1)

        # per-batch Adam preconditioning (v2_prev: [B, dk])
        k_sq   = (k.detach() ** 2).mean(dim=0)                              # [B, dk]
        v2_new = self.beta2 * v2_prev + (1.0 - self.beta2) * k_sq
        k_prec = k / (v2_new.sqrt().unsqueeze(0) + self.prec_eps)           # [C, B, dk]

        # parallel delta rule on detached W — no chained Jacobians through sequential reads
        W_frozen   = W.detach()
        delta_W    = torch.zeros_like(W)   # [B, dk, dv]
        error_norm = torch.zeros(B, device=y.device, dtype=y.dtype)
        for c in range(C):
            v_pred  = torch.bmm(W_frozen.transpose(-2, -1), k[c].unsqueeze(-1)).squeeze(-1)
            error   = v[c] - v_pred                                          # [B, dv]
            error_norm += (error.detach() ** 2).mean(dim=-1).sqrt()
            delta_W += torch.bmm(k_prec[c].unsqueeze(-1), error.unsqueeze(1))   # [B, dk, dv]

        delta_W = delta_W / C   # stability: alpha*C < 1 + effective_decay

        # velocity-based write-gate (ema_mem style): write ∝ state change, not prediction error
        vel   = (y.detach() - y_prev).norm(dim=-1).mean(dim=0)              # [B]
        m_new = self.ema_beta * m_prev + (1.0 - self.ema_beta) * vel
        alpha = (m_new / m_new.detach().max().clamp(min=1e-6)).clamp(0.0, 1.0)  # [B]

        # fixed forgetting (fullness-adaptive was < 5% always, negligible effect)
        lam = self.lam_base

        # learned forget gate: model decides how much to forget
        forget          = torch.sigmoid(self.forget_proj(y.mean(0))).squeeze(-1)   # [B]
        effective_decay = self.decay_min + (self.decay_max - self.decay_min) * forget

        W_new = (1.0 - lam) * effective_decay.view(B, 1, 1) * W \
              + alpha.view(B, 1, 1) * delta_W

        # read for each column
        msgs  = [
            self.proj_out(torch.bmm(W_new.transpose(-2, -1), q[c].unsqueeze(-1)).squeeze(-1))
            for c in range(C)
        ]
        h_msg = self.norm(torch.stack(msgs, dim=0))   # [C, B, H]

        with torch.no_grad():
            fullness = W.detach().norm(dim=(-2, -1)) / math.sqrt(self.dk * self.dv)
            stats = {
                'W_norm':     W_new.detach().norm(dim=(-2, -1)).mean().item(),
                'alpha':      alpha.mean().item(),
                'surprise':   m_new.mean().item(),
                'fullness':   fullness.mean().item(),
                'error':      (error_norm / C).mean().item(),
                'forget':     forget.mean().item(),
                'eff_decay':  effective_decay.mean().item(),
                'v2_norm':    v2_new.norm(dim=-1).mean().item(),
            }
        return h_msg, W_new, m_new, v2_new, stats


class FrozenReservoir(nn.Module):
    """Fixed random RNN columns with linearly-spaced spectral radii.

    Provides multi-scale temporal basis functions; Hopfield attention can query them
    for long-range context (improves text BPC). State update is always detached.
    Set n_cols=0 to disable (e.g. for SDQ where sequences are short).
    """

    def __init__(
        self, *,
        input_size: int,
        hidden_size: int,   # H_res — reservoir state dim
        out_size: int,      # H — trainable model hidden dim
        n_cols: int,
        r_min: float = 0.90,
        r_max: float = 0.999,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_cols      = n_cols

        if n_cols == 0:
            return

        radii = [r_min + (r_max - r_min) * c / max(n_cols - 1, 1) for c in range(n_cols)]
        self.register_buffer('spectral_r', torch.tensor(radii, dtype=torch.float32))

        W_list = []
        for r_c in radii:
            W = torch.empty(hidden_size, input_size)
            nn.init.orthogonal_(W)
            W_list.append(W * (1.0 - r_c) ** 0.5)   # scale so steady-state ||h|| ≈ 1
        self.register_buffer('W_res', torch.stack(W_list, dim=0))  # [C_res, H_res, H]

        # trainable read projections: [H_res → H]
        self.res_proj = nn.ModuleList([
            nn.Linear(hidden_size, out_size, bias=False) for _ in range(n_cols)
        ])
        for proj in self.res_proj:
            nn.init.normal_(proj.weight, 0.0, 0.01)  # start near-zero contribution

    def forward(
        self,
        x_in: torch.Tensor,        # [B, input_size]
        h_res_prev: torch.Tensor,  # [C_res, B, H_res]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_cols == 0:
            return h_res_prev, h_res_prev   # both empty [0, B, H_res]

        h_new_list = []
        for c in range(self.n_cols):
            r_c   = self.spectral_r[c]
            W_c   = self.W_res[c]   # [H_res, H]
            h_new = r_c * h_res_prev[c] + (1.0 - r_c) * torch.tanh(F.linear(x_in, W_c))
            h_new_list.append(h_new.detach())   # no gradient through reservoir dynamics

        h_res_new = torch.stack(h_new_list, dim=0)   # [C_res, B, H_res]
        y_res = torch.stack(
            [self.res_proj[c](h_res_new[c]) for c in range(self.n_cols)], dim=0
        )   # [C_res, B, H]
        return y_res, h_res_new


class HarmonicGridRNN(nn.Module):
    """Harmonic Grid RNN — integrates spectral LRU, surprise-delta memory,
    frozen reservoir, and Hopfield attention into a unified architecture.
    """

    def __init__(
        self, *,
        input_size: int,
        embedding_size: int,
        output_size: int,
        hidden_size: int,
        n_layers: int,
        n_columns: int,
        n_attn_heads: int,
        messaging: str = 'post',   # kept for config compat, always post-messaging
        dropout: float = 0.0,
        ffn_expansion: int = 2,
        attn_dropout: float = 0.0,
        # Spectral LRU: 2D hierarchy
        r_min_col: float = 0.3,      # r_max for fastest column (col 0, layer 0)
        r_min_layers: float = 0.7,   # r_max ceiling at layer 0
        r_max_layers: float = 0.999, # r_max ceiling at last layer
        # Surprise-Delta Memory
        dk: int | None = None,        # key dim; default H // 4
        dv: int | None = None,        # value dim; default H
        ema_beta: float = 0.9,        # EMA beta for top layer (slowest write)
        ema_beta_min: float = 0.7,    # EMA beta for bottom layer (fastest write); v3
        delta_decay: float = 0.99,    # decay for top layer (slowest)
        delta_decay_min: float = 0.95,# decay for bottom layer (fastest)
        lam_base: float = 0.01,
        # Frozen Reservoir (set n_reservoir_cols=0 to disable)
        n_reservoir_cols: int = 0,
        reservoir_hidden_size: int = 64,
        r_res_min: float = 0.90,
        r_res_max: float = 0.999,
        multi_col_head: bool = True,  # True=mean(cols), False=col0 only (for RL)
        **_kwargs,   # absorb extra config keys
    ):
        super().__init__()
        assert n_columns > 1

        self.input_size       = input_size
        self.output_size      = output_size
        self.n_layers         = n_layers
        self.n_columns        = n_columns
        self.n_reservoir_cols = n_reservoir_cols
        self.hidden_size      = hidden_size - hidden_size % n_attn_heads
        H = self.hidden_size

        self.dk = dk or max(H // 4, 1)
        self.dv = dv or H

        print(
            f'HarmonicGridRNN {n_layers}L×{n_columns}C LRU'
            f' + {n_reservoir_cols}Res'
            f' | hidden={H} heads={n_attn_heads}'
            f' mem=({self.dk}×{self.dv}) ema={ema_beta}'
        )

        self.embedding = nn.Embedding(input_size, embedding_size)

        # Block 1: Spectral LRU cells with 2D r_max[layer, col] hierarchy
        self.cells = nn.ModuleList()
        self.ffns  = nn.ModuleList()
        for l in range(n_layers):
            t        = l / max(n_layers - 1, 1)
            r_base_l = r_min_layers + (r_max_layers - r_min_layers) * t
            row_cells, row_ffns = nn.ModuleList(), nn.ModuleList()
            for c in range(n_columns):
                col_frac = c / max(n_columns - 1, 1)
                r_col    = r_min_col + (r_base_l - r_min_col) * col_frac
                r_col    = max(min(r_col, 0.9999), 1e-4)
                row_cells.append(LRUCell(
                    input_size  = self._cell_input_dim(l, c, embedding_size),
                    hidden_size = H,
                    r_min       = 0.0,
                    r_max       = r_col,
                    max_phase   = math.pi * 2 / 3,
                ))
                row_ffns.append(PositionwiseFFN(H, expansion=ffn_expansion, dropout=dropout))
            self.cells.append(row_cells)
            self.ffns.append(row_ffns)

        # Embedding residual skip: direct gradient path from head to embedding at each layer
        self.embed_skip = nn.ModuleList([
            nn.Linear(embedding_size, H, bias=False) for _ in range(n_layers)
        ])
        for skip in self.embed_skip:
            nn.init.normal_(skip.weight, 0.0, 0.01)

        # Block 2: Surprise-Delta Memory — one per layer, per-layer ema schedule
        self.mem_layers = nn.ModuleList([
            SurpriseDeltaMemory(
                hidden_size = H,
                n_cols      = n_columns,
                dk          = self.dk,
                dv          = self.dv,
                ema_beta    = ema_beta_min + (ema_beta - ema_beta_min)
                              * l / max(n_layers - 1, 1),  # 0.7 → 0.9 per layer
                delta_decay = delta_decay,
                decay_min   = delta_decay_min,
                lam_base    = lam_base,
                layer_idx   = l,
            )
            for l in range(n_layers)
        ])

        # Block 3: Frozen Reservoir — one per layer
        self.reservoirs = nn.ModuleList([
            FrozenReservoir(
                input_size  = H,
                hidden_size = reservoir_hidden_size,
                out_size    = H,
                n_cols      = n_reservoir_cols,
                r_min       = r_res_min,
                r_max       = r_res_max,
            )
            for _ in range(n_layers)
        ])

        # Block 4: Hopfield over all columns (trainable + reservoir)
        self.attn          = nn.ModuleList([
            HopfieldMessageLayer(H, n_attn_heads, attn_dropout) for _ in range(n_layers)
        ])
        self.attn_gates    = nn.ModuleList([
            nn.Linear(2 * H, 1) for _ in range(n_layers)
        ])
        for gate in self.attn_gates:
            nn.init.constant_(gate.bias, -2.0)  # sigmoid(-2)≈0.12 — Hopfield starts weak but active
        # normalize inputs to Hopfield to prevent attractor collapse (col norm explosion)
        self.pre_attn_norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(n_layers)])

        # inter-layer normalization: prevents magnitude explosion across layers
        self.out_norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(n_layers)])

        self.multi_col_head = multi_col_head
        # learned column weights for multi-col head (model decides which timescale matters)
        self.col_weights = nn.Parameter(torch.zeros(n_columns))
        self.head = nn.Linear(H, output_size)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(n_params)}')

    # -------------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,
        state=None,
        return_attn: bool = False,
        return_assoc_loss: bool = False,
        store_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
    ):
        tokens  = to_torch(tokens)
        x_embed = self.embedding(tokens.view(-1))   # [B, E]

        new_state, y_top, extras = self._grid_step(x_embed, state=state)

        y = self.head(y_top)

        assoc_loss = torch.tensor(0.0, device=y.device, dtype=y.dtype)
        if return_assoc_loss and store_mask is not None and query_mask is not None:
            assoc_loss = self._assoc_loss(y_top, store_mask, query_mask)

        if return_attn:
            if return_assoc_loss:
                return y, new_state, extras, assoc_loss
            return y, new_state, extras
        if return_assoc_loss:
            return y, new_state, assoc_loss
        return y, new_state

    def _grid_step(self, x_embed: torch.Tensor, *, state):
        B = x_embed.shape[0]
        H = self.hidden_size
        C = self.n_columns

        if state is None:
            state = self.init_state(B)

        h_s, h_res_s, W_s, m_s, v2_s, y_prev_s = state

        h_new_all     = []
        h_res_new_all = []
        W_new_all     = []
        m_new_all     = []
        v2_new_all    = []
        y_prev_new_all = []
        attn_list     = []
        gate_list     = []
        mem_stats_all = []   # [L] of dicts — memory diagnostics per layer
        col_stats_all = []   # [L] of dicts — column diagnostics per layer

        x_cols = self._prepare_grid_input(x_embed, B)   # list[C] of [B, E]

        for l in range(self.n_layers):
            # Block 1: Spectral LRU + FFN
            y_lru_list = []
            h_new_cols = []
            for c in range(self.n_columns):
                y_c, h_c_full = self.cells[l][c](x_cols[c], h_s[l, c])
                y_c = self.ffns[l][c](y_c)
                y_lru_list.append(y_c)
                # detach Im: gradient already flowed through LRUCell; Im accumulation → OOM
                h_new_c = torch.cat([h_c_full[:, :H], h_c_full[:, H:].detach()], dim=-1)
                h_new_cols.append(h_new_c)

            y_lru = torch.stack(y_lru_list, dim=0)   # [C, B, H]

            # Embedding residual skip (same projection broadcast across all columns)
            y_lru = y_lru + self.embed_skip[l](x_embed).unsqueeze(0)  # [1, B, H] broadcast

            # Block 2: Surprise-Delta Memory
            y_mem, W_new, m_new, v2_new, mem_stats = self.mem_layers[l](
                y_lru, W_s[l], m_s[l], v2_s[l], y_prev_s[l],
            )
            mem_stats_all.append(mem_stats)
            y_aug = y_lru + y_mem   # [C, B, H]

            # Block 3: Frozen Reservoir (fed by first trainable column output)
            y_res, h_res_new = self.reservoirs[l](y_aug[0], h_res_s[l])
            # y_res: [C_res, B, H];  h_res_new: [C_res, B, H_res]

            # Block 4: Hopfield over trainable + reservoir columns
            # normalize before attention to prevent attractor collapse (col norm explosion)
            y_for_attn = self.pre_attn_norms[l](y_aug.reshape(C * B, H)).view(C, B, H)
            if self.n_reservoir_cols > 0:
                all_cols = torch.cat([y_for_attn, y_res], dim=0)   # [C+C_res, B, H]
            else:
                all_cols = y_for_attn

            msgs, attn_w = self.attn[l](all_cols)
            attn_list.append(attn_w)

            msgs_train = msgs[:self.n_columns]   # [C, B, H]
            g = torch.sigmoid(self.attn_gates[l](
                torch.cat([y_aug, msgs_train], dim=-1)   # [C, B, 2H]
            ))   # [C, B, 1]
            gate_list.append(g)
            y_out = (1.0 - g) * y_aug + g * msgs_train   # [C, B, H]

            h_new_all.append(torch.stack(h_new_cols, dim=0))   # [C, B, 2H]
            h_res_new_all.append(h_res_new)
            W_new_all.append(W_new)
            m_new_all.append(m_new)
            v2_new_all.append(v2_new)
            y_prev_new_all.append(y_lru.detach())

            # column diagnostics: diversity and per-column norms
            with torch.no_grad():
                col_norms  = y_out.norm(dim=-1).mean(dim=-1)         # [C] — per-col activation norm
                diversity  = y_out.std(dim=0).mean().item()           # scalar — spread across cols
                gate_mean  = g.mean().item()                          # scalar — Hopfield gate strength
                col_stats_all.append({
                    'diversity': diversity,
                    'gate':      gate_mean,
                    **{f'col{c}_norm': col_norms[c].item() for c in range(C)},
                })

            # normalize before passing to next layer — prevents inter-layer magnitude growth
            y_out = self.out_norms[l](y_out.reshape(C * B, H)).view(C, B, H)
            x_cols = [y_out[c] for c in range(self.n_columns)]

        new_state = HarmonicState(
            h      = torch.stack(h_new_all,      dim=0),   # [L, C, B, 2H]
            h_res  = torch.stack(h_res_new_all,  dim=0),   # [L, C_res, B, H_res]
            W      = torch.stack(W_new_all,      dim=0),   # [L, B, dk, dv]
            m      = torch.stack(m_new_all,      dim=0),   # [L, B]
            v2     = torch.stack(v2_new_all,     dim=0),   # [L, B, dk]
            y_prev = torch.stack(y_prev_new_all, dim=0),   # [L, C, B, H]
        )
        # v3: average all columns → all temporal scales contribute to prediction
        # learned weighted sum over columns (or col0 for RL)
        if self.multi_col_head:
            w     = F.softmax(self.col_weights, dim=0)          # [C]
            y_top = (y_out * w.view(C, 1, 1)).sum(0)            # [B, H]
        else:
            y_top = y_out[0]
        with torch.no_grad():
            head_w = F.softmax(self.col_weights, dim=0).tolist() if self.multi_col_head else []
        extras = {
            'attn_weights': attn_list,
            'gates':        gate_list,
            'mem_stats':    mem_stats_all,
            'col_stats':    col_stats_all,
            'head_weights': head_w,   # [C] learned column weights in prediction head
        }
        return new_state, y_top, extras

    def _assoc_loss(
        self,
        z: torch.Tensor,
        store_mask: torch.Tensor,
        query_mask: torch.Tensor,
        margin: float = 0.5,
    ) -> torch.Tensor:
        s_idx = store_mask.nonzero(as_tuple=True)[0]
        q_idx = query_mask.nonzero(as_tuple=True)[0]
        if s_idx.numel() == 0 or q_idx.numel() == 0:
            return z.new_tensor(0.0)
        n        = min(s_idx.numel(), q_idx.numel())
        h_store  = F.normalize(z[s_idx[:n]], dim=-1)
        h_query  = F.normalize(z[q_idx[:n]], dim=-1)
        sim      = torch.matmul(h_query, h_store.T)       # [n, n]
        cos_pos  = sim.diagonal()
        cos_neg  = sim.masked_fill(torch.eye(n, device=z.device, dtype=torch.bool), -1.0).max(dim=-1).values
        return (-cos_pos + F.relu(cos_neg + margin)).mean()

    def init_state(self, bsz: int) -> HarmonicState:
        dev, dt  = self.head.weight.device, self.head.weight.dtype
        L, C, H  = self.n_layers, self.n_columns, self.hidden_size
        C_res    = self.n_reservoir_cols
        H_res    = self.reservoirs[0].hidden_size if C_res > 0 else 1
        return HarmonicState(
            h      = torch.zeros(L, C,    bsz, 2 * H,          device=dev, dtype=dt),
            h_res  = torch.zeros(L, C_res, bsz, H_res,          device=dev, dtype=dt),
            W      = torch.zeros(L, bsz, self.dk, self.dv,      device=dev, dtype=dt),
            m      = torch.zeros(L, bsz,                         device=dev, dtype=dt),
            v2     = torch.ones( L, bsz, self.dk,                device=dev, dtype=dt),
            y_prev = torch.zeros(L, C, bsz, H,                   device=dev, dtype=dt),
        )

    def reset_state(self, state, reset_mask) -> HarmonicState:
        if state is None:
            return self.init_state(reset_mask.shape[0])
        if not reset_mask.any():
            return state
        keep = (~reset_mask.bool()).to(dtype=state.h.dtype, device=state.h.device)
        reset = reset_mask.bool()
        return HarmonicState(
            h      = state.h     * keep.view(1, 1, -1, 1),   # [L, C, B, 2H]
            h_res  = state.h_res * keep.view(1, 1, -1, 1),   # [L, C_res, B, H_res]
            W      = state.W     * keep.view(1, -1, 1, 1),   # [L, B, dk, dv]
            m      = state.m     * keep.view(1, -1),          # [L, B]
            v2     = torch.where(reset.view(1, -1, 1), torch.ones_like(state.v2),   state.v2),
            y_prev = state.y_prev * keep.view(1, 1, -1, 1),  # zero on reset — no false velocity
        )

    def detach_state(self, state) -> HarmonicState | None:
        if state is None:
            return None
        return HarmonicState(
            h      = state.h.detach(),
            h_res  = state.h_res.detach(),
            W      = state.W.detach(),
            m      = state.m.detach(),
            v2     = state.v2.detach(),
            y_prev = state.y_prev.detach(),
        )

    @staticmethod
    def flatten_extras_stats(extras: dict) -> dict:
        out = {}
        for l, ms in enumerate(extras.get('mem_stats', [])):
            for k, v in ms.items():
                out[f'mem/{k}/L{l}'] = v
        for l, cs in enumerate(extras.get('col_stats', [])):
            for k, v in cs.items():
                out[f'col/{k}/L{l}'] = v
        for c, w in enumerate(extras.get('head_weights', [])):
            out[f'col/head_w{c}'] = w
        return out

    def _cell_input_dim(self, ix_layer: int, ix_col: int, embedding_size: int) -> int:
        if ix_layer == 0:
            return embedding_size if ix_col == 0 else self.hidden_size
        return self.hidden_size

    def _prepare_grid_input(self, x: torch.Tensor, bsz: int) -> list:
        H = self.hidden_size
        dummy = torch.zeros(bsz, H, device=x.device, dtype=x.dtype)
        return [x] + [dummy] * (self.n_columns - 1)
