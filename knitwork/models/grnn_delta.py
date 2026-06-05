from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch
from knitwork.models.lru import LRUBlock
from knitwork.models.hgrnn_lru import HopfieldMessageLayer


class DeltaGridState(NamedTuple):
    h:      torch.Tensor  # [L, C, B, 2H]
    W_fast: torch.Tensor  # [L, C, B, dk_f * dv_f]
    W_slow: torch.Tensor  # [L, C, B, dk_s * dv_s]


class TwoScaleMemLayer(nn.Module):
    """Vectorized two-timescale delta-rule memory for all columns in a layer.

    Shared projections across columns + per-column bias for key/query specialization.
    Processes all C columns in one batched call via [C*B, H] reshape.
    """

    def __init__(self, *, n_cols, hidden_size, dk_f, dv_f, dk_s, dv_s):
        super().__init__()
        self.n_cols = n_cols
        self.dk_f, self.dv_f = dk_f, dv_f
        self.dk_s, self.dv_s = dk_s, dv_s

        # fast delta memory (dk_f = H//8 typically)
        self.Wk_f = nn.Linear(hidden_size, dk_f, bias=False)
        self.Wv_f = nn.Linear(hidden_size, dv_f, bias=False)
        self.Wq_f = nn.Linear(hidden_size, dk_f, bias=False)
        self.Wg_f = nn.Linear(hidden_size, 1)
        self.Wo_f = nn.Linear(dv_f, hidden_size, bias=False)
        self.kb_f = nn.Parameter(torch.zeros(n_cols, dk_f))  # [C, dk_f] per-col bias
        self.qb_f = nn.Parameter(torch.zeros(n_cols, dk_f))

        # slow delta memory (dk_s = H//4 typically)
        self.Wk_s = nn.Linear(hidden_size, dk_s, bias=False)
        self.Wv_s = nn.Linear(hidden_size, dv_s, bias=False)
        self.Wq_s = nn.Linear(hidden_size, dk_s, bias=False)
        self.Wg_s = nn.Linear(hidden_size, 1)
        self.Wo_s = nn.Linear(dv_s, hidden_size, bias=False)
        self.kb_s = nn.Parameter(torch.zeros(n_cols, dk_s))
        self.qb_s = nn.Parameter(torch.zeros(n_cols, dk_s))

        self.norm = nn.LayerNorm(hidden_size)

        # small output init so memory contributes little at start of training
        nn.init.normal_(self.Wo_f.weight, 0.0, 0.001)
        nn.init.normal_(self.Wo_s.weight, 0.0, 0.001)

    def _delta_step(self, y_flat, W, Wk, Wv, Wq, Wg, Wo, kb, qb, decay, dk, dv, C, B):
        CB = C * B
        # per-col biases: [C, dk] → [C, B, dk] → [CB, dk]
        k_bias = kb.unsqueeze(1).expand(C, B, -1).reshape(CB, -1)
        q_bias = qb.unsqueeze(1).expand(C, B, -1).reshape(CB, -1)

        k = F.normalize(Wk(y_flat) + k_bias, dim=-1)   # [CB, dk]
        v = Wv(y_flat)                                    # [CB, dv]
        q = F.normalize(Wq(y_flat) + q_bias, dim=-1)   # [CB, dk]
        g = torch.sigmoid(Wg(y_flat))                    # [CB, 1]

        W_mat = W.reshape(CB, dk, dv)
        # delta rule: W ← decay*W + g * k ⊗ (v - W^T·k)
        v_old  = torch.bmm(W_mat, k.unsqueeze(-1)).squeeze(-1)      # [CB, dv]
        delta  = torch.einsum('bi,bj->bij', k, v - v_old)           # [CB, dk, dv]
        W_new  = decay * W_mat + g.unsqueeze(-1) * delta            # [CB, dk, dv]
        m      = torch.bmm(W_new.transpose(-2, -1), q.unsqueeze(-1)).squeeze(-1)  # [CB, dv]
        return Wo(m), W_new.reshape(C, B, dk * dv)

    def forward(
        self,
        y: torch.Tensor,        # [C, B, H]
        W_fast: torch.Tensor,   # [C, B, dk_f * dv_f]
        W_slow: torch.Tensor,   # [C, B, dk_s * dv_s]
        decay_f: float,
        decay_s: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        C, B, _ = y.shape
        y_flat = y.reshape(C * B, -1)

        m_f, Wf_new = self._delta_step(
            y_flat, W_fast, self.Wk_f, self.Wv_f, self.Wq_f, self.Wg_f, self.Wo_f,
            self.kb_f, self.qb_f, decay_f, self.dk_f, self.dv_f, C, B,
        )
        m_s, Ws_new = self._delta_step(
            y_flat, W_slow, self.Wk_s, self.Wv_s, self.Wq_s, self.Wg_s, self.Wo_s,
            self.kb_s, self.qb_s, decay_s, self.dk_s, self.dv_s, C, B,
        )

        m_out = self.norm(m_f + m_s).view(C, B, -1)  # [C, B, H]
        return m_out, Wf_new, Ws_new


class GridDelta(nn.Module):
    """Grid RNN with two-timescale delta-rule associative memory per layer.

    Each layer processes:
      1. LRUBlock per column (per-layer × per-col r_max hierarchy)
      2. TwoScaleMemLayer (vectorized fast + slow delta memories)
      3. HopfieldMessageLayer (cross-column attention with learnable beta)
      4. Gated post-messaging

    Optionally: top layer reads from bottom layer's slow memory (cross-layer skip).
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
        messaging: str = 'post',
        dropout: float = 0.0,
        ff_mult: int = 2,
        # LRU temporal hierarchy (per-layer interpolation)
        r_min_layers: float = 0.0,
        r_max_layers: float = 0.999,
        lru_r_per_col: bool = True,
        # two-scale delta memory dimensions
        dk_fast: int | None = None,
        dv_fast: int | None = None,
        dk_slow: int | None = None,
        dv_slow: int | None = None,
        # per-layer decay (list of length n_layers or scalar); None → auto
        mem_decay_fast=None,
        mem_decay_slow=None,
        # optional cross-layer memory skip
        use_cross_layer_skip: bool = False,
    ):
        super().__init__()
        assert n_columns > 1
        self.input_size     = input_size
        self.embedding_size = embedding_size
        self.output_size    = output_size
        self.n_layers       = n_layers
        self.n_columns      = n_columns

        self.hidden_size = hidden_size - hidden_size % n_attn_heads
        H = self.hidden_size

        self.dk_f = dk_fast or max(H // 8, 1)
        self.dv_f = dv_fast or max(H // 8, 1)
        self.dk_s = dk_slow or max(H // 4, 1)
        self.dv_s = dv_slow or max(H // 4, 1)

        self.use_cross_layer_skip = use_cross_layer_skip

        def _to_list(val, L):
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return [float(val)] * L
            return [float(v) for v in val]

        df = _to_list(mem_decay_fast, n_layers)
        ds = _to_list(mem_decay_slow, n_layers)
        if df is None:
            df = [0.3 + 0.5 * l / max(n_layers - 1, 1) for l in range(n_layers)]
        if ds is None:
            ds = [0.95 + 0.049 * l / max(n_layers - 1, 1) for l in range(n_layers)]
        self.decay_fast = df
        self.decay_slow = ds

        print(
            f'GridDelta {n_layers}L×{n_columns}C LRU hidden={H}'
            f'  mem=({self.dk_f}×{self.dv_f}+{self.dk_s}×{self.dv_s})'
            f'  cross_skip={use_cross_layer_skip}'
        )

        self.embedding = nn.Embedding(input_size, embedding_size)

        # LRU cells: per-layer r_max (interpolated) × per-col variation
        self.cells = nn.ModuleList()
        for l in range(n_layers):
            t = l / max(n_layers - 1, 1)
            r_base = r_min_layers + (r_max_layers - r_min_layers) * t
            row = nn.ModuleList()
            for c in range(n_columns):
                r_col = r_base * (0.85 + 0.15 * c / max(n_columns - 1, 1)) if lru_r_per_col else r_base
                r_col = max(min(r_col, 0.9999), 1e-4)
                row.append(LRUBlock(
                    input_size=self._cell_input_dim(l, c),
                    hidden_size=H, ff_mult=ff_mult,
                    r_min=0.0, r_max=r_col, dropout=dropout,
                ))
            self.cells.append(row)

        # per-layer two-scale memory
        self.mem_layers = nn.ModuleList([
            TwoScaleMemLayer(
                n_cols=n_columns, hidden_size=H,
                dk_f=self.dk_f, dv_f=self.dv_f,
                dk_s=self.dk_s, dv_s=self.dv_s,
            )
            for _ in range(n_layers)
        ])

        # per-layer Hopfield cross-column attention + gating
        self.attn  = nn.ModuleList([HopfieldMessageLayer(H, n_attn_heads) for _ in range(n_layers)])
        self.gates = nn.ModuleList([nn.Linear(2 * H, 1) for _ in range(n_layers)])

        # cross-layer memory skip: top layer reads from bottom layer's slow memory
        if use_cross_layer_skip:
            self.Wq_skip = nn.Linear(H, self.dk_s, bias=False)
            self.Wo_skip = nn.Linear(self.dv_s, H, bias=False)
            nn.init.normal_(self.Wo_skip.weight, 0.0, 0.001)

        self.head = nn.Linear(H, output_size)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(n_params)}')

    def forward(self, tokens: torch.Tensor, state=None, return_attn: bool = False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2

        x = self.embedding(tokens.view(-1))   # [B, E]

        state, y_cols, extras = self._grid_step(x, state=state)

        z = y_cols[0]       # top layer, col 0: [B, H]
        y = self.head(z)

        if return_attn:
            return y, state, extras
        return y, state

    def _grid_step(self, x: torch.Tensor, *, state):
        B = x.shape[0]
        H = self.hidden_size

        if state is None:
            state = self.init_state(B)

        h_s  = state.h       # [L, C, B, 2H]
        Wf_s = state.W_fast  # [L, C, B, dk_f*dv_f]
        Ws_s = state.W_slow  # [L, C, B, dk_s*dv_s]

        h_new_all  = []
        Wf_new_all = []
        Ws_new_all = []
        attn_list  = []
        gate_list  = []

        x_cols = self._prepare_grid_input(x, B)   # list[C] of [B, dim]

        for l in range(self.n_layers):
            h_layer  = h_s[l]   # [C, B, 2H]
            Wf_layer = Wf_s[l]  # [C, B, dk_f*dv_f]
            Ws_layer = Ws_s[l]  # [C, B, dk_s*dv_s]

            y_lru_list = []
            h_new_cols = []
            for c in range(self.n_columns):
                y_lru, h_n_full = self.cells[l][c](x_cols[c], h_layer[c])
                # detach Im part: gradient flowed through LRUCell, keeping graph avoids OOM
                h_n = torch.cat([h_n_full[:, :H], h_n_full[:, H:].detach()], dim=-1)
                y_lru_list.append(y_lru)
                h_new_cols.append(h_n)

            y_stack = torch.stack(y_lru_list, dim=0)  # [C, B, H]
            h_stack = torch.stack(h_new_cols, dim=0)  # [C, B, 2H]

            # two-scale delta memory (all columns batched)
            m_out, Wf_new, Ws_new = self.mem_layers[l](
                y_stack, Wf_layer, Ws_layer,
                self.decay_fast[l], self.decay_slow[l],
            )
            y_aug = y_stack + m_out   # [C, B, H]

            # cross-column Hopfield attention
            msgs, attn_w = self.attn[l](y_aug)
            attn_list.append(attn_w)

            g = torch.sigmoid(self.gates[l](torch.cat([y_aug, msgs], dim=-1)))  # [C, B, 1]
            gate_list.append(g)
            y_out = (1.0 - g) * y_aug + g * msgs   # [C, B, H]

            # cross-layer memory skip: top layer reads bottom layer's slow memory
            if self.use_cross_layer_skip and l == self.n_layers - 1:
                CB = self.n_columns * B
                q_skip = F.normalize(self.Wq_skip(y_out.reshape(CB, H)), dim=-1)
                # read from bottom layer slow memory (before this step's update)
                W_bot = Ws_s[0].reshape(CB, self.dk_s, self.dv_s)
                m_skip = torch.bmm(W_bot.transpose(-2, -1), q_skip.unsqueeze(-1)).squeeze(-1)
                y_out = y_out + self.Wo_skip(m_skip).view(self.n_columns, B, H)

            h_new_all.append(h_stack)
            Wf_new_all.append(Wf_new)
            Ws_new_all.append(Ws_new)
            x_cols = [y_out[c] for c in range(self.n_columns)]

        new_state = DeltaGridState(
            h      = torch.stack(h_new_all,  dim=0),   # [L, C, B, 2H]
            W_fast = torch.stack(Wf_new_all, dim=0),   # [L, C, B, dk_f*dv_f]
            W_slow = torch.stack(Ws_new_all, dim=0),   # [L, C, B, dk_s*dv_s]
        )
        extras = {'attn_weights': attn_list, 'gates': gate_list}
        return new_state, x_cols, extras

    def init_state(self, bsz: int) -> DeltaGridState:
        dev, dtyp = self.head.weight.device, self.head.weight.dtype
        L, C, H   = self.n_layers, self.n_columns, self.hidden_size
        return DeltaGridState(
            h      = torch.zeros(L, C, bsz, 2 * H,                  device=dev, dtype=dtyp),
            W_fast = torch.zeros(L, C, bsz, self.dk_f * self.dv_f,  device=dev, dtype=dtyp),
            W_slow = torch.zeros(L, C, bsz, self.dk_s * self.dv_s,  device=dev, dtype=dtyp),
        )

    def reset_state(self, state, reset_mask) -> DeltaGridState:
        if state is None:
            return self.init_state(reset_mask.shape[0])
        if not reset_mask.any():
            return state
        keep = (~reset_mask.bool()).to(dtype=state.h.dtype, device=state.h.device)
        k4 = keep.view(1, 1, -1, 1)   # broadcast [L, C, B, dim]
        return DeltaGridState(state.h * k4, state.W_fast * k4, state.W_slow * k4)

    def detach_state(self, state) -> DeltaGridState | None:
        if state is None:
            return None
        return DeltaGridState(state.h.detach(), state.W_fast.detach(), state.W_slow.detach())

    def _cell_input_dim(self, ix_layer: int, ix_col: int) -> int:
        if ix_layer == 0:
            return self.embedding_size if ix_col == 0 else self.hidden_size
        return self.hidden_size

    def _prepare_grid_input(self, x: torch.Tensor, bsz: int) -> list:
        dummy = torch.zeros(bsz, self.hidden_size, device=x.device, dtype=x.dtype)
        return [x] + [dummy] * (self.n_columns - 1)
