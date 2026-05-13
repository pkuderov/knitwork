from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch


class LRUCell(nn.Module):
    """Single-step Linear Recurrent Unit (Orvieto et al., ICML 2023).

    State: [Re | Im], shape (batch, 2*hidden_size).
    Lambda: exp(-exp(nu) + i*exp(theta)) — stable reparametrization keeping |lambda| in (0,1).
    """

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        r_min: float = 0.4,
        r_max: float = 0.99,
        max_phase: float = math.pi * 2 / 3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.r_dim = hidden_size

        nu_log = torch.empty(hidden_size).uniform_(
            math.log(-math.log(r_max + 1e-8)),
            math.log(-math.log(r_min + 1e-8)),
        )
        theta_log = torch.empty(hidden_size).uniform_(math.log(1e-4), math.log(max_phase))
        self.nu_log    = nn.Parameter(nu_log)
        self.theta_log = nn.Parameter(theta_log)

        # complex input projection B
        self.B_re = nn.Linear(input_size, hidden_size, bias=False)
        self.B_im = nn.Linear(input_size, hidden_size, bias=False)
        # output projection C
        self.C_re = nn.Linear(hidden_size, hidden_size, bias=False)
        self.C_im = nn.Linear(hidden_size, hidden_size, bias=False)
        # skip connection D
        self.D    = nn.Linear(input_size, hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.B_re.weight, 0.0, 0.01)
        nn.init.normal_(self.B_im.weight, 0.0, 0.01)
        nn.init.orthogonal_(self.C_re.weight)
        nn.init.orthogonal_(self.C_im.weight)
        nn.init.zeros_(self.D.bias)
        if self.input_size == self.hidden_size:
            nn.init.eye_(self.D.weight)
        else:
            nn.init.normal_(self.D.weight, 0.0, 0.02)

    def _get_lambda_gamma(self):
        # |lambda| in (0,1), gamma normalizes input contribution inversely to memory
        r      = torch.exp(-torch.exp(self.nu_log))
        theta  = torch.exp(self.theta_log)
        lam_re = r * torch.cos(theta)
        lam_im = r * torch.sin(theta)
        gamma  = torch.sqrt(torch.clamp(1.0 - r * r, min=1e-6))
        return lam_re, lam_im, gamma

    def forward(
        self,
        x: torch.Tensor,                       # [B, input_size]
        state: Optional[torch.Tensor] = None,  # [B, 2*hidden_size]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        if state is None:
            state = torch.zeros(B, 2 * self.r_dim, device=x.device, dtype=x.dtype)

        h_re = state[:, :self.r_dim]
        h_im = state[:, self.r_dim:]

        lam_re, lam_im, gamma = self._get_lambda_gamma()
        bx_re = self.B_re(x)
        bx_im = self.B_im(x)

        # complex multiply h_new = lambda * h_prev + gamma * B * x
        new_re = lam_re * h_re - lam_im * h_im + gamma * bx_re   # [B, H]
        new_im = lam_re * h_im + lam_im * h_re + gamma * bx_im   # [B, H]

        y = self.norm(self.C_re(new_re) - self.C_im(new_im) + self.D(x))
        h_new = torch.cat([new_re, new_im], dim=-1)               # [B, 2*H]
        return y, h_new


class HopfieldMessageLayer(nn.Module):
    """Modern Hopfield message passing with learnable beta per head."""

    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        self.dim      = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.W_q      = nn.Linear(dim, dim, bias=False)
        self.W_k      = nn.Linear(dim, dim, bias=False)
        self.W_v      = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        # one log_beta per head; init as log(1/sqrt(d_k)) = standard scaling
        self.log_beta     = nn.Parameter(
            torch.full((num_heads,), math.log(1.0 / math.sqrt(self.head_dim)))
        )
        self.norm         = nn.LayerNorm(dim)
        self.attn_dropout = nn.Dropout(p=attn_dropout)

        # small init: message negligible at start of training
        nn.init.normal_(self.out_proj.weight, 0.0, 0.001)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: [cols, B, dim]
        C, B, D = h.shape
        # projections: [heads, B, cols, head_dim]
        q = self.W_q(h).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        k = self.W_k(h).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        v = self.W_v(h).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)

        beta   = self.log_beta.exp().view(self.num_heads, 1, 1, 1)
        scores = beta * torch.matmul(q, k.transpose(-2, -1))
        attn   = self.attn_dropout(torch.softmax(scores, dim=-1))

        out = torch.matmul(attn, v)                                   # [heads, B, cols, head_dim]
        out = out.permute(2, 1, 0, 3).contiguous().view(C, B, D)     # [cols, B, dim]
        return self.norm(self.out_proj(out)), attn


class PositionwiseFFN(nn.Module):
    """Pre-LN GELU FFN with residual; adds nonlinearity to linear LRU block."""

    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class HopfieldGridLRU(nn.Module):
    """Hierarchical Grid RNN with LRU cells and Modern Hopfield message passing."""

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
        use_bias: bool = True,
        dropout: float = 0.0,
        ffn_expansion: int = 2,
        attn_dropout: float = 0.0,
        lru_r_min: float = 0.4,
        lru_r_max: float = 0.99,
        lru_max_phase: float = math.pi * 2 / 3,
    ):
        super().__init__()
        self.input_size     = input_size
        self.embedding_size = embedding_size
        self.output_size    = output_size
        self.n_layers       = n_layers
        self.n_columns      = n_columns
        self.n_attn_heads   = n_attn_heads

        assert n_columns > 1, "n_columns must be > 1"
        self.embedding   = nn.Embedding(input_size, embedding_size)
        self.hidden_size = hidden_size - hidden_size % n_attn_heads
        self.use_postmsg = (messaging == "post")

        print(
            f'HopfieldGridLRU: {n_layers}L x {n_columns}C LRU'
            f' hidden={self.hidden_size} heads={n_attn_heads} messaging={messaging}'
        )

        self.cells      = nn.ModuleList()
        self.ffns       = nn.ModuleList()
        self.attn       = nn.ModuleList()
        self.attn_gates = nn.ModuleList()

        for layer in range(n_layers):
            self.cells.append(nn.ModuleList([
                LRUCell(
                    input_size=self._cell_input_dim(layer, icol),
                    hidden_size=self.hidden_size,
                    r_min=lru_r_min, r_max=lru_r_max, max_phase=lru_max_phase,
                )
                for icol in range(n_columns)
            ]))
            self.ffns.append(nn.ModuleList([
                PositionwiseFFN(self.hidden_size, expansion=ffn_expansion, dropout=dropout)
                for _ in range(n_columns)
            ]))
            self.attn.append(HopfieldMessageLayer(self.hidden_size, n_attn_heads, attn_dropout))
            if self.use_postmsg:
                self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

        self.head = nn.Linear(self.hidden_size, output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(
        self,
        tokens: torch.Tensor,
        state=None,
        return_attn: bool = False,
        return_assoc_loss: bool = False,
        store_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
    ):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2

        x = self.embedding(tokens.view(-1))   # [B, embedding_size]

        if self.use_postmsg:
            h_new, all_attn, all_gates = self._grid_step_postmsg(x, h=state)
        else:
            h_new, all_attn, all_gates = self._grid_step_premsg(x, h=state)

        # Re-part of top layer, first col  [B, H]
        z = h_new[-1, 0, :, :self.hidden_size]
        y = self.head(z)

        assoc_loss = torch.tensor(0.0, device=y.device, dtype=y.dtype)
        if return_assoc_loss and store_mask is not None and query_mask is not None:
            assoc_loss = self._assoc_loss(z, store_mask, query_mask)

        if return_attn:
            extras = {"attn_weights": all_attn, "gates": all_gates}
            if return_assoc_loss:
                return y, h_new, extras, assoc_loss
            return y, h_new, extras

        if return_assoc_loss:
            return y, h_new, assoc_loss

        return y, h_new

    def _grid_step_postmsg(self, x: torch.Tensor, *, h):
        x_list    = self._prepare_grid_input(x)
        h_layers  = []
        all_attn  = []
        all_gates = []

        for layer_idx, (cells, ffns, attn_layer, gate) in enumerate(
            zip(self.cells, self.ffns, self.attn, self.attn_gates)
        ):
            hl_re_list, hl_full_list = [], []
            for ic in range(self.n_columns):
                prev_state = h[layer_idx, ic] if h is not None else None
                y_lru, h_new_full = cells[ic](x_list[ic], prev_state)
                hl_re_list.append(ffns[ic](y_lru))
                hl_full_list.append(h_new_full)

            hl_re   = torch.stack(hl_re_list,   dim=0)   # [cols, B, H]
            hl_full = torch.stack(hl_full_list, dim=0)   # [cols, B, 2*H]

            msg, attn_w = attn_layer(hl_re)
            all_attn.append(attn_w)

            g = torch.sigmoid(gate(torch.cat([hl_re, msg], dim=-1)))   # [cols, B, 1]
            all_gates.append(g)
            hl_re_gated = (1.0 - g) * hl_re + g * msg

            # Im is detached: gradient already flowed through LRUCell;
            # accumulating graph through Im causes OOM on long rollouts
            hl_im_stop  = hl_full[:, :, self.hidden_size:].detach()
            h_layers.append(torch.cat([hl_re_gated, hl_im_stop], dim=-1))
            x_list = hl_re_gated

        return torch.stack(h_layers, dim=0), all_attn, all_gates   # [layers, cols, B, 2*H]

    def _grid_step_premsg(self, x: torch.Tensor, *, h):
        x_list    = self._prepare_grid_input(x)
        h_layers  = []
        all_attn  = []
        all_gates = []

        for layer_idx, (cells, ffns, attn_layer) in enumerate(
            zip(self.cells, self.ffns, self.attn)
        ):
            if layer_idx == 0 or h is None:
                prev_re = torch.zeros(
                    self.n_columns, x.shape[0], self.hidden_size,
                    device=x.device, dtype=x.dtype,
                )
            else:
                prev_re = h[layer_idx - 1, :, :, :self.hidden_size]

            msg, attn_w = attn_layer(prev_re)
            all_attn.append(attn_w)
            all_gates.append(msg)   # placeholder for extras compatibility

            hl_re_list, hl_full_list = [], []
            for ic in range(self.n_columns):
                prev_state = h[layer_idx, ic] if h is not None else None
                inp = torch.cat([x_list[ic], msg[ic]], dim=-1)
                y_lru, h_new_full = cells[ic](inp, prev_state)
                hl_re_list.append(ffns[ic](y_lru))
                hl_full_list.append(h_new_full)

            hl_re   = torch.stack(hl_re_list,   dim=0)
            hl_full = torch.stack(hl_full_list, dim=0)

            hl_im_stop  = hl_full[:, :, self.hidden_size:].detach()
            h_layers.append(torch.cat([hl_re, hl_im_stop], dim=-1))
            x_list = hl_re

        return torch.stack(h_layers, dim=0), all_attn, all_gates

    def _assoc_loss(
        self,
        z: torch.Tensor,
        store_mask: torch.Tensor,
        query_mask: torch.Tensor,
        margin: float = 0.5,
    ) -> torch.Tensor:
        """Contrastive associative loss: push query close to its store, away from others."""
        s_idx = store_mask.nonzero(as_tuple=True)[0]
        q_idx = query_mask.nonzero(as_tuple=True)[0]
        if s_idx.numel() == 0 or q_idx.numel() == 0:
            return z.new_tensor(0.0)

        n = min(s_idx.numel(), q_idx.numel())
        h_store = F.normalize(z[s_idx[:n]], dim=-1)
        h_query = F.normalize(z[q_idx[:n]], dim=-1)

        sim_matrix = torch.matmul(h_query, h_store.T)         # [n, n]
        cos_pos = sim_matrix.diagonal()
        eye_mask = torch.eye(n, device=z.device, dtype=torch.bool)
        cos_neg = sim_matrix.masked_fill(eye_mask, -1.0).max(dim=-1).values
        return (-cos_pos + F.relu(cos_neg + margin)).mean()

    def _cell_input_dim(self, ix_layer: int, ix_col: int) -> int:
        if ix_layer == 0:
            base = self.embedding_size if ix_col == 0 else self.hidden_size
        else:
            base = self.hidden_size
        if not self.use_postmsg:
            base += self.hidden_size
        return base

    def _prepare_grid_input(self, x: torch.Tensor) -> list:
        # col 0 gets real embedding; others get zeros (communicate via Hopfield only)
        bsz   = x.shape[0]
        dummy = torch.zeros(bsz, self.hidden_size, device=x.device, dtype=x.dtype)
        return [x] + [dummy] * (self.n_columns - 1)

    def reset_state(self, state: torch.Tensor | None, reset_mask: torch.Tensor) -> torch.Tensor:
        if state is None:
            return self.init_state(reset_mask.shape[0])
        if not reset_mask.any():
            return state
        # multiply by keep-mask: cheaper than clone+index; grad through live envs preserved
        keep = (~reset_mask).to(dtype=state.dtype, device=state.device)
        return state * keep.view(1, 1, -1, 1)   # broadcast [layers, cols, batch, 2*H]

    def detach_state(self, state: torch.Tensor | None) -> torch.Tensor | None:
        return state.detach() if state is not None else None

    def init_state(self, bsz: int) -> torch.Tensor:
        # [layers, cols, batch, 2*hidden]
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, 2 * self.hidden_size,
            device=self.head.weight.device,
            dtype=self.head.weight.dtype,
        )
