from __future__ import annotations

import math
import torch
from torch import nn


class HGRUCell(nn.Module):
    """HGRU with LayerNorm on candidate state.

    Forget gate: λ_t = sigmoid(W_f·x + U_f·h) * (1 - β) + β
    where β is a learnable lower bound (hierarchical timescales).
    """

    def __init__(
        self, *,
        input_size: int,
        hidden_size: int,
        beta_init: float = 0.01,
        use_bias: bool = True,
        learnable_beta: bool = True,
        use_layer_norm: bool = True,
        dtype=torch.float64,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_f = nn.Linear(input_size, hidden_size, bias=use_bias, dtype=dtype)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.W_o = nn.Linear(input_size, hidden_size, bias=use_bias, dtype=dtype)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.W_c = nn.Linear(input_size, hidden_size, bias=use_bias, dtype=dtype)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.ln_c = nn.LayerNorm(hidden_size, dtype=dtype) if use_layer_norm else None

        beta_init = max(1e-4, min(1.0 - 1e-4, beta_init))
        self.beta_raw = nn.Parameter(
            torch.tensor(math.log(beta_init / (1.0 - beta_init)), dtype=dtype),
            requires_grad=learnable_beta,
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.ndim == 2:
                nn.init.orthogonal_(p)
            elif p.ndim == 1 and 'bias' in name:
                nn.init.zeros_(p)

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_raw)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: [B, input_size]  h: [B, hidden_size]
        o_t   = torch.sigmoid(self.W_o(x) + self.U_o(h))
        c_raw = self.W_c(x) + self.U_c(o_t * h)
        if self.ln_c is not None:
            c_raw = self.ln_c(c_raw)
        c_t   = torch.tanh(c_raw)
        lam_t = torch.sigmoid(self.W_f(x) + self.U_f(h)) * (1.0 - self.beta) + self.beta
        return lam_t * h + (1.0 - lam_t) * c_t


class BatchedHGRUColumns(nn.Module):
    """All trainable HGRU columns in parallel via batched matmuls.

    Input:  x [B, input_size], h [B, n_cols, hidden_size]
    Output: [B, n_cols, hidden_size]
    ~3-5x faster than a Python loop over columns.
    """

    def __init__(
        self, *,
        n_cols: int,
        input_size: int,
        hidden_size: int,
        beta_inits: list[float],
        use_bias: bool = True,
        learnable_beta: bool = True,
        use_layer_norm: bool = True,
        dtype=torch.float64,
    ):
        super().__init__()
        self.n_cols = n_cols
        self.hidden_size = hidden_size

        self.W_f = nn.Parameter(torch.empty(n_cols, hidden_size, input_size,  dtype=dtype))
        self.U_f = nn.Parameter(torch.empty(n_cols, hidden_size, hidden_size, dtype=dtype))
        self.W_o = nn.Parameter(torch.empty(n_cols, hidden_size, input_size,  dtype=dtype))
        self.U_o = nn.Parameter(torch.empty(n_cols, hidden_size, hidden_size, dtype=dtype))
        self.W_c = nn.Parameter(torch.empty(n_cols, hidden_size, input_size,  dtype=dtype))
        self.U_c = nn.Parameter(torch.empty(n_cols, hidden_size, hidden_size, dtype=dtype))

        if use_bias:
            self.b_f = nn.Parameter(torch.zeros(n_cols, hidden_size, dtype=dtype))
            self.b_o = nn.Parameter(torch.zeros(n_cols, hidden_size, dtype=dtype))
            self.b_c = nn.Parameter(torch.zeros(n_cols, hidden_size, dtype=dtype))
        else:
            self.b_f = self.b_o = self.b_c = None

        self.ln_c = (
            nn.ModuleList([nn.LayerNorm(hidden_size, dtype=dtype) for _ in range(n_cols)])
            if use_layer_norm else None
        )

        beta_raws = [
            math.log(max(1e-4, min(1 - 1e-4, b)) / (1.0 - max(1e-4, min(1 - 1e-4, b))))
            for b in beta_inits
        ]
        self.beta_raw = nn.Parameter(
            torch.tensor(beta_raws, dtype=dtype), requires_grad=learnable_beta
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for name in ['W_f', 'W_o', 'W_c', 'U_f', 'U_o', 'U_c']:
            p = getattr(self, name)
            for i in range(self.n_cols):
                nn.init.orthogonal_(p[i])

    @property
    def betas(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_raw)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: [B, in]  h: [B, n_cols, hid]
        x_t = x.T.unsqueeze(0).expand(self.n_cols, -1, -1)   # [n_cols, in, B]
        h_t = h.permute(1, 2, 0)                               # [n_cols, hid, B]

        def gate_x(W, b):
            out = torch.bmm(W, x_t).permute(0, 2, 1)           # [n_cols, B, hid]
            return out + b.unsqueeze(1) if b is not None else out

        def gate_h(U, h_src=None):
            src = h_src if h_src is not None else h_t
            return torch.bmm(U, src).permute(0, 2, 1)          # [n_cols, B, hid]

        o_t    = torch.sigmoid(gate_x(self.W_o, self.b_o) + gate_h(self.U_o))
        h_perm = h.permute(1, 0, 2)                             # [n_cols, B, hid]
        oh     = (o_t * h_perm).permute(0, 2, 1)
        c_raw  = gate_x(self.W_c, self.b_c) + gate_h(self.U_c, oh)

        if self.ln_c is not None:
            c_raw = torch.stack([self.ln_c[i](c_raw[i]) for i in range(self.n_cols)])

        c_t   = torch.tanh(c_raw)
        betas = self.betas.view(self.n_cols, 1, 1)
        lam_t = torch.sigmoid(gate_x(self.W_f, self.b_f) + gate_h(self.U_f)) * (1.0 - betas) + betas
        return (lam_t * h_perm + (1.0 - lam_t) * c_t).permute(1, 0, 2)   # [B, n_cols, hid]

    def get_betas_dict(self, layer_i: int) -> dict[str, float]:
        return {f"hgrn/beta/L{layer_i}_C{ci}": b.item()
                for ci, b in enumerate(self.betas.detach())}


class BatchedReservoirColumns(nn.Module):
    """Frozen GRU columns with fixed spectral radius (echo-state network).

    All columns processed in batch via bmm. requires_grad=False throughout.
    """

    def __init__(
        self, *,
        n_cols: int,
        input_size: int,
        hidden_size: int,
        spectral_radii: list[float],
        reservoir_scale: float = 0.1,
        dtype=torch.float64,
    ):
        super().__init__()
        self.n_cols = n_cols
        self.hidden_size = hidden_size

        W_ih = torch.empty(n_cols, 3 * hidden_size, input_size, dtype=dtype)
        W_hh = torch.empty(n_cols, 3 * hidden_size, hidden_size, dtype=dtype)
        for ci in range(n_cols):
            nn.init.uniform_(W_ih[ci], -reservoir_scale, reservoir_scale)
            for gi in range(3):
                block = torch.empty(hidden_size, hidden_size, dtype=dtype)
                nn.init.orthogonal_(block)
                ev = torch.linalg.eigvals(block).abs().max().item()
                if ev > 1e-10:
                    block.mul_(spectral_radii[ci] / ev)
                W_hh[ci, gi * hidden_size:(gi + 1) * hidden_size] = block

        self.W_ih = nn.Parameter(W_ih, requires_grad=False)
        self.W_hh = nn.Parameter(W_hh, requires_grad=False)
        self.b    = nn.Parameter(
            torch.zeros(n_cols, 3 * hidden_size, dtype=dtype), requires_grad=False
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: [B, in]  h: [B, n_cols, hid]
        x_t = x.T.unsqueeze(0).expand(self.n_cols, -1, -1)    # [n_cols, in, B]
        h_t = h.permute(1, 2, 0)                                # [n_cols, hid, B]

        gates = (torch.bmm(self.W_ih, x_t) + torch.bmm(self.W_hh, h_t)
                 + self.b.unsqueeze(-1)).permute(0, 2, 1)       # [n_cols, B, 3*hid]

        hid = self.hidden_size
        r = torch.sigmoid(gates[..., :hid])
        z = torch.sigmoid(gates[..., hid:2 * hid])

        h_p = h.permute(1, 0, 2)                                # [n_cols, B, hid]
        n_x = torch.bmm(self.W_ih[:, 2*hid:], x_t).permute(0, 2, 1)
        n_h = torch.bmm(self.W_hh[:, 2*hid:], (r * h_p).permute(0, 2, 1)).permute(0, 2, 1)
        n   = torch.tanh(n_x + n_h + self.b[:, 2*hid:].unsqueeze(1))

        return ((1.0 - z) * n + z * h_p).permute(1, 0, 2)      # [B, n_cols, hid]
