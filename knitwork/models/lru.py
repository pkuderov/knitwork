from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn


class LRUCell(nn.Module):
    """
    Linear Recurrent Unit.

    State x ∈ ℂ^H.
    Update:
        x_k = Λ · x_{k-1} + γ ⊙ (B_re · u  +  i · B_im · u)
        y_k = C([Re(x_k); Im(x_k)]) + D · u
    где Λ = diag(exp(-exp(ν) + i·exp(θ))), γ = sqrt(1 - |λ|²).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = math.pi * 2,
        use_d_feedthrough: bool = True,
    ):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.use_d       = use_d_feedthrough

        u1 = torch.rand(hidden_size)
        u2 = torch.rand(hidden_size)
        nu_log = -torch.log(-torch.log(u1 * (r_max - r_min) + r_min + 1e-8) + 1e-8)
        th_log = torch.log(max_phase * u2 + 1e-8)
        self.nu    = nn.Parameter(nu_log)
        self.theta = nn.Parameter(th_log)

        self.B_re = nn.Linear(input_size, hidden_size, bias=False)
        self.B_im = nn.Linear(input_size, hidden_size, bias=False)
        self.C    = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        if use_d_feedthrough:
            self.D = nn.Linear(input_size, hidden_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.B_re.weight)
        nn.init.xavier_normal_(self.B_im.weight)
        nn.init.xavier_normal_(self.C.weight)
        if self.use_d:
            nn.init.xavier_normal_(self.D.weight)

    def _lambda_gamma(self):
        log_r     = -torch.exp(self.nu)
        phi       = torch.exp(self.theta)
        lambda_re = torch.exp(log_r) * torch.cos(phi)
        lambda_im = torch.exp(log_r) * torch.sin(phi)
        gamma     = torch.sqrt(torch.clamp(1.0 - torch.exp(2.0 * log_r), min=1e-6))
        return lambda_re, lambda_im, gamma

    def forward(
        self,
        u: torch.Tensor,   # [B, input_size]
        h: torch.Tensor,   # [B, 2*hidden_size]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_re = h[:, :self.hidden_size]
        h_im = h[:, self.hidden_size:]
        lam_re, lam_im, gamma = self._lambda_gamma()

        new_re = lam_re * h_re - lam_im * h_im + gamma * self.B_re(u)
        new_im = lam_re * h_im + lam_im * h_re + gamma * self.B_im(u)
        h_n = torch.cat([new_re, new_im], dim=-1)   # [B, 2H]

        y = self.C(h_n)
        if self.use_d:
            y = y + self.D(u)
        return y, h_n

    def forward_sequence(
        self,
        u: torch.Tensor,   # [T, B, input_size]
        h0: torch.Tensor,  # [B, 2*hidden_size]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T, B, _ = u.shape
        u_flat = u.flatten(0, 1)
        Bu_re = self.B_re(u_flat).view(T, B, -1)
        Bu_im = self.B_im(u_flat).view(T, B, -1)

        lam_re, lam_im, gamma = self._lambda_gamma()

        h_re = h0[:, :self.hidden_size]
        h_im = h0[:, self.hidden_size:]
        hs = []
        for t in range(T):
            new_re = lam_re * h_re - lam_im * h_im + gamma * Bu_re[t]
            new_im = lam_re * h_im + lam_im * h_re + gamma * Bu_im[t]
            h_re, h_im = new_re, new_im
            hs.append(torch.cat([h_re, h_im], dim=-1))

        h_seq = torch.stack(hs, dim=0)                    # [T, B, 2H]
        y_seq = self.C(h_seq.flatten(0, 1)).view(T, B, -1)
        if self.use_d:
            y_seq = y_seq + self.D(u_flat).view(T, B, -1)
        return y_seq, h_seq[-1]


class LRUBlock(nn.Module):
    """RMSNorm → LRUCell → GLU-gate → RMSNorm → PFFN with to residual-connections.

    forward(x [B,in], h [B,2H]) → (y [B,H], h_n [B,2H])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        ff_mult: int = 2,
        r_min: float = 0.0,
        r_max: float = 0.999,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.input_proj = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size else nn.Identity()
        )

        self.norm1     = nn.RMSNorm(hidden_size)
        self.lru       = LRUCell(hidden_size, hidden_size, r_min=r_min, r_max=r_max)
        self.gate_proj = nn.Linear(hidden_size, 2 * hidden_size)

        self.norm2 = nn.RMSNorm(hidden_size)
        ff_dim = hidden_size * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout),
        )
        nn.init.normal_(self.ff[-2].weight, std=0.01 / (hidden_size ** 0.5))
        nn.init.zeros_(self.ff[-2].bias)

    def forward(
        self,
        x: torch.Tensor,   # [B, input_size]
        h: torch.Tensor,   # [B, 2*hidden_size]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)

        residual = x
        lru_out, h_n = self.lru(self.norm1(x), h)
        v, g  = self.gate_proj(lru_out).chunk(2, dim=-1)
        x     = residual + F.silu(v) * torch.sigmoid(g)

        x = x + self.ff(self.norm2(x))
        return x, h_n
