"""Mamba baseline — selective SSM (Gu & Dao, arXiv 2312.00752).

Recurrent form: one token at a time, matching the project's step-by-step interface.
State per layer: h ∈ R^{d_state × d_inner}.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn


class _MambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        D = d_model * expand  # d_inner
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = D
        self.d_conv  = d_conv

        # input projection → x and z branches
        self.in_proj  = nn.Linear(d_model, D * 2, bias=False)

        # conv buffer is handled as part of state; weights here
        self.conv_weight = nn.Parameter(torch.empty(D, 1, d_conv))
        self.conv_bias   = nn.Parameter(torch.zeros(D))
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))

        # SSM projections (input-dependent B, C, Δ)
        self.x_proj = nn.Linear(D, d_state + d_state + 1, bias=False)  # Δ,B,C
        self.dt_proj = nn.Linear(1, D, bias=True)
        nn.init.uniform_(self.dt_proj.bias, -4, -1)  # init Δ small

        # A: fixed log-space eigenvalues (learnable)
        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(D, -1)
        self.log_A = nn.Parameter(torch.log(A))  # [D, N]

        # D: skip connection
        self.D = nn.Parameter(torch.ones(D))

        self.out_proj = nn.Linear(D, d_model, bias=False)
        self.norm     = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        # x: [B, d_model]
        # state: (h [B, D, N], conv_buf [B, D, d_conv-1])
        h, conv_buf = state
        B = x.shape[0]
        D, N = self.d_inner, self.d_state

        xz = self.in_proj(x)             # [B, 2D]
        xi, z = xz.split(D, dim=-1)     # [B, D] each

        # causal conv step: shift buffer, apply conv
        xi_unsq = xi.unsqueeze(-1)       # [B, D, 1]
        buf_new  = torch.cat([conv_buf, xi_unsq], dim=-1)   # [B, D, d_conv]
        # conv: sum over d_conv window
        xi_conv = (buf_new * self.conv_weight.squeeze(1)).sum(-1) + self.conv_bias  # [B, D]
        xi_conv = F.silu(xi_conv)
        conv_buf_new = buf_new[:, :, 1:]  # drop oldest

        # SSM
        bcdt = self.x_proj(xi_conv)             # [B, N+N+1]
        B_vec = bcdt[:, :N]                      # [B, N]
        C_vec = bcdt[:, N:2*N]                   # [B, N]
        dt    = F.softplus(self.dt_proj(bcdt[:, 2:3]))  # [B, D]

        # discretize: A_bar = exp(dt * A),  B_bar = dt * B
        A = -torch.exp(self.log_A)               # [D, N], negative
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # [B, D, N]
        dB = dt.unsqueeze(-1) * B_vec.unsqueeze(1)          # [B, D, N]

        h_new = dA * h + dB * xi_conv.unsqueeze(-1)  # [B, D, N]
        y_ssm = (h_new * C_vec.unsqueeze(1)).sum(-1)  # [B, D]
        y_ssm = y_ssm + self.D * xi_conv

        out = y_ssm * F.silu(z)
        residual = x + self.norm(self.out_proj(out))
        return residual, (h_new, conv_buf_new)


class Mamba(nn.Module):
    """Mamba sequential baseline for character-level LM.

    State per layer: (h [B, d_inner, d_state], conv_buf [B, d_inner, d_conv-1]).
    """

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        embedding_size: int,
        hidden_size: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.d_state     = d_state
        self.d_conv      = d_conv
        self.d_inner     = hidden_size * expand

        self.embedding  = nn.Embedding(input_size, embedding_size)
        self.input_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layers     = nn.ModuleList([
            _MambaLayer(hidden_size, d_state, d_conv, expand) for _ in range(n_layers)
        ])
        self.norm_out = nn.RMSNorm(hidden_size)
        self.head     = nn.Linear(hidden_size, output_size, bias=False)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Mamba {n_layers}L  H={hidden_size}  d_state={d_state}  expand={expand}  params={n_params:,}')

    def init_state(self, batch_size: int, device, dtype=torch.float32):
        D, N = self.d_inner, self.d_state
        return [
            (
                torch.zeros(batch_size, D, N, device=device, dtype=dtype),
                torch.zeros(batch_size, D, self.d_conv - 1, device=device, dtype=dtype),
            )
            for _ in self.layers
        ]

    def forward(self, tokens: torch.Tensor, h=None):
        # tokens: [B, 1]
        x = self.embedding(tokens.view(-1))  # [B, emb]
        x = self.input_proj(x)               # [B, H]
        B = x.shape[0]

        if h is None:
            h = self.init_state(B, x.device, x.dtype)

        new_h = []
        for layer, layer_state in zip(self.layers, h):
            x, layer_state_new = layer(x, layer_state)
            new_h.append(layer_state_new)

        logits = self.head(self.norm_out(x))
        return logits, new_h

    def reset_state(self, h, mask: torch.Tensor):
        if h is None:
            return h
        m = (~mask.bool()).float()
        return [
            (
                ssm_h * m[:, None, None],
                conv_buf * m[:, None, None],
            )
            for ssm_h, conv_buf in h
        ]

    def detach_state(self, h):
        if h is None:
            return h
        return [(ssm_h.detach(), conv_buf.detach()) for ssm_h, conv_buf in h]

    def get_top_h(self, h) -> torch.Tensor:
        """Last-layer SSM hidden for critic: [B, H]."""
        ssm_h, _ = h[-1]
        return ssm_h.sum(-1)[:, :self.hidden_size]
