"""HGRN2 baseline — outer-product state expansion (Qin et al., COLM 2024, arXiv 2404.07904)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _HGRN2Layer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, expand: int):
        super().__init__()
        H, D = hidden_size, hidden_size * expand
        self.W_f  = nn.Linear(input_size, H, bias=True)
        self.W_i  = nn.Linear(input_size, H, bias=True)
        self.W_g  = nn.Linear(input_size, D, bias=True)
        self.W_o  = nn.Linear(D, H, bias=False)
        self.norm = nn.RMSNorm(H)
        self.ff   = nn.Sequential(nn.Linear(H, H * 2), nn.GELU(), nn.Linear(H * 2, H))
        nn.init.normal_(self.ff[-1].weight, std=0.01 / (H ** 0.5))
        nn.init.zeros_(self.ff[-1].bias)
        nn.init.zeros_(self.W_f.bias)
        nn.init.zeros_(self.W_i.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        # x: [B, input_size],  h: [B, H, D]
        f   = torch.sigmoid(self.W_f(x))           # [B, H] forget gate
        i   = F.silu(self.W_i(x)) * (1.0 - f)     # [B, H] input gate, coupled to f
        g   = F.silu(self.W_g(x))                   # [B, D] content
        # outer-product state expansion: h_t[row] = f[row]*h_{t-1}[row] + i[row]*g
        h   = f.unsqueeze(-1) * h + i.unsqueeze(-1) * g.unsqueeze(1)  # [B, H, D]
        y   = self.norm(self.W_o(h.sum(dim=1)))      # sum rows → [B, D] → [B, H]
        y   = y + self.ff(y)
        return y, h


class HGRN2(nn.Module):
    """HGRN2 baseline.

    State per layer: matrix h ∈ R^{H×D} where D = hidden_size * expand.
    Update: h_t = diag(f_t) h_{t-1} + i_t ⊗ g_t  (outer-product write).
    """

    def __init__(
        self,
        *,
        input_size: int,
        embedding_size: int,
        output_size: int,
        hidden_size: int,
        n_layers: int,
        expand: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.expand      = expand

        self.embedding  = nn.Embedding(input_size, embedding_size)
        self.input_proj = nn.Linear(embedding_size, hidden_size)
        self.layers     = nn.ModuleList([
            _HGRN2Layer(hidden_size, hidden_size, expand) for _ in range(n_layers)
        ])
        self.norm_out   = nn.RMSNorm(hidden_size)
        self.head       = nn.Linear(hidden_size, output_size)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'HGRN2 {n_layers}L  H={hidden_size}  expand={expand}  params={n_params:,}')

    def forward(self, tokens: torch.Tensor, h=None):
        # tokens: [B, 1]
        x = self.embedding(tokens.view(-1))  # [B, emb]
        x = self.input_proj(x)               # [B, H]
        B = x.shape[0]

        if h is None:
            h = self.init_state(B, x.device)
        mats, y_prev = h

        new_mats, new_y = [], []
        for layer, hi in zip(self.layers, mats):
            x, hi_new = layer(x, hi)
            new_mats.append(hi_new)
            new_y.append(x)

        y_stk  = torch.stack(new_y)  # [L, B, H]
        logits = self.head(self.norm_out(x))
        return logits, (new_mats, y_stk)

    def init_state(self, batch_size: int, device):
        H, D = self.hidden_size, self.hidden_size * self.expand
        mats = [torch.zeros(batch_size, H, D, device=device) for _ in self.layers]
        y    = torch.zeros(self.n_layers, batch_size, H, device=device)
        return (mats, y)

    def reset_state(self, h, mask: torch.Tensor):
        if h is None:
            return h
        mats, y = h
        m = (~mask.bool()).float()
        return (
            [hi * m[:, None, None] for hi in mats],
            y * m[None, :, None],
        )

    def detach_state(self, h):
        if h is None:
            return h
        mats, y = h
        return ([hi.detach() for hi in mats], y.detach())

    def get_top_h(self, h) -> torch.Tensor:
        """Last-layer output for critic: [B, H]."""
        _, y = h
        return y[-1]


class HGRN2Core(nn.Module):
    """Feature-level HGRN2 core for use with model wrappers."""
    has_attn = False

    def __init__(
            self, *,
            hidden_size, n_layers, expand=4,
            dtype, device,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.expand = expand
        self.dtype = dtype
        self.device = device

        self.layers = nn.ModuleList([
            _HGRN2Layer(hidden_size, hidden_size, expand)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.RMSNorm(hidden_size)

        print(
            f'HGRN2 core {n_layers}L w/ {hidden_size} hidden units'
            f' and {expand}x expansion'
        )

    def forward(self, x: torch.Tensor, state: dict, **_):
        assert x.shape[0] == 1
        x = x.squeeze(0)
        if state is None:
            state = self.init_state(x.shape[0])

        new_h = []
        for layer, h in zip(self.layers, state['h']):
            x, h = layer(x, h)
            new_h.append(h)

        return self.norm_out(x), {'h': new_h}, {}

    def reset_state(self, state=None, reset_mask=None, *, bsz=None):
        if state is None:
            bsz = reset_mask.shape[0] if reset_mask is not None else bsz
            return self.init_state(bsz)

        keep = (~reset_mask.flatten())[:, None, None]
        return {'h': [h * keep for h in state['h']]}

    def detach_state(self, state):
        if state is None:
            return state
        return {'h': [h.detach() for h in state['h']]}

    def init_state(self, bsz):
        expanded_size = self.hidden_size * self.expand
        h = [
            torch.zeros(
                bsz, self.hidden_size, expanded_size,
                device=self.device, dtype=self.dtype,
            )
            for _ in self.layers
        ]
        return {'h': h}
