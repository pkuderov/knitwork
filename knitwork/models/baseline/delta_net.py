"""DeltaNet baseline — delta-rule linear attention (Yang et al., arXiv 2406.06484)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _DeltaLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        H = hidden_size
        self.W_q = nn.Linear(H, H, bias=False)
        self.W_k = nn.Linear(H, H, bias=False)
        self.W_v = nn.Linear(H, H, bias=False)
        self.W_b = nn.Linear(H, 1,  bias=True)   # write-gate
        self.norm = nn.RMSNorm(H)
        ff = H * 2
        self.ff = nn.Sequential(nn.Linear(H, ff), nn.GELU(), nn.Linear(ff, H))
        nn.init.normal_(self.ff[-1].weight, std=0.01 / (H ** 0.5))
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, x: torch.Tensor, S: torch.Tensor):
        # x: [B, H],  S: [B, H, H] — KV matrix state
        k  = F.normalize(self.W_k(x), dim=-1)                       # unit key
        v  = self.W_v(x)
        q  = self.W_q(x)
        b  = torch.sigmoid(self.W_b(x))                              # [B, 1]
        Sk = torch.bmm(S, k.unsqueeze(-1)).squeeze(-1)               # current estimate
        dv = v - Sk                                                   # delta correction
        S  = S + b.unsqueeze(-1) * torch.bmm(                        # outer-product write
            dv.unsqueeze(-1), k.unsqueeze(1)
        )
        y  = torch.bmm(S, q.unsqueeze(-1)).squeeze(-1)               # retrieve
        x  = x + self.norm(y)
        x  = x + self.ff(x)
        return x, S


class DeltaNet(nn.Module):
    """Delta-rule linear attention baseline.

    State per layer: KV matrix S ∈ R^{H×H}.
    Update: S_t = S_{t-1} + beta * (v - S_{t-1}k) ⊗ k  (targeted memory write).
    """

    def __init__(
        self,
        *,
        input_size: int,
        embedding_size: int,
        output_size: int,
        hidden_size: int,
        n_layers: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.embedding  = nn.Embedding(input_size, embedding_size)
        self.input_proj = nn.Linear(embedding_size, hidden_size)
        self.layers     = nn.ModuleList([_DeltaLayer(hidden_size) for _ in range(n_layers)])
        self.norm_out   = nn.RMSNorm(hidden_size)
        self.head       = nn.Linear(hidden_size, output_size)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'DeltaNet {n_layers}L  hidden={hidden_size}  params={n_params:,}')

    def forward(self, tokens: torch.Tensor, h=None):
        # tokens: [B, 1]
        x = self.embedding(tokens.view(-1))  # [B, emb]
        x = self.input_proj(x)               # [B, H]
        B = x.shape[0]

        if h is None:
            h = self.init_state(B, x.device)
        S_list, y_prev = h

        new_S, new_y = [], []
        for layer, S in zip(self.layers, S_list):
            x, S_new = layer(x, S)
            new_S.append(S_new)
            new_y.append(x)

        y_stk  = torch.stack(new_y)  # [L, B, H]
        logits = self.head(self.norm_out(x))
        return logits, (new_S, y_stk)

    def init_state(self, batch_size: int, device):
        H = self.hidden_size
        S = [torch.zeros(batch_size, H, H, device=device) for _ in self.layers]
        y = torch.zeros(self.n_layers, batch_size, H, device=device)
        return (S, y)

    def reset_state(self, h, mask: torch.Tensor):
        if h is None:
            return h
        S_list, y = h
        m = (~mask.bool()).float()
        return (
            [S * m[:, None, None] for S in S_list],
            y * m[None, :, None],
        )

    def detach_state(self, h):
        if h is None:
            return h
        S_list, y = h
        return ([S.detach() for S in S_list], y.detach())

    def get_top_h(self, h) -> torch.Tensor:
        """Last-layer output for critic: [B, H]."""
        _, y = h
        return y[-1]


class DeltaNetCore(nn.Module):
    """Feature-level DeltaNet core for use with model wrappers."""
    has_attn = False

    def __init__(
            self, *,
            hidden_size, n_layers,
            dtype, device,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dtype = dtype
        self.device = device

        self.layers = nn.ModuleList([
            _DeltaLayer(hidden_size)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.RMSNorm(hidden_size)

        print(f'DeltaNet core {n_layers}L w/ {hidden_size} hidden units')

    def forward(self, x: torch.Tensor, state: dict, **_):
        assert x.shape[0] == 1
        x = x.squeeze(0)
        if state is None:
            state = self.init_state(x.shape[0])

        new_S = []
        for layer, S in zip(self.layers, state['S']):
            x, S = layer(x, S)
            new_S.append(S)

        return self.norm_out(x), {'S': new_S}, {}

    def reset_state(self, state=None, reset_mask=None, *, bsz=None):
        if state is None:
            bsz = reset_mask.shape[0] if reset_mask is not None else bsz
            return self.init_state(bsz)

        keep = (~reset_mask.flatten())[:, None, None]
        return {'S': [S * keep for S in state['S']]}

    def detach_state(self, state):
        if state is None:
            return state
        return {'S': [S.detach() for S in state['S']]}

    def init_state(self, bsz):
        S = [
            torch.zeros(
                bsz, self.hidden_size, self.hidden_size,
                device=self.device, dtype=self.dtype,
            )
            for _ in self.layers
        ]
        return {'S': S}
