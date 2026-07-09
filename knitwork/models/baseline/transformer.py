"""Segment-recurrent Transformer baseline for text8 / character-level LM.

Each layer caches past K/V vectors in a rolling buffer (mem_len tokens).
At each step only one token is processed; the cache acts as the memory.
No explicit PE — relative ordering is implicit in the causal cache layout.

State per layer: (k [B, mem_len, H], v [B, mem_len, H]).
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn


class _TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff,   d_model, bias=False)
        nn.init.zeros_(self.ff2.weight)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.scale = math.sqrt(self.d_head)

    def forward(self, x: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        # x: [B, H]  k_cache/v_cache: [B, mem_len, H]
        B, H = x.shape

        # self-attention: Q from current token, K/V from full cache
        y = self.norm1(x)
        q = self.q_proj(y).view(B, self.n_heads, self.d_head)          # [B, nh, dh]
        k = self.k_proj(y).view(B, self.n_heads, self.d_head)          # [B, nh, dh]
        v = self.v_proj(y).view(B, self.n_heads, self.d_head)          # [B, nh, dh]

        # roll cache: drop oldest, append current
        k_cache_new = torch.cat([k_cache[:, 1:], k_cache.new_zeros(B, 1, H)], dim=1)
        v_cache_new = torch.cat([v_cache[:, 1:], v_cache.new_zeros(B, 1, H)], dim=1)
        # write current k,v into last slot
        k_cache_new[:, -1] = k.reshape(B, H)
        v_cache_new[:, -1] = v.reshape(B, H)

        # attention: Q [B, nh, 1, dh] × K^T [B, nh, dh, mem] → [B, nh, 1, mem]
        K = k_cache_new.view(B, -1, self.n_heads, self.d_head).permute(0, 2, 3, 1)  # [B, nh, dh, mem]
        V = k_cache_new.view(B, -1, self.n_heads, self.d_head)                       # for shape; use v below
        V = v_cache_new.view(B, -1, self.n_heads, self.d_head).permute(0, 2, 1, 3)  # [B, nh, mem, dh]

        attn = (q.unsqueeze(2) @ K) / self.scale        # [B, nh, 1, mem]
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out  = (attn @ V).squeeze(2)                     # [B, nh, dh]
        out  = out.reshape(B, H)
        x    = x + self.out_proj(out)

        # FFN
        x = x + self.ff2(F.silu(self.ff1(self.norm2(x))))
        return x, k_cache_new, v_cache_new


class Transformer(nn.Module):
    """Rolling-cache Transformer for step-by-step character-level LM.

    State per layer: (k_cache [B, mem_len, H], v_cache [B, mem_len, H]).
    """

    def __init__(
        self,
        *,
        input_size: int,
        embedding_size: int,
        output_size: int,
        hidden_size: int,
        n_layers: int,
        n_heads: int = 4,
        d_ff: int | None = None,
        mem_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.mem_len     = mem_len

        d_ff = d_ff or hidden_size * 4

        self.embedding  = nn.Embedding(input_size, embedding_size)
        self.input_proj = nn.Linear(embedding_size, hidden_size, bias=False) if embedding_size != hidden_size else nn.Identity()
        self.layers = nn.ModuleList([
            _TransformerLayer(hidden_size, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, output_size, bias=False)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Transformer {n_layers}L  H={hidden_size}  heads={n_heads}  ff={d_ff}  mem={mem_len}  params={n_params:,}')

    def init_state(self, batch_size: int, device, dtype=torch.float32):
        H, M = self.hidden_size, self.mem_len
        return [
            (
                torch.zeros(batch_size, M, H, device=device, dtype=dtype),  # k_cache
                torch.zeros(batch_size, M, H, device=device, dtype=dtype),  # v_cache
            )
            for _ in self.layers
        ]

    def forward(self, tokens: torch.Tensor, h=None, return_attn: bool = False):
        # tokens: [B, 1]
        B = tokens.shape[0]
        x = self.embedding(tokens.view(-1))  # [B, emb]
        x = self.input_proj(x)               # [B, H]

        if h is None:
            h = self.init_state(B, x.device, x.dtype)

        new_h = []
        for layer, (k_cache, v_cache) in zip(self.layers, h):
            x, k_new, v_new = layer(x, k_cache, v_cache)
            new_h.append((k_new, v_new))

        logits = self.head(self.norm_out(x))   # [B, vocab]
        if return_attn:
            return logits, new_h, {}
        return logits, new_h

    def reset_state(self, h, mask: torch.Tensor):
        if h is None:
            return h
        m = (~mask.bool()).float()[:, None, None]  # [B, 1, 1]
        return [(k * m, v * m) for k, v in h]

    def detach_state(self, h):
        if h is None:
            return h
        return [(k.detach(), v.detach()) for k, v in h]

    def get_top_h(self, h) -> torch.Tensor:
        """Last-layer last cached key as a summary vector [B, H]."""
        k, _ = h[-1]
        return k[:, -1]
