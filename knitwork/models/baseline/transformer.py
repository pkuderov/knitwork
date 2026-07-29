"""Segment-recurrent Transformer baselines for character-level language modeling.

The legacy Transformer preserves its original one-token implementation. The
wrapper-compatible TransformerCore adds RoPE positions, valid-cache masking,
and batched rollout processing.
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


class _TransformerCoreLayer(_TransformerLayer):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__(d_model, n_heads, d_ff, dropout)
        assert self.d_head % 2 == 0
        self.dropout_p = dropout
        inv_freq = 1.0 / (10_000 ** (
            torch.arange(0, self.d_head, 2) / self.d_head
        ))
        self.register_buffer('rope_inv_freq', inv_freq, persistent=False)

    def _apply_rope(self, x, positions):
        # x: [B, heads, time, head_dim], positions: [B, time]
        angles = positions.to(x.dtype)[..., None] * self.rope_inv_freq
        cos = angles.cos()[:, None, :, :, None]
        sin = angles.sin()[:, None, :, :, None]

        x = x.view(*x.shape[:-1], self.d_head // 2, 2)
        x_re, x_im = x.unbind(dim=-1)
        x = torch.stack([
            x_re * cos[..., 0] - x_im * sin[..., 0],
            x_re * sin[..., 0] + x_im * cos[..., 0],
        ], dim=-1)
        return x.flatten(start_dim=-2)

    def forward(
            self, x, k_cache, v_cache, cache_valid,
            positions, reset_mask,
    ):
        # x: [B, time, H], k_cache/v_cache: [B, mem_len, H]
        B, T, H = x.shape

        y = self.norm1(x)
        q = self.q_proj(y).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(y).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(y).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = self._apply_rope(q, positions)
        k = self._apply_rope(k, positions)

        current_k = k.transpose(1, 2).reshape(B, T, H)
        current_v = v.transpose(1, 2).reshape(B, T, H)
        M = k_cache.shape[1]
        k_all = torch.cat([k_cache, current_k], dim=1)
        v_all = torch.cat([v_cache, current_v], dim=1)

        K = k_all.view(B, M + T, self.n_heads, self.d_head).transpose(1, 2)
        V = v_all.view(B, M + T, self.n_heads, self.d_head).transpose(1, 2)

        no_reset_prefix = reset_mask.cumsum(dim=1) == 0
        time = torch.arange(T, device=x.device)[None]
        cache_ix = torch.arange(M, device=x.device)[None, None]
        cache_mask = (
            cache_valid[:, None, :]
            & no_reset_prefix[:, :, None]
            & (cache_ix >= time[:, :, None] + 1)
        )
        segment = reset_mask.cumsum(dim=1)
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        last_reset = torch.where(reset_mask, time, -1).cummax(dim=1).values
        window_start = torch.maximum(last_reset, time - M + 1).clamp_min(0)
        sequence_ix = torch.arange(T, device=x.device)[None, None]
        sequence_mask = (
            (segment[:, :, None] == segment[:, None, :])
            & causal
            & (sequence_ix >= window_start[:, :, None])
        )
        attn_mask = torch.cat([cache_mask, sequence_mask], dim=-1)[:, None]
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, K, V,
            attn_mask=attn_mask, dropout_p=dropout_p,
        )
        out = out.squeeze(2).transpose(1, 2).reshape(B, T, H)
        x = x + self.out_proj(out)

        x = x + self.ff2(F.silu(self.ff1(self.norm2(x))))

        has_reset = reset_mask.any(dim=1, keepdim=True)
        cache_valid = cache_valid & ~has_reset
        last_reset = torch.where(reset_mask, time, -1).max(dim=1).values
        current_valid = time >= last_reset[:, None]
        valid = torch.cat([cache_valid, current_valid], dim=1)[:, -M:]
        return x, k_all[:, -M:], v_all[:, -M:], valid


class TransformerCore(nn.Module):
    """Feature-level Transformer with RoPE and valid rolling K/V memory."""
    has_attn = False

    def __init__(
            self, *,
            hidden_size, n_layers,
            n_heads=4, d_ff=None, mem_len=256, dropout=0.0,
            dtype, device,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.mem_len = mem_len
        self.dtype = dtype
        self.device = device

        d_ff = d_ff or hidden_size * 4
        self.layers = nn.ModuleList([
            _TransformerCoreLayer(hidden_size, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.RMSNorm(hidden_size)

        print(
            f'Transformer core {n_layers}L w/ {hidden_size} hidden units,'
            f' {n_heads} heads, FF={d_ff}, mem={mem_len}'
        )

    def forward(self, x: torch.Tensor, state: dict, *, reset_mask=None, **_):
        # x: [time, batch, hidden_size]
        T, B = x.shape[:2]
        if state is None:
            state = self.init_state(B)
        if reset_mask is None:
            reset_mask = torch.zeros(T, B, dtype=torch.bool, device=x.device)

        reset_mask = reset_mask.transpose(0, 1).bool()
        time = torch.arange(T, device=x.device)[None]
        last_reset = torch.where(reset_mask, time, -1).cummax(dim=1).values
        positions = torch.where(
            last_reset >= 0,
            time - last_reset,
            state['pos'][:, None] + time,
        )
        x = x.transpose(0, 1)

        new_kv = []
        for layer, (k_cache, v_cache) in zip(self.layers, state['kv']):
            x, k_cache, v_cache, valid = layer(
                x, k_cache, v_cache, state['valid'],
                positions, reset_mask,
            )
            new_kv.append((k_cache, v_cache))

        state = {
            'kv': new_kv,
            'valid': valid,
            'pos': positions[:, -1] + 1,
        }
        y = self.norm_out(x).transpose(0, 1)
        if T == 1:
            y = y.squeeze(0)
        return y, state, {}

    def reset_state(self, state=None, reset_mask=None, *, bsz=None):
        if state is None:
            bsz = reset_mask.shape[0] if reset_mask is not None else bsz
            return self.init_state(bsz)

        keep = (~reset_mask.flatten())[:, None, None]
        kv = [
            (k_cache * keep, v_cache * keep)
            for k_cache, v_cache in state['kv']
        ]
        keep = keep[:, 0, 0]
        valid = state['valid'] & keep[:, None]
        pos = state['pos'] * keep
        return {'kv': kv, 'valid': valid, 'pos': pos}

    def detach_state(self, state):
        if state is None:
            return state
        kv = [
            (k_cache.detach(), v_cache.detach())
            for k_cache, v_cache in state['kv']
        ]
        return {
            'kv': kv,
            'valid': state['valid'].detach(),
            'pos': state['pos'].detach(),
        }

    def init_state(self, bsz):
        kv = [
            (
                torch.zeros(
                    bsz, self.mem_len, self.hidden_size,
                    device=self.device, dtype=self.dtype,
                ),
                torch.zeros(
                    bsz, self.mem_len, self.hidden_size,
                    device=self.device, dtype=self.dtype,
                ),
            )
            for _ in self.layers
        ]
        valid = torch.zeros(
            bsz, self.mem_len,
            device=self.device, dtype=torch.bool,
        )
        pos = torch.zeros(bsz, device=self.device, dtype=torch.long)
        return {'kv': kv, 'valid': valid, 'pos': pos}
