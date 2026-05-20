from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from typing import NamedTuple


class EngramState(NamedTuple):
    h: torch.Tensor   # [layers, cols, batch, hidden]
    M: list           # list of per-layer/col Tensors [batch, n_slots, hidden]


class EngramMemory(nn.Module):
    """Associative memory with Hebbian write and sparse top-K cosine read."""

    def __init__(
        self,
        hidden_size: int,
        n_slots: int = 16,
        top_k: int = 4,
        hebb_lr: float = 0.1,
        gate_write: bool = True,
        dtype=torch.float64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_slots = n_slots
        self.top_k = min(top_k, n_slots)
        self.hebb_lr = hebb_lr
        self.dtype = dtype

        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype)
        nn.init.orthogonal_(self.W_q.weight)
        nn.init.orthogonal_(self.W_r.weight)
        nn.init.zeros_(self.W_r.bias)

        self.write_gate = None
        if gate_write:
            self.write_gate = nn.Sequential(
                nn.Linear(hidden_size * 2, 1, dtype=dtype),
                nn.Sigmoid(),
            )

        self.read_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size, dtype=dtype),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_size, dtype=dtype)

    def init_memory(self, bsz: int, device, dtype) -> torch.Tensor:
        return torch.zeros(bsz, self.n_slots, self.hidden_size, device=device, dtype=dtype)

    def _sparse_attention(self, query: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        # cosine similarity [B, S]
        q_norm = F.normalize(query.unsqueeze(1), dim=-1)  # [B, 1, H]
        scores = (q_norm * F.normalize(M, dim=-1)).sum(dim=-1)

        if self.top_k < self.n_slots:
            threshold = scores.topk(self.top_k, dim=-1).values[:, -1:]
            scores = scores.masked_fill(scores < threshold, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(torch.isnan(attn), torch.full_like(attn, 1.0 / self.n_slots), attn)
        return attn

    def read(self, h: torch.Tensor, M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn = self._sparse_attention(self.W_q(h), M)
        r = self.norm(self.W_r(torch.bmm(attn.unsqueeze(1), M).squeeze(1)))
        g = self.read_gate(torch.cat([h, r], dim=-1))
        return g * r, attn

    @torch.no_grad()
    def write(self, h: torch.Tensor, M: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        if self.write_gate is not None:
            w = self.write_gate(torch.cat([h, M.mean(dim=1)], dim=-1))  # [B, 1]
        else:
            w = torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype)

        delta = h.unsqueeze(1) - M                    # [B, S, H]
        lr = self.hebb_lr * w.unsqueeze(-1)           # [B, 1, 1]
        M_new = M + lr * attn.unsqueeze(-1) * delta
        return M_new / M_new.norm(dim=-1, keepdim=True).clamp(min=1.0)

    def forward(self, h: torch.Tensor, M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read then write; return (retrieval, new_M, attn_weights)."""
        r, attn = self.read(h, M)
        return r, self.write(h, M, attn), attn
