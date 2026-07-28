from __future__ import annotations

import torch
import torch.nn.functional as F

from knitwork.models.grnn_fix_v4 import GridRnnFixV4, PerColumnAttention


class RoutedAttention(PerColumnAttention):
    # Two structural knobs on the column->column routing, both off by default.
    # Self-attention is already forbidden by the base class (the diagonal is masked to
    # -inf), so the degenerate mode here is not "every column reads itself" but "every
    # column reads the same hub" -- which is what these attack.
    def __init__(self, *a, top_k: int = 0, noise_std: float = 0.0, **kw):
        super().__init__(*a, **kw)
        self.top_k = top_k
        self.noise_std = noise_std

    def _shape_logits(self, logits):
        # noise scaled by the logits' own spread. Absolute-variance noise is defeated by
        # the learnable beta: growing beta makes fixed noise negligible, and the task
        # gradient actively pushes that way. A relative perturbation is scale-invariant,
        # so there is no escape by rescaling. std is detached: it is a measuring stick.
        if self.training and self.noise_std > 0:
            scale = logits.detach().std().clamp_min(1e-6)
            logits = logits + self.noise_std * scale * torch.randn_like(logits)
        # hard capacity: a query column may read at most top_k others. A cap cannot be
        # traded off against the task loss the way a penalty can -- five arms of soft
        # penalties were all partly gamed.
        if self.top_k and 0 < self.top_k < logits.shape[-1]:
            kth = logits.topk(self.top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float('-inf'))
        return logits

    def forward(self, h, return_weights: bool = False, col_mask=None):
        # duplicate of the parent body up to the softmax, with _shape_logits inserted;
        # the parent computes softmax inline so there is no cheaper hook
        C, B, D = h.shape
        q = self.W_q(h + self.ids_q).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        k = self.W_k(h + self.ids_k).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        v = self.W_v(h).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)

        beta = self.beta_floor + self.log_beta.exp().T.unsqueeze(1).unsqueeze(-1)
        logits = beta * torch.matmul(q, k.transpose(-2, -1))     # [heads, B, C_q, C_k]
        eye = torch.eye(C, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(eye[None, None], float('-inf'))
        if col_mask is not None:
            logits = logits.masked_fill(~col_mask.to(logits.device)[None, None], float('-inf'))
        logits = self._shape_logits(logits)
        attn = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)

        with torch.no_grad():
            self.last_attn = attn.detach().mean(dim=(0, 1))
        self._attn = attn if self.stash_attn else None
        out = torch.matmul(attn, v)
        out = out.permute(2, 1, 0, 3).contiguous().view(C, B, D)
        return self.out_proj(out), (attn.mean(dim=(0, 1)) if return_weights else None)


class GridRnnRoute(GridRnnFixV4):
    # v4 with the communication graph as a first-class object: balance who RECEIVES
    # attention, cap how many sources a column may read, and perturb the routing so an
    # early accidental winner cannot lock in. Motivated by the 12C study, where the causal
    # mass collapsed onto two columns and no metric ever looked at the routing.
    def __init__(
            self, *,
            aux_route_weight: float = 0.0,
            top_k: int = 0,
            noise_std: float = 0.0,
            **kw
    ):
        self._route_kw = dict(top_k=top_k, noise_std=noise_std)
        super().__init__(**kw)
        self.needs_attn_graph = aux_route_weight > 0
        self.aux_route_weight = aux_route_weight
        print(
            f'GridRnnRoute aux_route={aux_route_weight} top_k={top_k}'
            f' noise_std={noise_std}'
        )

    def _make_attention(self, H, **kw):
        return RoutedAttention(H, **kw, **self._route_kw)

    def _route_aux(self):
        """Switch-style load balance on in-degree: no column may become the hub.

        Same non-saturating form as GridRnnBalance._lb_aux, applied to the routing matrix
        instead of the readout: 0 for a uniform split, C-1 for total collapse.
        """
        C, total = self.n_columns, 0.0
        n = 0
        for a in self.attn:
            attn = a._attn
            a._attn = None                       # drop the graph reference immediately
            if attn is None:
                continue
            P = attn.mean(dim=(0, 1, 2))                              # [C_k] mean received
            f = F.one_hot(attn.argmax(dim=-1), C).to(attn.dtype).mean(dim=(0, 1, 2))
            total = total + C * (f.detach() * P).sum() - 1.0
            n += 1
        return total / max(n, 1)

    def _readout_aux(self, o):
        t = super()._readout_aux(o)
        if self.aux_route_weight > 0:
            t = t + self.aux_route_weight * self._route_aux()
        return t
