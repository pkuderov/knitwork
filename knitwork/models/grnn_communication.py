"""Matched routing/readout controls for the MoSAIC communication experiment.

All conditions allocate and execute the same dense projections. ``none`` means
no inter-column exchange, not no input or no within-column feedback.
"""

import math

import torch
from torch.nn import functional as F

from knitwork.common.torch import normalize_entropy
from knitwork.models.grnn_core import GridRnn as ProductionGridRnn
from knitwork.models.grnn_core import StochasticMessagePassingLayer


class ControlledMessagePassingLayer(StochasticMessagePassingLayer):
    def forward(self, q, k, v, return_weights=False):
        if self.routing == 'dynamic':
            return super().forward(q, k, v, return_weights=return_weights)

        Cq, B, D = q.shape
        Ckv = k.shape[0]
        if self.routing == 'static':
            # Identities alone determine routes; messages still carry their contents.
            q = self.ids[0][:Cq].expand(-1, B, -1)
            k = self.ids[1][:Ckv].expand(-1, B, -1)
        else:
            # Keep content-dependent local input/feedback gating in independent columns.
            q = q + self.ids[0][:Cq]
            k = k + self.ids[1][:Ckv]

        W_q, W_k, W_v = self.mha.in_proj_weight.split(D, dim=0)
        b_q, b_k, b_v = self.mha.in_proj_bias.split(D, dim=0)
        q = F.silu(F.linear(q, W_q, b_q))
        k = F.silu(F.linear(k, W_k, b_k))
        v = F.silu(F.linear(v, W_v, b_v))
        q, k, v = map(self.split_heads, (q, k, v))

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.training and self.noise_std > 0.0:
            logits = logits + self.noise_std * torch.randn_like(logits)
        logits = F.softplus(self.pi_logtemp) * logits
        if self.routing == 'none':
            # Bottom: own delayed top state + external input. Above: own lower state.
            logits = logits.masked_fill(~self.local_sources, float('-inf'))
        pi_route = torch.softmax(logits, dim=-1)

        msg = torch.matmul(pi_route, v)
        msg = msg.permute(2, 0, 1, 3).reshape(Cq, B, D)
        msg = F.linear(msg, self.mha.out_proj.weight, self.mha.out_proj.bias)
        if self.ln_msg is not None:
            msg = self.ln_msg(msg)

        info = {}
        if self.training:
            cost = 1.0 - pi_route.diagonal(dim1=-2, dim2=-1)
            if Ckv > Cq:
                extra = torch.arange(Cq, dtype=pi_route.dtype, device=pi_route.device)
                cost = cost + pi_route[..., -1] * (extra * 2 - 1)
            entropy = -(pi_route * torch.log(pi_route.clamp_min(torch.finfo(pi_route.dtype).tiny))).sum(dim=-1)
            info = {
                'comm_loss': cost.mean(),
                'comm_entropy': normalize_entropy(entropy.mean(), Ckv),
            }
        if return_weights:
            info['attn_weights'] = pi_route.detach().mean(dim=(0, 1))
        return msg, info


class GridRnn(ProductionGridRnn):
    def __init__(self, *, routing='dynamic', readout='column0', **kwargs):
        if routing not in ('dynamic', 'static', 'none'):
            raise ValueError('routing must be dynamic, static or none')
        if readout not in ('column0', 'mean'):
            raise ValueError('readout must be column0 or mean')
        if kwargs.get('mha', 2) != 2 or kwargs.get('top_k') is not None:
            raise ValueError('Communication controls require dense mha=2 without top_k')
        if kwargs.get('n_inputs', 1) != 1 or kwargs.get('n_outputs', 1) != 1:
            raise ValueError('Communication controls require one external input/output')
        kwargs['mha'] = 2
        super().__init__(**kwargs)
        self.routing = routing
        self.readout = readout
        # Reuse initialized production modules, including exactly the same RNG draws.
        for layer in self.attn:
            layer.routing = routing
            n_q = self.n_columns
            n_kv = layer.ids.shape[1]
            allowed = torch.eye(n_q, n_kv, dtype=torch.bool)
            if n_kv > n_q:
                allowed[:, n_q:] = True
            layer.register_buffer('local_sources', allowed, persistent=False)

    def _message_passing_class(self, index):
        return ControlledMessagePassingLayer

    def forward(self, x, state, *, capture=False, **kwargs):
        y, state, info = super().forward(x, state, capture=capture, **kwargs)
        if self.readout == 'mean':
            y = state['h'][-1].mean(dim=0)
        return y, state, info
