from __future__ import annotations

import torch
import torch.nn.functional as F

from knitwork.models.grnn_fix_v4 import GridRnnFixV4


class GridRnnBalance(GridRnnFixV4):
    # Two independent replacements for v4's `_head_aux`, each weight-gated so a config
    # arm enables exactly one. They factorize the head penalty along two axes:
    #
    #            what is measured        how it is penalized
    #   REG2     logit magnitude        relu(max share - target)
    #   causal   leave-one-out effect   relu(max share - target)     <- changes "what"
    #   lb       logit magnitude        Switch load balance          <- changes "how"
    #
    # Neither adds parameters, so every arm stays at the 388.41k of the 12C study.
    def __init__(
            self, *,
            aux_causal_weight: float = 0.0,
            causal_share_target: float | None = None,
            aux_lb_weight: float = 0.0,
            **kw
    ):
        super().__init__(**kw)
        self.aux_causal_weight = aux_causal_weight
        self.aux_lb_weight = aux_lb_weight
        # same 2/C default as head_share_target: no column above twice the fair share
        self.causal_share_target = (
            2.0 / self.n_columns if causal_share_target is None else causal_share_target
        )
        print(
            f'GridRnnBalance aux_causal={aux_causal_weight}'
            f' target={self.causal_share_target:.3f} aux_lb={aux_lb_weight}'
            f' aux_head={self.aux_head_weight}'
        )

    def _pool(self, o):
        # o: [..., C, B, H] -> head input [..., B, in]
        if self.pooled_head:
            return o.mean(dim=-3)
        lead = o.shape[:-3]
        return o.movedim(-3, -2).reshape(*lead, o.shape[-2], -1)

    def _causal_aux(self, o):
        """Balance the causal profile: how much does dropping each column change the output.

        The offline analysis scores a column by the accuracy drop when it is ablated, and
        that is the metric we actually care about -- but the head penalty optimizes logit
        magnitude, which the 12C study showed is a different thing (REG2 cut magnitude
        monopoly to 0.285 while one column still owned a causal delta of 0.256). This
        measures the real quantity: KL between the full prediction and the prediction with
        column c zeroed, exactly matching `ablate_mode='zero'` in the analysis script.

        Shares rather than raw KL, so the term cannot be satisfied by making every column
        irrelevant -- only by spreading the sensitivity evenly.
        """
        C = self.n_columns
        keep = 1.0 - torch.eye(C, device=o.device, dtype=o.dtype)  # [C_drop, C]
        od = o.unsqueeze(0) * keep[:, :, None, None]               # [C_drop, C, B, H]
        logits_d = self.head(self._pool(od))                       # [C_drop, B, V]
        # reference is the model's own current output: detached, it is a target to stay
        # close to, not something the term is allowed to move
        ref = F.softmax(self.head(self._pool(o)), dim=-1).detach().unsqueeze(0)
        kl = (ref * (ref.clamp_min(1e-8).log() - F.log_softmax(logits_d, dim=-1)))
        kl = kl.sum(dim=-1).mean(dim=-1)                           # [C_drop]
        share = kl / (kl.sum() + 1e-6)
        return F.relu(share.max() - self.causal_share_target)

    def _lb_aux(self, o):
        """Switch-Transformer load balance over columns: C * sum_c f_c * P_c, minus 1.

        `_head_aux` is relu(max share - target), so gradient reaches exactly one column --
        the current leader. It flattens the top and stops as soon as the leader matches
        the runner-up, leaving the tail untouched: tripling its weight moved max share
        only 0.285 -> 0.271. This form touches every column on every step and does not
        saturate. Value is 0 for a uniform split and C-1 for total collapse.
        """
        C = self.n_columns
        mag = self._col_contrib(o).abs().mean(dim=2)               # [C, B]
        p = mag / (mag.sum(dim=0, keepdim=True) + 1e-6)            # per-example routing
        P = p.mean(dim=1)                                          # [C] mean share
        # f_c: fraction of examples where c dominates. Non-differentiable by design
        # (it is the empirical load), gradient flows through P only.
        f = F.one_hot(mag.argmax(dim=0), C).to(o.dtype).mean(dim=0)
        return C * (f * P).sum() - 1.0

    def _readout_aux(self, o):
        t = super()._readout_aux(o)
        if self.aux_causal_weight > 0:
            t = t + self.aux_causal_weight * self._causal_aux(o)
        if self.aux_lb_weight > 0:
            t = t + self.aux_lb_weight * self._lb_aux(o)
        return t
