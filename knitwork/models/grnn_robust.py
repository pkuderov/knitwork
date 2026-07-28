from __future__ import annotations

import torch

from knitwork.models.grnn_fix_v4 import GridRnnFixV4


class GridRnnRobust(GridRnnFixV4):
    # v4 with column robustness as the explicit objective. The 12C study
    # (docs/reviews/column_collapse_12c.md) measured four arms and found:
    #   - activity decorrelation alone decorrelates better than the Barlow term
    #     (CKA mean 0.325 vs 0.517), so aux_div defaults to 0 here
    #   - low pairwise CKA is the wrong target: the arm with the best CKA was the
    #     most brittle (four columns with causal delta > 0.3). The right target is a
    #     low *maximum* causal contribution, so we regularize that directly
    # Column dropout is the training-time analogue of the causal ablation the offline
    # analysis performs: if any single column can be dropped during training, none can
    # become a single point of failure at eval.
    def __init__(
            self, *,
            col_dropout: float = 0.1,
            col_dropout_ramp_steps: int = 0,
            col_dropout_rescale: bool = True,
            **kw
    ):
        # defaults that differ from v4; still overridable from config
        kw.setdefault('aux_div_weight', 0.0)
        kw.setdefault('aux_head_weight', 0.15)
        super().__init__(**kw)
        self.col_dropout = col_dropout
        self.col_dropout_ramp_steps = max(int(col_dropout_ramp_steps), 0)
        self.col_dropout_rescale = col_dropout_rescale
        print(
            f'GridRnnRobust col_dropout={col_dropout}'
            f' ramp={self.col_dropout_ramp_steps} aux_div={self.aux_div_weight}'
            f' aux_head={self.aux_head_weight}'
        )

    def _col_dropout_p(self) -> float:
        if self.col_dropout_ramp_steps <= 0:
            return self.col_dropout
        step = self._aux_env_step()
        return self.col_dropout * min(1.0, step / self.col_dropout_ramp_steps)

    def _mask_columns(self, o):
        o = super()._mask_columns(o)                              # inference ablation
        if not self.training or self.col_dropout <= 0 or self.col_keep_mask is not None:
            return o
        p = self._col_dropout_p()
        if p <= 0:
            return o
        # one mask per batch element so columns are dropped independently across envs
        C, B = o.shape[0], o.shape[1]
        keep = (torch.rand(C, B, 1, device=o.device) >= p).to(o.dtype)  # [C, B, 1]
        # never drop every column: fall back to the full state for those envs
        dead = keep.sum(dim=0, keepdim=True) == 0                 # [1, B, 1]
        keep = torch.where(dead, torch.ones_like(keep), keep)
        if self.col_dropout_rescale:
            keep = keep * (C / keep.sum(dim=0, keepdim=True).clamp(min=1.0))
        return o * keep
