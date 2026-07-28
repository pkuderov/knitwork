from __future__ import annotations

import torch
import torch.nn as nn

from knitwork.models.grnn_fix_v4 import GridRnnFixV4


class GridRnnPlastic(GridRnnFixV4):
    # ReDo-style revival of dead columns.
    #
    # Every aux term we have shapes the structure through the loss, and the 12C study
    # showed that structure is settled by ~0.6M steps -- 1% of training -- and then flat
    # for the remaining 60M (col_cka/frac_gt_06 reaches its final level by 0.6M in every
    # run that logged it). A loss weight applied after that cannot undo an early
    # commitment. Re-initialization can: it is the only mechanism here that moves a
    # column out of a bad basin instead of trying to argue it out.
    #
    # The columns that die are not random. Causal delta correlates negatively with column
    # index in all five 12C arms (rho -0.22..-0.84), and the fast half carries 76% of the
    # mass: the linear update-gate stagger makes the slow half useless on the short
    # episodes early in the curriculum, so they never earn gradient. Revival gives them a
    # second draw once the rest of the network is trained and more niches exist.
    def __init__(
            self, *,
            redo_every: int = 0,
            redo_threshold: float = 0.02,
            redo_max_cols: int = 1,
            **kw
    ):
        super().__init__(**kw)
        self.redo_every = max(int(redo_every), 0)
        self.redo_threshold = redo_threshold
        self.redo_max_cols = redo_max_cols
        self._redo_last = 0.0
        self._last_share = None
        self.redo_count = 0
        print(
            f'GridRnnPlastic redo_every={self.redo_every}'
            f' redo_thresh={redo_threshold} redo_max={redo_max_cols}'
        )

    def _readout_aux(self, o):
        if self.redo_every > 0 and self.training:
            # detached snapshot only; the reset itself must not happen mid-forward, or
            # backward would differentiate activations that no longer match the weights
            mag = self._col_contrib(o).abs().mean(dim=(1, 2)).detach()
            self._last_share = mag / (mag.sum() + 1e-6)
        return super()._readout_aux(o)

    @torch.no_grad()
    def apply_redo(self, optimizer=None) -> int:
        """Re-init collapsed columns. Call AFTER optimizer.step(), never mid-forward."""
        if self.redo_every <= 0 or self._last_share is None:
            return 0
        step = self._aux_env_step()
        if step - self._redo_last < self.redo_every:
            return 0
        self._redo_last = step

        share = self._last_share
        dead = (share < self.redo_threshold).nonzero().flatten().tolist()
        if not dead:
            return 0
        dead = sorted(dead, key=lambda c: float(share[c]))[:self.redo_max_cols]

        H, touched = self.hidden_size, []
        for c in dead:
            for layer in range(self.n_layers):
                cell = self.cells[layer][c]
                for name, p in cell.named_parameters():
                    nn.init.xavier_uniform_(p) if p.dim() > 1 else nn.init.zeros_(p)
                    touched.append(p)
                # resetting the cell alone is not enough: the attention identities decide
                # whether anyone ever reads this column, and a revived cell nobody reads
                # dies again within a few thousand steps
                a = self.attn[layer]
                alpha = (1 / a.dim) ** 0.5
                a.ids_q[c].normal_(0.0, 0.1 * alpha)
                a.ids_k[c].normal_(0.0, 0.1 * alpha)
                touched += [a.ids_q, a.ids_k]
            if not self.pooled_head:
                # a head slice at exactly zero gets no gradient through its own
                # contribution and would stay dead by construction
                self.head.weight[:, c * H:(c + 1) * H].normal_(0.0, 0.01)
                touched.append(self.head.weight)

        # stale Adam moments would drag the fresh weights straight back; the ReDo paper
        # resets optimizer state for revived units for exactly this reason
        if optimizer is not None:
            for p in touched:
                optimizer.state.pop(p, None)

        self.redo_count += len(dead)
        print(f'[redo] step {int(step):,}: revived columns {dead}'
              f' (share {[round(float(share[c]), 4) for c in dead]})')
        return len(dead)
