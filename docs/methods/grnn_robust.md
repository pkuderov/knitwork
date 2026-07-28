# GridRnnRobust (`grnn_robust`)

## Summary

A 12-column GridRNN can reach high SDQ accuracy while only two or three columns actually
carry the answer: the four-arm study in `docs/reviews/column_collapse_12c.md` measured an
effective column count of 2.68 on the unregularized baseline, with a single column whose
causal ablation delta was 0.79. `GridRnnRobust` subclasses
[`grnn_fix_v4`](grnn_fix_v4.md) and makes column robustness the training objective rather
than a property we hope for. It changes three things: the Barlow feature-decorrelation
term is off by default (the study found it correlates with *worse* CKA than activity
decorrelation alone), a head-balance penalty caps any column's share of the logit, and
columns are randomly dropped during training — the training-time analogue of the causal
ablation the offline analysis performs.

The guiding result from the study is that low pairwise CKA is the wrong target. The arm
with the best CKA (mean 0.325) was the most brittle of the four. Redundancy between
columns *is* the backup that makes ablation survivable. The quantity worth minimizing is
the **maximum** causal contribution of any single column, which is exactly what column
dropout attacks.

## Measured result: the hypothesis did not hold

The 61M-step run at 388.41k params came out **worse than `grnn_fix_v4_12c_reg2` on every
axis**: Acc 0.955 vs 0.962, max causal delta 0.380 vs 0.256, effective columns 6.16 vs
8.53, and 28/66 CKA pairs above 0.6 — the worst redundancy of all five arms.

The structure that emerged is two-tier: C0 and C3 are specialists (probe on `target` 0.658
and 0.747 against a 0.285 majority baseline, causal delta 0.306 and 0.380), while C4-C11
are eight interchangeable blanks (probe ~0.31-0.38, delta below 0.086). All the high CKA
sits among the blanks.

The likely mechanism: under a mean-pooled head, dropout demands that the *mean over a
random subset* still answer correctly. The cheapest way to satisfy that is to make the
droppable columns identical and harmless while leaving the load-bearing ones alone — at
p=0.1 the two specialists almost never drop out together, so the model simply absorbs the
rare loss. Dropout ends up punishing specialization in exactly the columns it removes.

Caveat on attribution: this run changed four things at once relative to REG2 (`aux_div`
0.05 → 0, `aux_div_max` 0.25 → 0.1, `aux_head` 0.05 → 0.15, dropout 0 → 0.1), so "dropout
is to blame" is a mechanism-backed hypothesis, not an established fact. Full numbers and
the follow-up plan are in `docs/reviews/column_collapse_12c.md`, section 3b.

**Do not use this model as a default.** It is kept as the reference implementation of the
column-dropout idea and as the base for `grnn_robust_concat`, which tests whether a
per-column readout reverses the outcome.

## Key mechanism

Column dropout hooks into `_mask_columns`, the extension point `grnn_fix_v4` exposes on
the post-attention output. Masks are drawn per column *and* per environment, so different
envs in the same batch see different surviving subsets:

```python
def _mask_columns(self, o):
    o = super()._mask_columns(o)                              # inference ablation
    if not self.training or self.col_dropout <= 0 or self.col_keep_mask is not None:
        return o
    p = self._col_dropout_p()
    C, B = o.shape[0], o.shape[1]
    keep = (torch.rand(C, B, 1, device=o.device) >= p).to(o.dtype)  # [C, B, 1]
    dead = keep.sum(dim=0, keepdim=True) == 0                 # [1, B, 1]
    keep = torch.where(dead, torch.ones_like(keep), keep)
    if self.col_dropout_rescale:
        keep = keep * (C / keep.sum(dim=0, keepdim=True).clamp(min=1.0))
    return o * keep
```

Two details matter. The `dead` guard restores the full state for any env that happened to
lose every column, which would otherwise feed a zero vector to the readout. And the
`col_keep_mask is not None` short-circuit keeps dropout out of the way during offline
causal ablation, so the analysis measures the trained model and not a second source of
noise.

The dropout rate ramps in over `col_dropout_ramp_steps` env-steps via `_aux_env_step()`,
sharing the same corrected step accounting as the aux schedules (the run script sets
`rnn.aux_tick_scale = gen.n_envs`; without it every schedule silently shrinks by a factor
of `n_envs`).

## Hyperparameters

| Name | Default | Notes |
|---|---|---|
| `col_dropout` | `0.1` | Per-column drop probability. At C=12 this removes ~1.2 columns per env per step. |
| `col_dropout_ramp_steps` | `0` (config: `1e7`) | Linear ramp in env-steps. Ramping avoids fighting the early curriculum, which is already hard enough. |
| `col_dropout_rescale` | `True` | Inverted-dropout scaling, so the expected magnitude of `o` matches eval. Disable to let the model see a genuinely smaller state. |
| `aux_div_weight` | `0.0` | Overrides the v4 default. The Barlow term at full weight measured *worse* CKA than activity decorrelation alone. |
| `aux_head_weight` | `0.15` | Overrides the v4 default. At 0.05 the max logit share fell 0.81 → 0.56 but missed the 2/C = 0.167 target, so this arm triples it. |

Both defaults use `setdefault`, so a config block can still restore them.

## Variants in `extend_config.yaml`

- `grnn_robust` — pooled (mean) head, E=64, H=44, L=2, C=12, 388,414 params (parity with
  every prior 12C arm).
- `grnn_robust_concat` — concat head, `pooled_head: false`, `aux_head_weight: 0.0`. The
  concat head costs +4,840 params, and `hidden_size` is quantized to multiples of
  `n_attn_heads`, so E=62 is the closest parity point: 386,932 params (−0.4%). This arm
  tests whether a per-column readout makes head balance regularization unnecessary,
  since each column already owns its own slice of the weight matrix.
