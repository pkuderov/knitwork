# GridRnnBalance (`grnn_bal_causal`, `grnn_bal_lb`)

## Summary

The 12C study (`docs/reviews/column_collapse_12c.md`) ended with two open questions about
v4's head-balance penalty `_head_aux`. First, it measures the wrong quantity: it caps a
column's share of the *logit magnitude*, while the thing we actually care about is the
accuracy drop when that column is ablated — and REG2 cut magnitude monopoly to 0.285 while
one column still owned a causal delta of 0.256. Second, its form saturates: `relu(share.max()
- target)` sends gradient to exactly one column, so tripling its weight moved max share only
0.285 → 0.271.

`GridRnnBalance` subclasses `grnn_fix_v4` and adds two weight-gated terms that attack one
question each, so a config arm enables exactly one:

| arm | what is measured | how it is penalized |
|---|---|---|
| `grnn_fix_v4_12c_reg2` (reference) | logit magnitude | `relu(max share − target)` |
| `grnn_bal_causal` | leave-one-out effect | `relu(max share − target)` |
| `grnn_bal_lb` | logit magnitude | Switch load balance |

Neither term adds parameters, so both arms sit at exactly 388.41k — the same count as all
five earlier 12C runs.

## Key mechanism

**Causal balance.** Zero out each column in turn, push all C variants through the head, and
measure how far each drifts from the model's own current prediction. Zeroing (rather than
mean-substituting) matches `ablate_mode='zero'` in `inference/analyze_columns_sdq.py`, so
the training signal and the offline metric are the same quantity:

```python
keep = 1.0 - torch.eye(C, device=o.device, dtype=o.dtype)  # [C_drop, C]
od = o.unsqueeze(0) * keep[:, :, None, None]               # [C_drop, C, B, H]
logits_d = self.head(self._pool(od))                       # [C_drop, B, V]
ref = F.softmax(self.head(self._pool(o)), dim=-1).detach().unsqueeze(0)
kl = (ref * (ref.clamp_min(1e-8).log() - F.log_softmax(logits_d, dim=-1)))
kl = kl.sum(dim=-1).mean(dim=-1)                           # [C_drop]
share = kl / (kl.sum() + 1e-6)
return F.relu(share.max() - self.causal_share_target)
```

Two details carry the design. The reference distribution is **detached**: it is a target to
stay close to, not something the term may move. And the penalty is on *shares*, not raw KL —
otherwise the cheapest solution is to make every column irrelevant, which minimizes the
maximum by destroying the readout.

Cost is C extra evaluations of a `Linear(44, 10)`, on aux steps only.

**Load balance.** The Switch-Transformer auxiliary loss, with per-column contribution
magnitude standing in for the router distribution:

```python
mag = self._col_contrib(o).abs().mean(dim=2)               # [C, B]
p = mag / (mag.sum(dim=0, keepdim=True) + 1e-6)
P = p.mean(dim=1)                                          # [C]
f = F.one_hot(mag.argmax(dim=0), C).to(o.dtype).mean(dim=0)
return C * (f * P).sum() - 1.0
```

`f` is the empirical load and is deliberately non-differentiable; gradient flows through `P`.
The value is 0 for a uniform split and C−1 for total collapse, and every column contributes
on every step, so unlike `relu(max)` it does not switch off once the leader matches the
runner-up.

## Why these act from step 0

Both hook into `_readout_aux`, which — unlike the div and act terms — is **not** multiplied
by `_aux_scale()`. That is deliberate. `col_cka/frac_gt_06` shows the column structure is
settled by roughly 0.5M steps and then flat for the remaining 60M, so anything ramping in
over `aux_div_ramp_steps=1e7` arrives about twenty times too late to shape it.

## Hyperparameters

| Name | Default | Notes |
|---|---|---|
| `aux_causal_weight` | `0.0` (arm: `0.25`) | Off unless set. 0.25 puts the term near `aux_act` in magnitude. |
| `causal_share_target` | `2/C` = 0.167 | Same fair-share convention as `head_share_target`. |
| `aux_lb_weight` | `0.0` (arm: `0.05`) | Off unless set. The loss ranges over [0, C−1], so 0.05 caps its contribution near 0.55. |

`grnn_bal_causal` keeps `aux_head_weight: 0.05` from REG2 — the causal term is *added*, a
single change. `grnn_bal_lb` sets `aux_head_weight: 0.0`, since the load-balance term
*replaces* `relu(max)` on the same measured quantity.
