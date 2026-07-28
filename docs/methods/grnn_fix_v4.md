# GridRnnFixV4

Fourth iteration of fixed attention. Diagnostics of v3 on SDQ (~28M steps): Barlow decorrelation and beta sharpening worked (CKA L1 dropped from 0.65-0.72 to 0.30-0.52, attention maps gained structure, Acc++ 0.55@28M vs 0.38@28M for v2), but gates still degenerated into a shared constant and role specialization across columns didn't emerge. V4 adds **distinct inter-column attention** structurally, and deliberately task-agnostic (no tie to SDQ phases): per-column Q/K identities, per-column beta sharpness, an input-aware layer gate, and multi-timescale cell init.

## Key mechanism

Each column attends to its neighbors differently — its own query/key identities and its own learnable sharpness per head:

```python
# per-column identities: each column asks its own question  [C, 1, D]
q = self.W_q(h + self.ids_q)
k = self.W_k(h + self.ids_k)
# per-(column, head) sharpness, staggered init 0.5x..2x around beta_scale
beta = self.log_beta.exp().T.unsqueeze(1).unsqueeze(-1)   # [heads, 1, C, 1]
attn = torch.softmax(beta * q @ k.transpose(-2, -1), dim=-1)
```

Beta is indexed by the receiving column: one column reads neighbors selectively (beta -> 2x), another broadly (beta -> 0.5x); both modes remain learnable afterward.

## Important implementation details

The gate is a learnable per-column scalar (staggered init). Input-conditioned Linear gates degenerated into constants across all v3/v4/hgrnn-v4 runs — the task needs a shared mixing level rather than input-dependent routing, so the mechanism was honestly simplified and the parameters folded back into H:

```python
self.attn_gates.append(nn.Parameter(
    torch.tensor([-2.5 - 0.5 * ic for ic in range(n_columns)])
))
g = torch.sigmoid(gates).view(self.n_columns, 1, 1)   # [C, 1, 1]
```

Dropout (config `dropout`) applies to input projections and the inter-layer path; enabled at 0.1 for text8.

Multi-timescale prior: staggered init of the GRU update-gate bias across columns — fast/medium/slow column (analogous to HGRN's beta-floors and LRU's r_max hierarchy):

```python
# z -> 1 keeps old state (slow column); z -> 0 rewrites (fast)
shift = timescale_spread * (2 * ic / (n_columns - 1) - 1)
cell.bias_ih[H:2 * H] += shift
```

Everything else is as in v3: additive message with protected recurrent state, RMSNorm between layers, tiny-init `out_proj` with no post-norm, Barlow/gate-std/activity aux losses every `aux_every` calls; gate-loss weight raised 0.02 -> 0.1 (in v3 gates collapsed together — the hinge was too weak against the CE gradient).

Based on the v4 SDQ run (~50M), two more fixes were added: **Barlow weight grows with depth** (0.5 -> 2.0 across layers — top-layer CKA kept creeping back to 0.6, the collapsing pressure is stronger there) and an **upper-layer saturation penalty**:

```python
# anti tanh-saturation: penalize upper-layer |h| above target (0.8)
if layer > 0:
    aux_sat += F.relu(hl_n.abs().mean() - self.sat_target)
```

## Memory optimization (`optim` flag)

At >1M active weights and rollout=64 (text8) v4 blows up GPU memory. The dominant term is **not** the column attention (it is over <=8 columns) but the Barlow decorrelation tensor `cross = einsum('cbh,dbk->cdhk', z, z)` of shape `[C, C, H, H]`, i.e. `O(C^2 H^2)`, retained for backward on every aux step across the whole rollout. A single boolean `optim` switch (default off, fully backward-compatible) enables all of the following:

1. **Latent Barlow.** Project `z` from `H` to a small latent `d` (default 32) with a fixed orthonormal buffer before the cross-covariance, shrinking `[C,C,H,H] -> [C,C,d,d]` (~64x at H=256, d=32). This is the correct place to apply a "latent/MLA-style" compression — the real bottleneck, the diversity loss.
2. **Aux batch subsample.** Compute the Barlow statistic on a random `aux_batch_frac` (0.25) of the batch.
3. **Pooled head.** Mean-pool over columns then `Linear(H, V)` instead of `Linear(C*H, V)` — fewer params (matters at the 10M budget) and smaller head-gradient activations. Controlled by `pooled_head`, which follows `optim` unless set explicitly: at 12 columns the pooled head funnels 81% of the correct-class logit through a single column even when the causal load is spread across the grid.
4. **Per-step gradient checkpointing.** The model sets `grad_checkpoint = optim`; the SDQ/text runners wrap each timestep in `torch.utils.checkpoint`, so BPTT over rollout=64 no longer stores every step's internals. The aux counter is incremented under a grad-state guard so the two checkpoint passes stay consistent.

Measured (H=256, C=8, L=2, rollout=64, B=128): peak CUDA memory **3259 MB -> 305 MB (~10.7x)**. Scale presets `grnn_fix_v4_2m` (~2.0M) and `grnn_fix_v4_10m` (~10.1M) ship with `optim: true`.

`optim` used to also force `aux_act_weight = 0`, silently discarding the configured value. Over a 61M-step 12-column run that let 7 of 12 columns collapse into one subspace (pairwise CKA 0.51-0.83) with near-zero causal effect, since activity decorrelation is the only term penalizing synchronous column updates. That override is gone; see `docs/reviews/column_collapse_12c.md`.

## Aux schedule and column regularization

Aux weights follow a **ramp -> hold -> decay** envelope (`_aux_scale`), applied to both the Barlow terms and `aux_act`. Constant pressure to the end of training cost long-gap recall (`[16,32)`: 0.751 -> 0.719) and slowed curriculum progress, so the intended shape is: build column structure early, then release capacity back to the task.

```python
if self.aux_div_ramp_steps > 0 and step < self.aux_div_ramp_steps:
    return step / self.aux_div_ramp_steps          # ramp
if step <= max(self.aux_hold_steps, self.aux_div_ramp_steps):
    return 1.0                                      # hold
return 1.0 + (self.aux_final_frac - 1.0) * frac     # decay
```

All `*_steps` values are **env-steps**, the same unit as `n_steps`. The internal `_aux_tick` counts forward calls, so run scripts must set `rnn.aux_tick_scale = gen.n_envs`; without it every schedule is `n_envs` times shorter than configured (at `n_envs=128` a 2e7 ramp reached only 2.4% of nominal weight over 61M steps).

`aux_head_weight` (`_head_aux`) caps one column's share of the readout: per-column contributions `o_c @ W_c.T` are reduced to magnitude shares and `relu(share.max() - head_share_target)` is penalized, with `head_share_target` defaulting to `2/C`.

## Inference hooks (analysis only)

Two attributes support post-hoc column analysis (`inference/analyze_columns_sdq.py`) and are
inert during training — both default to `None`/`'zero'` and are only set by the analysis script:

- `attn_col_mask` `[C, C]` bool — restrict which columns may attend to which (attention ablation).
- `col_keep_mask` `[C]` (1=keep, 0=ablate) + `col_ablate_mode` `'zero'|'mean'` — causal column
  ablation applied to the post-attention output `o` at **every** layer, so a column is silenced
  both at the head (top layer) and as downstream input to the next layer. This lets the analysis
  measure the model's **own** end-to-end accuracy drop, not just a retrained probe.

```python
if self.col_keep_mask is not None:
    keep = self.col_keep_mask.view(self.n_columns, 1, 1).to(o.dtype)
    o = o * keep if self.col_ablate_mode != 'mean' else o * keep + (1 - keep) * o.mean(0, keepdim=True)
```

When `return_attn=True`, `grid_step` also returns `extras['o_cols']` `[C, B, H]` (the per-column
top-layer head input) so the analysis can do **exact** direct-logit attribution from the linear
head — for both the concat head (`optim=False`) and the mean-pooled head (`optim=True`, weight `1/C`).

## Hyperparameters

| Parameter | Description |
|---|---|
| `beta_scale` | 3.0 — center of the per-column sharpness spread (0.5x-2x around it) |
| `timescale_spread` | 1.0 — amplitude of the update-gate bias shift across columns (+-1) |
| `aux_gate_weight` | 0.1 — boosted x5 after v3's gate-diversity failure |
| `hidden_size` | 64 at 2L x 3C — ~203K parameters (parity with v2/v3) |
| `optim` | false — master switch for the memory optimizations above |
| `pooled_head` | follows `optim` — set explicitly to keep the concat head under `optim` |
| `div_latent_dim` | 32 — latent width for the Barlow projection when `optim` |
| `aux_batch_frac` | 0.25 — batch fraction used for the aux stats when `optim` |
| `aux_act_weight` | 0.02 — activity decorrelation; the only term against synchronous column updates |
| `aux_div_ramp_steps` / `aux_hold_steps` / `aux_decay_steps` | env-steps for the ramp/hold/decay envelope; `decay_steps=0` holds to the end (old behavior) |
| `aux_final_frac` | 1.0 — weight multiplier reached at the end of the decay |
| `aux_head_weight` | 0.0 — penalty on one column monopolizing the readout |
| `head_share_target` | `2/C` — allowed share of the logit magnitude per column |
