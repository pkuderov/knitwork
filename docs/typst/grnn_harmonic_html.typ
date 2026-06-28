#import "_template_html.typ": template
#show: template.with(title: "grnn_harmonic")

= HarmonicGridRNN
== Description
#strong[HarmonicGridRNN] — a harmonic Grid RNN combining four semantic
blocks into a unified architecture. The name reflects the \"harmonics\"
idea: each block operates at its own \"frequency\" and amplifies the
others. The goal is to overcome limitations of existing architectures:
`hgrnn_lru` is unstable in RL; `grnn_ema_mem` is weaker on associative
tasks; `grnn_delta` fails to train due to optimization complexity.

== Four Blocks
#table(
  columns: 4,
  inset: 6pt,
  [\#], [Block], [Trainable?], [Role],
  [1],
  [Spectral LRU],
  [✓],
  [2D hierarchy of temporal scales],
  [2],
  [Surprise-Delta Memory],
  [✓],
  [Adaptive KV memory without interference],
  [3],
  [Frozen Reservoir],
  [✗],
  [Long-term context (text), disabled for SDQ/RL],
  [4],
  [Hopfield Integration],
  [✓],
  [Sharp cross-column associative retrieval],
)

== Key Mechanisms
=== Block 1: Spectral LRU
#quote(block: true)[
Implementation:
#link("https://github.com/pkuderov/knitwork/blob/main/knitwork/models/hgrnn_lru.py#L13")[`hgrnn_lru.py — LRUCell`]
]

Each column and each layer has its own `r_max`, forming a 2D spectral
grid:

- Rows (layers): `r_base_layer` interpolated from `r_min_layers=0.7` to
  `r_max_layers=0.999`
- Columns: `r_col = r_min_col + (r_base_layer - r_min_col) * col_frac`

```python
# r_max[layer=0, col=0] ≈ 0.3   → τ ≈ 1 step (fastest)
# r_max[layer=2, col=3] ≈ 0.999 → τ ≈ 1000 steps (slowest)
```

=== Block 2: SurpriseDeltaMemory — key new module
#quote(block: true)[
Implementation:
#link("https://github.com/pkuderov/knitwork/blob/main/knitwork/models/grnn_harmonic.py#L30")[`grnn_harmonic.py — SurpriseDeltaMemory`]
]

EMA write gate + delta rule + adaptive forgetting:

```python
# 1. Parallel delta rule (no chained Jacobians):
v = F.normalize(proj_v(y), dim=-1)   # normalized → ||delta_W||_F bounded
v_pred = W.T @ k[c]
error  = v[c] - v_pred               # ∈ [-2, 2]
delta_W += k[c] ⊗ error              # outer product

# 2. EMA surprise — how "unexpected" the input was:
m_new = ema_beta * m + (1-ema_beta) * mean(error²)
alpha = (m_new / max(m_new + eps)).clamp(0, 1)   # batch-normalized

# 3. Adaptive forgetting:
fullness = ||W||_F / sqrt(dk * dv)
lam = lam_base * fullness.clamp(0, 1)

# 4. Delta update (per-layer delta_decay: 0.95 → 0.99):
W_new = (1-lam) * delta_decay[layer] * W + alpha * delta_W / C
```

#strong[Numerical stabilization (v2):]

- `delta_W /= C` — without normalization eigenvalue \= `decay - alpha*C`
  → explosion
- `F.normalize(v)` — delta\_W is bounded
- `alpha = m / max(m)` — no freeze when m≈0
- `out_norms[l]` — inter-layer LayerNorm

=== Block 3: Frozen Reservoir
#quote(block: true)[
Implementation:
#link("https://github.com/pkuderov/knitwork/blob/main/knitwork/models/grnn_harmonic.py#L152")[`grnn_harmonic.py — FrozenReservoir`]
]

Fixed random RNNs with different spectral radii. At `r=0.999`, τ≈1000
steps. For SDQ/RL: `n_reservoir_cols=0`.

=== Block 4: Hopfield Cross-Column Integration
#quote(block: true)[
Implementation:
#link("https://github.com/pkuderov/knitwork/blob/main/knitwork/models/hgrnn_lru.py#L99")[`hgrnn_lru.py — HopfieldMessageLayer`]
]

Modern Hopfield Networks with learnable β per head. Takes all trainable
columns plus reservoir projections.

=== Embedding Residual Skip
#quote(block: true)[
Implementation:
#link("https://github.com/pkuderov/knitwork/blob/main/knitwork/models/grnn_harmonic.py#L299")[`grnn_harmonic.py:299 — embed_skip`]
]

Direct gradient path from output to input embedding through each layer:

```python
y_lru = y_lru + self.embed_skip[l](x_embed).unsqueeze(0)  # [1, B, H] broadcast → [C, B, H]
```

`unsqueeze(0)` + broadcast applies one skip to all C columns — the
embedding does not specialize by column, only by content.

== Versions
#table(
  columns: 3,
  inset: 6pt,
  [Version], [Date], [Change],
  [v1],
  [06-09],
  [First implementation, NaN in gradients],
  [v2],
  [06-09],
  [Numerical stabilization: delta\_W/\=C, normalize(v), alpha\=m/max(m),
  out\_norms, per-layer delta\_decay],
  [v3],
  [06-12],
  [Broadcast x\_embed to all columns, multi-col head, per-layer EMA
  beta, rollout\_len\=16],
  [v3.1],
  [06-13],
  [Revert broadcast (harmed specialization), added multi\_col\_head
  param],
  [v4],
  [06-17],
  [Learned forget gate, Adam preconditioning (v2 buffer), pre-attn
  LayerNorm (Hopfield norm), col\_weights],
  [v5],
  [06-17],
  [Per-batch v2 in HarmonicState (bugfix RL), velocity-based surprise,
  noise in v, fixed lam],
)

== Hyperparameters
#table(
  columns: 3,
  inset: 6pt,
  [Parameter], [Value], [Description],
  [`hidden_size`],
  [128],
  [Hidden state dimensionality],
  [`n_layers`],
  [3],
  [Number of layers],
  [`n_columns`],
  [4],
  [Number of trainable columns],
  [`n_attn_heads`],
  [4],
  [Hopfield attention heads],
  [`dk`],
  [H#"/"#"/"4\=32],
  [Memory key dimensionality],
  [`dv`],
  [H\=128],
  [Memory value dimensionality],
  [`ema_beta`],
  [0.9],
  [EMA beta for top layer (slow write)],
  [`ema_beta_min`],
  [0.7],
  [EMA beta for bottom layer (fast write) — v3],
  [`delta_decay`],
  [0.99],
  [W decay for top layer],
  [`delta_decay_min`],
  [0.95],
  [W decay for bottom layer],
  [`lam_base`],
  [0.01],
  [Base forgetting rate],
  [`r_min_col`],
  [0.3],
  [r\_max for col\=0, layer\=0],
  [`r_min_layers`],
  [0.7],
  [r\_base for layer\=0],
  [`r_max_layers`],
  [0.999],
  [r\_base for last layer],
  [`multi_col_head`],
  [true/false],
  [mean(all cols) for LM/SDQ; col0 for RL],
  [`n_reservoir_cols`],
  [0/4],
  [0 for SDQ/RL, 4 for text],
)

== State
```python
HarmonicState(
    h:      [L, C, B, 2H]        # LRU (Re | Im); Im is detached
    h_res:  [L, C_res, B, H_res] # reservoir; always detached
    W:      [L, B, dk, dv]       # delta memory matrix
    m:      [L, B]               # EMA velocity-surprise per layer (v5)
    v2:     [L, B, dk]           # per-batch Adam preconditioner (v5, was global in v4)
    y_prev: [L, C, B, H]         # previous LRU output for velocity computation (v5)
)
```

#line(length: 100%)

== Experiment Results
=== SDQ v2 (\#043) — 2.11M params, 3L×4C, no reservoir
Launched 2026-06-10. Killed manually at 300M steps (out of 1000M).
fps\=3k.

#table(
  columns: 6,
  inset: 6pt,
  [Step], [Loss], [Acc], [Acc/query], [Acc/distract], [Acc++],
  [5M],
  [1.281],
  [0.519],
  [0.241],
  [—],
  [0.157],
  [10M],
  [0.911],
  [0.653],
  [0.391],
  [—],
  [0.295],
  [20M],
  [0.584],
  [0.775],
  [0.575],
  [—],
  [0.497],
  [30M],
  [0.478],
  [0.816],
  [0.656],
  [—],
  [0.590],
  [50M],
  [0.441],
  [0.831],
  [0.696],
  [—],
  [0.635],
  [75M],
  [0.419],
  [0.839],
  [0.722],
  [—],
  [0.664],
  [95M],
  [0.396],
  [0.849],
  [#strong[0.750]],
  [—],
  [0.696],
  [100M],
  [0.406],
  [0.845],
  [0.746],
  [—],
  [0.690],
  [110M],
  [0.449],
  [0.827],
  [0.722],
  [—],
  [0.659],
  [170M],
  [0.659],
  [0.740],
  [0.618],
  [—],
  [0.556],
  [200M],
  [0.533],
  [0.793],
  [0.676],
  [—],
  [0.615],
  [300M],
  [0.583],
  [0.773],
  [0.665],
  [—],
  [0.602],
)

#strong[Key observations:]

- Peak at 95M: Aq\=0.750, then instability and regression
- At 170M — sharp drop: Loss 0.44→0.66, Aq 0.694→0.618, then partial
  recovery
- Plateau/oscillation: Aq oscillates between 0.66 and 0.75 after 100M
  steps
- `Acc/store` \= NaN throughout training (sq\_gaps mask does not
  trigger)
- fps 3k → 2k toward the end (other tasks on GPU)

#strong[Memory diagnostics (at 300M):]

- `mem/W_norm`: L0\=1.06, L1\=1.46, L2\=1.88 — monotone growth across
  layers ✓ (slower layers write more)
- `mem/alpha`: L0\=0.85, L1\=0.86, L2\=0.87 — high and uniform (surprise
  is high, writes are active)
- `mem/surprise`: L0\=0.0075, L1\=0.007, L2\=0.006 — decreases across
  layers (upper layers are less \"surprised\")
- `mem/fullness`: L0\=0.016, L1\=0.023, L2\=0.029 — W matrix is 1.6-2.9%
  full, plenty of headroom
- `mem/error`: L0\=0.087, L1\=0.084, L2\=0.082 — slight decrease,
  prediction error is stable

=== SDQ v3.1 (\#055) — 2.11M params, 3L×4C, multi\_col\_head\=True
Launched 2026-06-12, killed at 110M. fps\=1k (shares GPU with text8
\#048).

#table(
  columns: 6,
  inset: 6pt,
  [Step], [Loss], [Acc], [Acc/query], [Acc/distract], [Acc++],
  [5M],
  [1.310],
  [0.513],
  [0.237],
  [—],
  [0.153],
  [10M],
  [0.968],
  [0.641],
  [0.380],
  [—],
  [0.272],
  [20M],
  [0.668],
  [0.749],
  [0.545],
  [—],
  [0.446],
  [30M],
  [0.548],
  [0.792],
  [0.625],
  [—],
  [0.537],
  [50M],
  [0.504],
  [0.807],
  [0.667],
  [0.963],
  [0.587],
  [75M],
  [0.481],
  [0.813],
  [0.687],
  [—],
  [0.613],
  [90M],
  [0.471],
  [0.816],
  [0.699],
  [—],
  [0.632],
  [100M],
  [0.475],
  [0.814],
  [0.699],
  [—],
  [0.634],
  [110M],
  [0.453],
  [0.822],
  [#strong[0.715]],
  [0.981],
  [0.653],
)

#strong[Memory diagnostics (at 110M):]

- `mem/W_norm`: L0\=1.26, L1\=0.93, L2\=1.61 — anomaly: L1 \< L0 (lower
  layer writes more than upper)
- `mem/alpha`: L0\=0.68, L1\=0.66, L2\=0.74 — lower than v2 (0.85) →
  writes are more selective
- `mem/surprise`: L0\=0.0039, L1\=0.0022, L2\=0.0022 — significantly
  lower than v2 (0.0075) → less \"surprise\"
- `mem/fullness`: L0\=0.019, L1\=0.014, L2\=0.024 — normal
- `mem/error`: L0\=0.063, L1\=0.046, L2\=0.051 — lower than v2 (0.087),
  prediction error decreased

#strong[Column diagnostics (at 110M) — CRITICAL:]

- `col/diversity`: L0\=2.34, L1\=2.80, L2\=#strong[7.79] — column
  diversity grows sharply toward upper layers
- `col/gate`: L0\=0.93, L1\=0.57, L2\=0.66 — gate at L0 is nearly
  saturated
- `col/col0_norm/L0`: 74.6 vs col1\=16.0, col2\=16.9, col3\=17.3 —
  #strong[col0 is 4.5x larger than others at L0]
- `col/col0_norm/L1`: 38.6, col1\=29.8, col2\=32.2, col3\=#strong[75.1]
  — #strong[col3 explodes at L1]
- `col/col0_norm/L2`: #strong[325.7], col1\=148.8, col2\=23.3,
  col3\=18.8 — #strong[catastrophic growth of col0 and col1 at L2]

#strong[Comparison v2 vs v3.1 by Acc/query:]

#table(
  columns: 4,
  inset: 6pt,
  [Step], [v2 Aq], [v3.1 Aq], [Δ],
  [10M],
  [0.391],
  [0.380],
  [-0.011],
  [30M],
  [0.656],
  [0.625],
  [-0.031],
  [50M],
  [0.696],
  [0.667],
  [-0.029],
  [75M],
  [0.722],
  [0.687],
  [-0.035],
  [95M],
  [#strong[0.750]],
  [0.701],
  [-0.049],
  [110M],
  [0.722],
  [#strong[0.715]],
  [-0.007],
)

#strong[Conclusion]: v3.1 converges SLOWER than v2 (lags ~0.03 Aq at
each step). Column norms grow unboundedly — potential explosion later.

#line(length: 100%)

=== text8 v3 (\#048) — 2.14M params, 3L×4C + 4Res, rollout\_len\=16
Launched 2026-06-12, killed at 270M (out of 520M). fps\=2k.

#table(
  columns: 5,
  inset: 6pt,
  [Step], [Loss], [BPC], [Acc], [T (context)],
  [5M],
  [1.961],
  [2.829],
  [0.407],
  [100],
  [10M],
  [1.783],
  [2.572],
  [0.458],
  [101],
  [20M],
  [1.556],
  [2.245],
  [0.521],
  [103],
  [30M],
  [1.423],
  [2.053],
  [0.559],
  [106],
  [50M],
  [1.313],
  [1.895],
  [0.589],
  [114],
  [75M],
  [1.264],
  [1.824],
  [0.603],
  [124],
  [100M],
  [1.237],
  [#strong[1.784]],
  [0.611],
  [135],
  [125M],
  [1.217],
  [1.756],
  [0.617],
  [147],
  [150M],
  [1.206],
  [1.740],
  [0.620],
  [160],
  [175M],
  [1.195],
  [1.725],
  [0.623],
  [174],
  [200M],
  [1.184],
  [1.708],
  [0.627],
  [189],
  [225M],
  [1.173],
  [1.692],
  [0.630],
  [206],
  [250M],
  [1.169],
  [1.686],
  [0.631],
  [224],
  [270M],
  [1.162],
  [#strong[1.676]],
  [0.634],
  [239],
)

#strong[Memory diagnostics (at 270M):]

- `mem/W_norm`: L0\=1.07, L1\=1.53, L2\=1.90 — monotone growth ✓
- `mem/alpha`: L0\=0.855, L1\=0.873, L2\=0.884 — high but stable
- `mem/surprise`: L0\=0.0075, L1\=0.0069, L2\=0.0066 — stable,
  decreasing across layers
- `mem/fullness`: L0\=0.017, L1\=0.024, L2\=0.030 — small, W is only
  1-3% full
- `mem/error`: L0\=0.086, L1\=0.083, L2\=0.082 — stable

#strong[Column diagnostics (at 270M):]

- `col/diversity`: L0\=1.92, L1\=1.80, L2\=2.16 — moderate, NO explosion
  (unlike SDQ)
- `col/gate`: L0\=0.356, L1\=0.428, L2\=0.671 — grows across layers
  (upper layers use Hopfield more)
- `col/col0_norm`: L0\=30.3, L1\=29.0, L2\=47.7 — moderate growth, no
  dominance

#strong[Comparison v2 vs v3 text8:]

#table(
  columns: 4,
  inset: 6pt,
  [Step], [v2 BPC (\#045)], [v3 BPC (\#048)], [Δ],
  [50M],
  [#strong[1.811]],
  [1.895],
  [+0.084 (v3 worse!)],
  [100M],
  [#strong[1.767]],
  [1.784],
  [+0.017],
  [150M],
  [#strong[1.737]],
  [1.740],
  [+0.003],
  [215M],
  [1.687],
  [—],
  [—],
  [270M],
  [—],
  [#strong[1.676]],
  [—],
)

#strong[Conclusion]: v3 is worse than v2 early on (broadcast x\_embed
harmed specialization), but converges by 150M. rollout\_len\=16 provides
richer BPTT context, which helps at long distances. BPC\=1.676 at 270M —
slow progress (~0.010 BPC/50M in late training).

#strong[Comparison with state of the art (char-level text8):]

#table(
  columns: 4,
  inset: 6pt,
  [Model], [Params], [BPC], [Source],
  [Transformer-XL],
  [277M],
  [1.06],
  [Dai et al., ACL 2019],
  [SHA-RNN],
  [53M],
  [1.067],
  [Merity, arXiv:1911.11423],
  [AWD-LSTM (3-layer)],
  [24M],
  [1.186],
  [Merity et al., ICLR 2018],
  [Mogrifier LSTM],
  [24M],
  [1.193],
  [Melis et al., ICLR 2020],
  [ON-LSTM],
  [4M],
  [~1.37],
  [Shen et al., ICLR 2019],
  [#strong[grnn\_harmonic v3]],
  [#strong[2.11M]],
  [#strong[1.676]],
  [this repo, \#048],
  [#strong[grnn\_harmonic v2]],
  [#strong[2.11M]],
  [1.687],
  [this repo, \#045],
)

#emph[No direct baseline LSTM at ~2M params on char text8 exists in the
literature. AWD-LSTM at 24M → 1.186 is the closest reference, but
comparison is invalid due to 12× parameter difference.]

#line(length: 100%)

=== text8 v2 (\#045) — killed early at 215M steps
#table(
  columns: 4,
  inset: 6pt,
  [Step], [Loss], [BPC], [Acc],
  [50M],
  [1.255],
  [1.811],
  [0.604],
  [100M],
  [1.225],
  [1.767],
  [0.611],
  [150M],
  [1.204],
  [1.737],
  [0.620],
  [215M],
  [1.169],
  [#strong[1.687]],
  [0.631],
)

#line(length: 100%)

=== Shakespeare v2 (\#044) — 2.22M params, 4Res
Killed early at 340M steps (out of 520M).

#table(
  columns: 4,
  inset: 6pt,
  [Step], [Loss], [BPC], [Acc],
  [50M],
  [0.930],
  [1.342],
  [0.749],
  [100M],
  [0.827],
  [1.193],
  [0.773],
  [200M],
  [0.692],
  [0.998],
  [0.799],
  [300M],
  [0.594],
  [0.857],
  [0.807],
  [340M],
  [0.547],
  [#strong[0.788]],
  [0.823],
)

BPC\=0.788 — a good result. Stable linear descent without plateau.

#line(length: 100%)

=== MIKASA RepeatFirstEasy v3.1 (\#058) — 2.11M params, 0Res,
multi\_col\_head\=False
Killed at 14.5M/200M steps (7%).

#table(
  columns: 5,
  inset: 6pt,
  [Step], [PL], [VL], [H (entropy)], [EpRet],
  [1M],
  [-0.000],
  [0.013],
  [1.02],
  [-0.479],
  [5M],
  [0.001],
  [0.014],
  [0.97],
  [-0.566],
  [8M],
  [0.002],
  [0.015],
  [0.86],
  [-0.618],
  [10M],
  [0.004],
  [0.016],
  [0.73],
  [-0.581],
  [12M],
  [0.002],
  [0.014],
  [0.76],
  [-0.447],
  [13.5M],
  [0.002],
  [0.012],
  [0.79],
  [-0.308],
  [14.5M],
  [0.002],
  [0.011],
  [0.83],
  [-0.419],
)

#strong[Observations:]

- VL\=0.011-0.016 — value function is training (in v3 VL\=0.000 →
  complete degradation)
- Entropy dropped from 1.10 → 0.83 — policy is specializing
- EpRet stays in -0.3 .. -0.6 — positive reward not yet reached
- PL slowly grows 0.001→0.004 — extremely slow policy progress
- Too few steps for evaluation (RepeatFirstEasy requires 20-50M for
  learning to begin)

#strong[For comparison]: grnn\_lru on RepeatFirstEasy reaches EpRet \> 0
around 30-50M steps.

#line(length: 100%)

== Issues and Hypotheses
=== 1. Unbounded column norm growth in SDQ
<1-unbounded-column-norm-growth-in-sdq>
```
col0_norm/L2: 50M → 206, 110M → 325  (growth +119 over 60M steps)
col3_norm/L1: 50M → 81, 110M → 75    (anomalously high)
```

Hopfield attention with softmax can sharply amplify one column
(attractor mode). LayerNorm after each layer was supposed to prevent
this, but it only works on output — before Hopfield, the hidden state
can grow.

#strong[Hypothesis]: Hopfield with high β (learnable) enters \"sharp
attractor\" mode, concentrating all information in a single column. This
degrades to a single-column RNN regime.

=== 2. Acc/store \= NaN
<2-accstore--nan>
The `Acc/store` metric (store accuracy) \= NaN throughout training in v2
and v3.1. Reason: the `sq_gaps < -1.0` mask does not trigger. This means
we are not measuring the quality of the store operation — we do not know
whether the model is writing correctly.

#strong[Implication]: it is unknown whether the problem is in writing
(store) or reading (query).

=== 3. SDQ v3.1 slower than v2
<3-sdq-v31-slower-than-v2>
v3.1 (multi\_col\_head\=True, per-layer EMA) converges slower:

- At 50M: v3.1 Aq\=0.667 vs v2 Aq\=0.696
- At 95M: v3.1 Aq\=0.701 vs v2 Aq\=0.750

Note that v3.1 runs at 1kfps (sharing GPU with text8), while v2 ran at
3kfps alone — step comparison is correct, but wall time is 3x longer.

=== 4. SDQ v2: instability after 100M
<4-sdq-v2-instability-after-100m>
At 170M — Aq drop 0.694→0.618 (Loss 0.44→0.66). Then partial recovery to
~0.675. The \"unlearning\" pattern points to catastrophic forgetting in
delta memory or LR schedule collapse.

=== 5. text8: slow progress in late steps
<5-text8-slow-progress-in-late-steps>
At 250-270M BPC changes by 0.010/50M steps — very slow. T (effective
context length) grew to 224-239 steps, but memory fullness remains small
(1-3%). Memory is effectively not used at full capacity.

=== 6. MIKASA: slow training
<6-mikasa-slow-training>
14.5M steps — task not solved yet. For RepeatFirstEasy ep\_len\=51,
reward\_scale\=1/50\=0.02. This is within normal range — usually
requires 30-80M steps. But current policy progress (PL~0.002) is very
small.

#line(length: 100%)

== Metrics — Interpretation
#table(
  columns: 3,
  inset: 6pt,
  [Metric], [Normal value], [Warning sign],
  [`mem/W_norm/Ll`],
  [0.5–2.5, grows across layers],
  [\> 5 or decreasing → memory overloaded],
  [`mem/alpha/Ll`],
  [0.5–0.9],
  [\< 0.1 → writes stopped; \= 1.0 → no regularization],
  [`mem/surprise/Ll`],
  [0.003–0.01],
  [→ 0 → model stopped training memory],
  [`mem/fullness/Ll`],
  [0.01–0.1],
  [\> 0.5 → matrix saturated, forgetting too weak],
  [`mem/error/Ll`],
  [0.05–0.15],
  [\> 0.3 → memory not working; \< 0.01 → perfect retrieval],
  [`col/diversity/Ll`],
  [1.5–3.0],
  [\> 5 → single column dominance],
  [`col/gate/Ll`],
  [0.3–0.7],
  [\> 0.95 → Hopfield saturated; \< 0.1 → ignored],
  [`col/col{i}_norm/Ll`],
  [15–50],
  [\> 100 → norm explosion, instability],
)
