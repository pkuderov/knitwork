# GridRnnFeedback

Base GridRNN with **top-down feedback**: in the plain GridRNN only column 0 receives external input (the token embedding) and every other column starts each step from a dummy zero. Here each buffer column `c >= 1` instead receives, as its layer-0 input, **its own top-layer output from the previous step** (projected back to embedding size) plus a distinct learnable seed. This gives buffer columns a self-referential recurrent loop through the whole grid and, via the seeds, breaks the initial symmetry so they specialise differently. Post-messaging attention only.

## Key mechanism

Layer-0 inputs are built from the previous state's top layer (`h[-1]`), one projection + one seed per buffer column:

```python
prev_top = h[-1]                                   # [C, B, H] previous step, top layer
xl = [x]                                            # col 0: token embedding
for c in range(1, self.n_columns):
    fb = self.fb_proj[c - 1](prev_top[c])           # [B, E] top-down feedback
    xl.append(self.col_seeds[c - 1] + fb)           # + distinct learnable seed
```

At reset the previous state is zero, so the input collapses to exactly `col_seeds[c-1]` — a distinct starting point per column. `fb_proj` is initialised small so the seeds dominate early and the feedback grows in as columns differentiate. The rest is standard post-messaging GridRNN: per-cell GRU, `MessagePassingLayer` attention over columns, input-conditioned mixing gate `g = sigmoid(attn_gate([h; msg]))`, head reads the top-layer external column.

## Hyperparameters

| Parameter | Description |
|---|---|
| `seed_scale` | 1.0 — scale of the orthogonally-initialised per-column seed vectors (symmetry breaking strength) |
| `fb_init` | 0.1 — std scale of the top-down feedback projection; small so seeds lead early training |
| `col_identities` | true — learnable per-column identities inside `MessagePassingLayer` |
| `n_columns` | > 1; buffer columns (all but col 0) get the feedback loop |
