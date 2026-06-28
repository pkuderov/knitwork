#import "_template_html.typ": template
#show: template.with(title: "grnn_reservoir")

= GridRnnReservoir
GridRnnReservoir is a Grid RNN with Echo State Network (ESN)-like
reservoir columns. The idea: some columns in each layer are frozen after
initialization (weights are not updated), and their recurrent matrix is
scaled to a target spectral radius. This creates a rich random
projection space with controlled memory dynamics, which trainable
columns can read through the inter-column message passing mechanism.

== Key Mechanism
```python
# scale recurrent weight_hh of each GRU gate to target spectral radius
for gate_idx in range(3):
    block = cell.weight_hh.data[gate_idx * hid:(gate_idx + 1) * hid]  # [H, H]
    _scale_to_spectral_radius(block, spectral_radius)

# freeze reservoir columns — no gradients
for param in cell.parameters():
    param.requires_grad = False
```

The function `_scale_to_spectral_radius` uses `torch.linalg.eigvals` for
matrices up to 512×512 and power iteration for larger ones, then
multiplies weights by `target / current_radius`.

== Important Implementation Details
#strong[Split into trainable and reservoir columns.] The first
`n_trainable_cols` columns train normally, the last `n_reservoir_cols`
are frozen:

```python
first_reservoir = self.n_trainable_cols   # = n_columns - n_reservoir_cols
for icol in range(first_reservoir, self.n_columns):
    self._init_reservoir_cell(cell, spectral_radius, reservoir_scale)
    for param in cell.parameters():
        param.requires_grad = False
```

#strong[Reservoir cell initialization.] Input weights are uniformly
scaled to `reservoir_scale`, biases are zeroed for stability:

```python
nn.init.uniform_(cell.weight_ih, -scale, scale)
nn.init.zeros_(cell.bias_ih)
nn.init.zeros_(cell.bias_hh)
```

#strong[Spectral radius monitoring.] The `reservoir_info()` method
returns the actual spectral radii of all gates in reservoir cells —
useful for verifying correct initialization.

== Hyperparameters
#table(
  columns: 2,
  inset: 6pt,
  [Parameter], [Description],
  [`n_reservoir_cols`],
  [Number of frozen columns (\< n\_columns). Reservoir columns are last
  by index],
  [`spectral_radius`],
  [Spectral radius of the recurrent matrix. \< 1 — fading memory, ≈ 1 —
  critical regime (0.9 recommended)],
  [`reservoir_scale`],
  [Input weight initialization scale. Small values (0.1) ensure weak
  input influence on the reservoir],
)

== Results
=== SDQ (Store-Distract-Query, hard)
Two runs in `grid-rnn-sdq` with different numbers of columns and hidden
sizes:

#table(
  columns: 8,
  inset: 6pt,
  [Configuration], [H], [Train / Res. cols], [Layers], [Acc], [Acc++],
  [Loss], [Steps],
  [2 layers, 4 cols (3+1 res.)],
  [145],
  [3 / 1],
  [2],
  [0.693],
  [0.354],
  [0.826],
  [~47M],
  [3 layers, 5 cols (3+2 res.), bs\=300],
  [300],
  [3 / 2],
  [3],
  [#strong[0.925]],
  [#strong[0.851]],
  [#strong[0.187]],
  [~84M],
)

The large reservoir variant (H\=300, 5 cols, 3 layers) reaches the level
of hgrnn (~0.950), suggesting that the rich multi-scale dynamics of
frozen columns effectively substitute for trainable parameters. The
small variant (2 layers) is significantly worse — a reservoir without
sufficient depth does not realize its potential.

=== Text experiments
#table(
  columns: 8,
  inset: 6pt,
  [Configuration], [H], [Train / Res. cols], [Dataset], [Acc], [BPC],
  [PPL], [Steps],
  [5 cols (3+2 res.), 3 layers (`text-gru-reservoir`)],
  [128],
  [3 / 2],
  [shakespeare],
  [0.617],
  [1.780],
  [3.43],
  [~70M],
  [5 cols (3+2 res.), 3 layers (`text-hgru`)],
  [128],
  [3 / 2],
  [shakespeare],
  [0.628],
  [#strong[1.730]],
  [#strong[3.32]],
  [~70M],
  [5 cols (3+2 res.), 3 layers (`grid-rnn-text`)],
  [—],
  [3 / 2],
  [shakespeare],
  [0.612],
  [1.820],
  [3.53],
  [~156M],
)

On shakespeare, the reservoir variant with 5 columns (BPC\=1.730–1.820)
slightly falls behind the fully trainable grnn 4/3 (BPC\=1.721), but
significantly outperforms the baseline 2/1 (BPC\=1.954). HGRU cells in
the reservoir configuration are slightly better than GRU (BPC\=1.730 vs
1.780).
