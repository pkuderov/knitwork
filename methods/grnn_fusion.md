# GridRnnFusion

GridRNN Fusion v3 — a hybrid architecture combining trainable HGRN cells with frozen reservoir columns. Addresses the problem of hidden representation homogeneity: reservoir columns with different spectral radii create rich multi-scale dynamics without additional parameters, while cross-attention lets trainable columns explicitly read from the reservoir. Diversity loss forces trainable columns to maintain diverse representations. Batched column operations eliminate the Python loop and provide a 3–5× speedup.

## Key Mechanism

At each layer: HGRN columns → cross-attention with reservoir → joint message passing → gated merge:

```python
# x_cols: (batch, n_cols, emb/hidden) — each column sees its own projection
h_t_new = self._batched_trainable_forward(li, x_t_in, h_t_in)  # (batch, n_t, hidden)
h_r_new = self._batched_reservoir_forward(li, x_r_in, h_r_in)  # (batch, n_r, hidden)

# trainable columns read from reservoir via cross-attention
if self.cross_attns is not None:
    h_t_new = self.cross_attns[li](h_t_new, h_r_new)

# joint message passing over all columns  [n_cols, batch, hidden]
h_all_seq = cat([h_t_new, h_r_new], dim=1).permute(1, 0, 2)
msg, attn_w = self.attn[li](h_all_seq, return_weights=return_attn)

# gate only for trainable columns
gate_logit = self.attn_gates[li](cat([h_t_seq, msg_t], dim=-1))
g = sigmoid(gate_logit)
h_t_merged = (1.0 - g) * h_t_seq + g * msg_t
```

## Important Implementation Details

**Batched HGRN cells** — HGRN formula without a Python loop over columns:

```python
# batched HGRN update  [n_t, batch, hidden]
o_t  = sigmoid(gx(W_o, b_o) + gh(U_o))          # output gate
c_t  = tanh(layer_norm(gx(W_c, b_c) + gh(U_c, o_t * h_p)))  # candidate
lam  = sigmoid(gx(W_f, b_f) + gh(U_f)) * (1 - betas) + betas  # forget (λ)
h_new = lam * h_p + (1.0 - lam) * c_t           # HGRN recurrence
```

Beta (`β`) is the lower bound of the forget gate, increasing from lower to upper layers, giving lower layers faster and upper layers slower dynamics.

**Reservoir columns** with different spectral radii for multi-scale memory:

```python
# spectral_radii assigned per reservoir column, e.g. [0.7, 0.95]
# W_hh frozen after init, scaled to target spectral radius
# GRU-like update without backprop through reservoir weights
```

**Diversity loss** — three components for column diversity:

```python
# cosine: penalizes cosine similarity > margin between column pairs
# covariance: penalizes off-diagonal covariance (VICReg-style)
# variance: penalizes low within-column variance
# gate_entropy: maximizes entropy of gate values
total = (cos_t + cov_t + var_t + gate_t) * cfg.total_weight
```

## Hyperparameters

| Parameter | Description |
|---|---|
| `n_reservoir_cols` | Number of frozen reservoir columns; default 2 out of `n_columns` |
| `spectral_radii` | List of spectral radii for each reservoir column; if None — assigned automatically (e.g., `[0.7, 0.95]`) |
| `reservoir_scale` | Scale of reservoir input weights |
| `beta_min / beta_max` | Range of λ lower bound in HGRN; `beta_min` for lower layers, `beta_max` for upper layers |
| `learnable_beta` | If True — `β` is learned, otherwise fixed |
| `use_cross_attention` | Enables cross-attention of trainable columns to reservoir columns |
| `all_cols_get_input` | All columns receive input through different orthogonal projections (not just the first) |
| `diversity_loss.total_weight` | Overall scale of diversity loss; default 0.05 |
| `diversity_loss.compute_every_n` | Compute diversity loss every N steps for speedup |
| `use_final_output_gate` | Sigmoid gate on top of the upper layer output before the head |

## Results

### SDQ (Store-Distract-Query, hard)

Both runs in project `grid-rnn-sdq`, configuration H=192:

| Version | all\_cols\_get\_input | diversity\_loss.total\_weight | Acc | Acc++ | Loss | Steps |
|---|---|---|---|---|---|---|
| grnn\_fusion v1 | `false` | 1.0 | **0.831** | **0.708** | 0.437 | ~101M |
| grnn\_fusion v2 | `true` | 0.05 | 0.823 | 0.646 | 0.437 | ~96M |

Both Fusion versions perform at the same level (~Loss=0.437). Variant v1 with stronger diversity\_loss (1.0 vs 0.05) and without wide input showed slightly better Acc++. Introducing `all_cols_get_input=True` in v2 provided no improvement and reduced Acc++. Compared to the baseline grnn 4/3 (Acc=0.960), Fusion with H=192 falls significantly short, indicating difficulties in balancing the multi-component loss.
