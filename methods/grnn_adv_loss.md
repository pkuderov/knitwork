# GridRnn (grnn_adv_loss)

The model addresses the column collapse problem in Grid RNN — the situation where the grid's internal columns learn similar representations to each other and stop carrying distinct information. The core idea: introduce an auxiliary loss `ColumnSpecializationLoss` that explicitly penalizes decorrelation failure, low variance, cosine similarity, and "non-whitened" hidden states of columns, forcing them to specialize on different aspects of the input. In parallel, `MessagePassingLayer` is enhanced with stronger identity anchors, per-column Q/K projections, and post-attention nonlinearity that physically separates column spaces.

## Key mechanism

```python
# compute specialization loss over hidden states [L, C, B, D]
if compute_spec_loss:
    spec_loss, spec_details = self.spec_loss(h_new)
    extras["spec_loss"]    = spec_loss * self.spec_loss_weight
    extras["spec_details"] = spec_details
```

`ColumnSpecializationLoss` takes the full hidden state tensor `h_new` of shape `[layers, cols, batch, D]` and returns a scalar loss weighted by `spec_loss_weight`. The loss consists of four components (decorrelation, variance, cosine, whitening), each with its own `lambda`.

## Important implementation details

**Enhanced identity anchors in MessagePassingLayer:**

```python
# larger std => columns start further apart in embedding space
nn.init.normal_(self.ids, 0.0, 0.1 * xavier_alpha)  # vs 0.01 in base grnn
```

Increased standard deviation when initializing `ids` prevents columns from collapsing to a single representation in the early steps of training.

**Per-column Q/K projections:**

```python
# column-specific Q/K fingerprint via low-rank projection
proj = torch.einsum('cbd,cdp->cbp', qh, self.col_proj)        # [C, B, proj_dim]
proj = torch.einsum('cbp,cpd->cbd', proj, self.col_proj_out)  # [C, B, D]
qh = kh = qh + 0.1 * proj  # residual, keeps main signal intact
```

Each column has its own pair of projection matrices for forming a unique "fingerprint" in query and key space.

**Post-attention nonlinearity:**

```python
# per-column nonlinear transform after MHA
for c in range(C):
    out_list.append(h_mixed[c] + self.post_proj[c](h_mixed[c]))
```

Each column has its own two-layer MLP with SiLU activation, applied on top of the attention output as a residual.

## Hyperparameters

| Parameter | Description |
|---|---|
| `spec_lambda_decorr` | Weight of the decorrelation component in `ColumnSpecializationLoss`; controls how strongly linear dependence between columns is penalized |
| `spec_lambda_var` | Weight of the variance component; penalizes low intra-column variability |
| `spec_lambda_cosine` | Weight of the cosine penalty; direct penalty for angular similarity of representations |
| `spec_lambda_whiten` | Weight of the whitening penalty; requires isotropic distribution of activations |
| `spec_target_layers` | List of layers to which the specialization loss is applied; `None` = all layers |
| `spec_loss_weight` | Final scalar multiplier before adding to the total loss; allows annealing on a schedule |

## Results

Configuration: H=128, 4 columns, 4 layers.

### SDQ (Store-Distract-Query, hard)

| Experiment | Acc | Acc++ | Loss | Steps |
|---|---|---|---|---|
| grnn loss adv sdq (`grid-rnn-sdq`) | **0.807** | **0.629** | **0.505** | ~92M |

The result is comparable to grnn\_fusion v1 (Acc=0.831) and better than grnn\_disc (Acc≤0.655), but lower than grnn\_loss (Acc=0.862). The multi-component `ColumnSpecializationLoss` (decorrelation + variance + cosine + whitening) together with enhanced identity anchors and per-column Q/K projections gives a noticeable improvement over baseline grnn 2/1 (Acc=0.734), but the simpler diversity loss in grnn\_loss proves more effective.
