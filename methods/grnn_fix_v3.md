# GridRnnFixV3

Third iteration of fixed inter-column attention. Diagnostics of v2 on SDQ (~97M steps) showed: gates opened (0.05 -> ~0.6) but degenerated into a constant with no variance across columns; attention maps stayed nearly uniform (beta never sharpened); the top layer went into tanh saturation (|h| norm ~ 7 out of 8, gradients 10x smaller than the bottom layer); MI(column, phase) ~ 0 — columns don't specialize by task phase. V3 adds to the v2 fixes: sharp beta init, RMSNorm on the inter-layer path, per-column gates with staggered init, and three column-specialization auxiliary losses.

## Key mechanism

Per-column gates and specialization losses; the message stays additive and the recurrent state stays protected:

```python
# per-column gates with staggered init  [C, B, 1]
g = torch.stack([torch.sigmoid(gates[ic](gate_in[ic])) for ic in range(C)], dim=0)
o = hl_n + g * msg          # additive message
h_n.append(hl_n)            # recurrent state stays unmixed
x = self.mid_norms[layer](o)  # RMSNorm: anti tanh-saturation for upper layers
```

Gate bias init is staggered (-2.5, -3.0, -3.5): columns start with different "openness" to other columns' messages, and the loss keeps them from collapsing back together.

## Column-specialization losses

Returned as the fourth forward value and folded into total_loss through the runner's kl-slot (weight ramps to 1.0 after 50k steps). Computed once every `aux_every=8` calls (~once per optimizer step) with a scaled weight — overhead is nearly zero:

```python
# (1) Barlow-style feature decorrelation: kills CKA-like redundancy [C, C, H, H]
z = (hl_n - mean) / std                # batch-standardized per feature
cross = torch.einsum('cbh,dbk->cdhk', z, z) / B
div = cross[iu, ju].pow(2).mean()
# (2) gate diversity: keep per-column mean gates apart
gate = F.relu(self.gate_std_target - gm.std())
# (3) activity decorrelation: columns update at different times
u = (hl_n - hl).norm(dim=-1)           # per-column update magnitude [C, B]
act = F.relu(corr[iu, ju]).mean()
```

Loss (1) targets the problem CKA revealed (0.65-0.72 between L1 columns in v2) but that cosine-similarity of mean states misses: it penalizes correlation between columns' **features**. Loss (3) is a label-free proxy for phase specialization: if columns update at different points in the stream (store vs distract vs query), the batch correlation of their |dh| drops; only positive correlation is penalized.

## Important implementation details

Learnable beta is initialized 3x sharper than standard scaling (`beta_scale=3.0` -> `log_beta = log(3/sqrt(d_k))`), so attention starts in a selective regime rather than as uniform averaging. In the RL runner (treasure) the aux losses must be disabled via zero weights — it unpacks strictly `(y, h)`.

## Hyperparameters

| Parameter | Description |
|---|---|
| `beta_scale` | 3.0 — initial attention sharpness relative to 1/sqrt(d_k) |
| `aux_div_weight` | 0.05 — weight of Barlow feature decorrelation between columns |
| `aux_gate_weight` | 0.02 — target std of per-column gates is 0.15 |
| `aux_act_weight` | 0.02 — decorrelation of column activity timing |
| `aux_every` | 8 — aux losses computed once every N calls (~once per optimizer step) |
| `hidden_size` | 64 at 2L x 3C gives ~202K parameters (parity with grnn_fix v2) |
