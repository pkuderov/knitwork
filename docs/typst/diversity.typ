#import "_template.typ": template
#show: template.with(title: "diversity")

= ColumnDiversityLoss
`ColumnDiversityLoss` addresses the representation collapse problem in
Grid RNN: without a special regularizer, the network\'s columns tend to
converge to similar hidden states and stop specializing. The module
simultaneously minimizes cosine similarity between column pairs,
suppresses covariance within each column (analogous to VICReg),
encourages sufficient activation variance, and maximizes the entropy of
aggregation gates — four weighted components summed into `total`.

== Key mechanism
```python
# h: [cols, B, H] — hidden states of all columns at one layer
def cosine_diversity(self, h, lw):
    for i in range(cols):
        for j in range(i + 1, cols):
            sim = F.cosine_similarity(h[i], h[j], dim=-1).mean()
            loss += F.relu(sim - self.cfg.cosine_margin)  # penalize only above margin
    return lw * self.cfg.cosine_weight * (loss / n_pairs)
```

The penalty accumulates only for pairs whose similarity exceeds the
`cosine_margin` threshold, allowing columns to maintain moderate
correlation without requiring orthogonality.

== Important implementation details
#strong[Layer weighting.] `layer_weights` is a buffer of shape
`[n_layers]`, allowing stronger loss on upper layers where
specialization matters more:

```python
lw = self.layer_weights[i].item()
cos += self.cosine_diversity(h, lw)
```

#strong[Covariance loss (VICReg-style).] Suppresses off-diagonal
correlations between dimensions of a single column\'s hidden vector:

```python
z   = h[c] - h[c].mean(dim=0, keepdim=True)       # [B, H] centered
cov = (z.T @ z) / (bsz - 1)                        # [H, H]
loss += (cov * (1.0 - eye)).pow(2).sum() / d        # off-diagonal only
```

#strong[Gate entropy loss.] Penalizes overly confident (near 0 or 1)
aggregation gates, encouraging more uniform mixing:

```python
H = -(gc * gc.log() + (1.0 - gc) * (1.0 - gc).log())
total -= H.mean()   # negative entropy = positive loss
```

== Hyperparameters
#table(
  columns: 2,
  inset: 6pt,
  [Parameter], [Description],
  [`cosine_margin`],
  [Threshold below which cosine similarity is not penalized. Allows
  columns to remain moderately similar],
  [`var_threshold`],
  [Target standard deviation of activations; loss activates only when
  std \< threshold (hinge form)],
  [`layer_weights`],
  [List of per-layer weights; if `None` — all layers are equal],
  [`gate_entropy_weight`],
  [Weight of gate entropy penalty; typically the smallest of the four,
  since gates may legitimately be saturated],
)

== Results
`ColumnDiversityLoss` is a component module. Results of models using
this loss:

#table(
  columns: 4,
  inset: 6pt,
  [Model], [SDQ Acc], [SDQ Acc++], [Note],
  [#link("grnn_loss.md")[grnn\_loss]],
  [#strong[0.862]],
  [#strong[0.743]],
  [best of the diversity family],
  [#link("grnn_fusion.md")[grnn\_fusion]],
  [0.831],
  [0.708],
  [diversity + HGRN + reservoir],
  [#link("grnn_adv_loss.md")[grnn\_adv\_loss]],
  [0.807],
  [0.629],
  [ColumnSpecializationLoss (4 components)],
)

Automatically tuned per-layer weights in `grnn_loss` outperform manual
tuning in other approaches. Diversity loss on language modeling tasks
does not improve quality relative to baseline grnn.
