#import "_template_html.typ": template
#show: template.with(title: "grnn_loss")

= GridRnnLoss
GridRnnLoss extends the base GridRnn with auxiliary column diversity
loss functions. The problem it addresses: grid columns can collapse into
homogeneous representations, losing the potential for parallel
specialization. The method computes cosine, covariance, and entropy
losses over hidden states and gates across all layers, adding their
weighted sum to the main cross-entropy loss, thereby explicitly
encouraging diversity between columns.

== Key Mechanism
```python
# collect per-layer hidden states and gates during grid_step_postmsg
h_layer_list.append(hl_n)   # [cols, batch, H] per layer
gate_list.append(g)

# compute diversity loss from extras
div_losses = model.compute_diversity_loss(extras)
total_loss = ce_loss + div_losses['total']
```

The `grid_step_postmsg` method is overridden so that the standard
`extras` dictionary is augmented with `h_layers` — a list of hidden
states per layer. Then `compute_diversity_loss` passes them to
`ColumnDiversityLoss`, which aggregates several loss components.

== Important Implementation Details
#strong[Automatic layer weight configuration.] If `diversity_cfg` is not
provided explicitly, layer weights increase linearly from 0.5 to 2.0 —
deeper layers are penalized more:

```python
# layer weights grow linearly: 0.5 (first) .. 2.0 (last)
layer_w = [0.5 + 1.5 * i / max(n_layers - 1, 1) for i in range(n_layers)]
diversity_cfg = DiversityLossConfig(layer_weights=layer_w)
```

#strong[Safe zero return.] If `extras` does not contain `h_layers`
(e.g., when called without `return_attn`), the method returns zero
tensors for all components without error:

```python
if not h_layers:
    zero = torch.tensor(0.0)
    return {k: zero for k in ('cosine', 'covariance', 'variance', 'gate_entropy', 'total')}
```

== Hyperparameters
#table(
  columns: 2,
  inset: 6pt,
  [Parameter], [Description],
  [`diversity_cfg`],
  [Diversity loss configuration (`DiversityLossConfig`). If `None` —
  created automatically with linearly increasing layer weights],
)

== Results
Configuration: H\=128, 4 columns, 4 layers.

=== SDQ (Store-Distract-Query, hard)
#table(
  columns: 5,
  inset: 6pt,
  [Experiment], [Acc], [Acc++], [Loss], [Steps],
  [grnn with loss sdq (`grid-rnn-sdq`)],
  [#strong[0.862]],
  [#strong[0.743]],
  [#strong[0.338]],
  [~120M],
)

Best result among variants with explicit diversity loss: outperforms
grnn\_adv\_loss (Acc\=0.807) and grnn\_fusion (Acc\=0.831).
Automatically configured per-layer weight coefficients (linear growth
0.5→2.0) proved more effective than manual tuning in other approaches.
However, the model still falls short of grnn 4/3 without additional
losses (Acc\=0.960), indicating that columns specialize sufficiently at
the right topology even without an explicit regularizer.

=== Text experiments
#table(
  columns: 6,
  inset: 6pt,
  [Experiment], [Dataset], [Acc], [BPC], [PPL], [Steps],
  [grnn loss text (`grid-rnn-text`)],
  [text8],
  [0.570],
  [2.093],
  [4.27],
  [~100M],
)

On text8, diversity loss brings no improvement over the baseline grnn
2/1 (BPC\=2.088). The diversity loss components are useful in the SDQ
context but neutral for text experiments.
