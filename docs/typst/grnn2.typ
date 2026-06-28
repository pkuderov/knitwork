#import "_template.typ": template
#show: template.with(title: "grnn2")

= GridRnn2
GridRnn2 extends the base GridRnn with three independently toggleable
mechanisms to improve generalization and training stability. The VAE
bottleneck on input replaces the deterministic embedding with a
probabilistic one: instead of an exact vector, the network receives a
sample from a Gaussian distribution whose parameters are predicted from
the embedding — this regularizes representations and adds a KL penalty
to the loss. Column Time-Gate allows each column to blend in the state
of its left neighbor column from the previous step, creating a \"wave\"
of left-to-right processing. Column Dropout randomly zeros out entire
columns during training, forcing each column to be useful independently
of its neighbors.

== Key mechanism
VAE embedding via the reparameterization trick allows gradients to flow
through stochastic sampling:

```python
def reparameterize(self, mu, log_var):
    if self.training:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std   # [batch, latent_dim]
    return mu                   # deterministic at inference

# KL penalty added to main loss
kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
return x, kl * self.kl_weight
```

At inference, deterministic `mu` is returned — without noise.

== Important implementation details
Column Time-Gate mixes the fresh state of column `j` with the state of
column `j-1` from the previous step via learnable element-wise gates:

```python
# h_new, h_prev: [cols, batch, hidden]
combined = torch.cat([h_new[j], h_prev[j - 1]], dim=-1)
g = torch.sigmoid(gate_fn(combined))           # [batch, hidden]
mixed = (1.0 - g) * h_new[j] + g * h_prev[j - 1]
```

Gate bias is initialized to a negative value (`delay_scale=-2.0`), so
initially the gates are nearly closed and do not interfere with
training.

#line(length: 100%)

Column Dropout scales the remaining columns by `1/(1-p)` to keep the
expected value unchanged:

```python
scale = 1.0 / (1.0 - self.drop_prob + 1e-8)
for i, col_idx in enumerate(range(start, self.n_columns)):
    if not keep[i]:
        result[col_idx] = 0.0
    else:
        result[col_idx] = result[col_idx] * scale
```

The zeroth (outer) column is always preserved (`keep_first=True`).

#line(length: 100%)

`forward` returns a triple `(logits, h, kl_loss)` instead of a pair, so
the calling code must sum `kl_loss` with the main cross-entropy:

```python
y, h, kl_loss = model(tokens, h)
loss = ce_loss + kl_loss   # kl_loss already scaled by kl_weight
```

== Hyperparameters
#table(
  columns: 2,
  inset: 6pt,
  [Parameter], [Description],
  [`vae_latent_dim`],
  [Dimensionality of VAE latent space; `None` disables VAE and uses
  standard `nn.Embedding`],
  [`vae_kl_weight`],
  [KL penalty weight; a small value (1e-4..1e-2) prevents regularization
  from dominating the main loss],
  [`use_time_gate`],
  [Enables Column Time-Gate; `False` — behavior identical to base
  GridRnn],
  [`time_gate_delay_scale`],
  [Initial bias of the delay gate; negative value \= gates nearly closed
  at start],
  [`col_drop_prob`],
  [Probability of zeroing one column per step; `0.0` disables Column
  Dropout],
)

== Results
=== Text experiments (text8)
Both runs in `grid-rnn-text`, configuration 3 columns / 2 layers:

#table(
  columns: 8,
  inset: 6pt,
  [Variant], [vae\_latent\_dim], [use\_time\_gate], [col\_drop\_prob],
  [Acc], [BPC], [PPL], [Steps],
  [grnn vae 64],
  [64],
  [`false`],
  [0.01],
  [0.557],
  [2.152],
  [4.44],
  [~28M],
  [grnn vae 48 + time gate],
  [48],
  [`true`],
  [0.01],
  [#strong[0.590]],
  [#strong[1.954]],
  [#strong[3.88]],
  [~181M],
)

The variant with time gate and a smaller latent space (48 vs 64) under
long training (181M steps) achieves BPC\=1.954, PPL\=3.88 — a result
comparable to baseline grnn 2/1 on shakespeare. The version without time
gate (BPC\=2.152) is worse than baseline grnn on text8 (BPC\=2.088). VAE
bottleneck without time gate is more of a hindrance: stochastic
embedding does not compensate for accuracy loss with a small number of
steps.
