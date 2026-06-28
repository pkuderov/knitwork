#import "_template.typ": template
#show: template.with(title: "grnn_ema_mem")

= GridRnnEmaMem
Grid RNN with surprise momentum-based memory (EMA Surprise Memory). Fast
Weights with constant decay (`grnn_fw`) writes to matrix `A` with equal
strength at every step, making no distinction between important and
routine inputs. This model solves the problem: write strength is
proportional to the exponentially moving average of the prediction error
(surprise EMA) — the model itself identifies \"unexpected\" inputs and
writes more strongly. In parallel, adaptive forgetting operates: the
higher the Frobenius norm of matrix `A`, the more intense the decay,
preventing memory overflow.

== Key mechanism
Accumulate errors across all columns, compute surprise EMA, normalize to
write strength `alpha`, apply adaptive forgetting `lam`:

```python
# accumulate prediction errors across columns, compute surprise EMA  [B,]
for col_j in range(n_cols):
    error = torch.bmm(A, k_j.unsqueeze(2)).squeeze(2) - v_j   # (B, H)
    surprise_acc += (error ** 2).mean(dim=-1)
    delta_A += torch.bmm(error.unsqueeze(2), k_j.unsqueeze(1)) # (B, H, H)

surprise = surprise_acc / n_cols
m_new    = ema_beta * m_prev + (1 - ema_beta) * surprise       # EMA state
alpha    = m_new / (m_new.detach().max() + 1e-6)               # write strength in [0,1]

fullness = A.detach().norm(dim=(-2,-1)) / H                    # capacity estimate (B,)
lam      = lam_base * fullness.clamp(0.0, 1.0)                 # adaptive forget rate

A_new = (1 - lam.view(B, 1, 1)) * A - alpha.view(B, 1, 1) * delta_A
```

`delta_A` is the sum of delta updates across all columns; multiplying by
`alpha` means that at zero surprise the write is fully suppressed.

== Important implementation details
#strong[Model state] — a triple `(h, A, m)`:

```python
# state = (h, A, m)
# h: (n_layers, n_cols, B, H)
# A: (n_layers, B, H, H)
# m: (n_layers, B)           ← surprise EMA per layer per sample
```

`m` is reset during `reset_state` together with `A`: accumulated
surprise history does not carry over between episodes.

#strong[Retrieval] — standard dot-product read from the updated `A`:

```python
# content-based retrieval after A is updated  [cols, B, H]
msgs = [torch.bmm(A_new, F.normalize(q[i], dim=-1).unsqueeze(2)).squeeze(2)
        for i in range(n_cols)]
h_msg = self.norm(torch.stack(msgs, dim=0))
```

#strong[Normalization trick]: `alpha` is normalized by the batch `max`
with `detach()`, to avoid passing gradients through the normalization —
this is a heuristic, not a learnable mechanism.

== Hyperparameters
#table(
  columns: 2,
  inset: 6pt,
  [Parameter], [Description],
  [`ema_beta`],
  [Surprise EMA decay; 0.9 — fast adaptation (~10 step memory), 0.99 —
  slow (~100 steps); high values smooth noise but lag behind],
  [`lam_base`],
  [Base forgetting rate at full matrix; 0.01 — weak forgetting, 0.1 —
  aggressive; scaled by `fullness`, so at empty matrix decay \= 0],
)

== Results
=== MIKASA / POPGym (in progress, 74M/200M steps)
#table(
  columns: 5,
  inset: 6pt,
  [Environment], [Memory type], [EpRet], [Progress], [Conclusion],
  [RepeatFirstEasy],
  [Object],
  [#strong[~0.95]],
  [74M/200M (37%)],
  [Nearly solved],
  [HigherLowerEasy],
  [Sequential],
  [~0.41],
  [73M/200M (37%)],
  [Partial learning],
)

#strong[RepeatFirstEasy] — the task reduces to retaining the first token
of the episode. The Surprise-EMA mechanism is well-suited: the first
token is maximally \"surprising\" (high EMA gradient), written strongly;
subsequent routine tokens produce low surprise and do not displace the
write. EpRet ~0.95 by 37% of training — the task is essentially solved.

#strong[HigherLowerEasy] — requires tracking relative card values across
a sequence of comparisons. EMA surprise is less specific here: each new
card can be \"surprising\" in different ways, without a direct link to
whether it needs to be remembered. EpRet ~0.41 — the model is learning,
but convergence is slow; task partially solved by 37% of training.
