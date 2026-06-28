#import "_template.typ": template
#show: template.with(title: "lru")

= LRUCell
`LRUCell` implements single-step recurrence in complex state space based
on the Linear Recurrent Unit. The key idea: a diagonal recurrent matrix
Λ in complex numbers allows explicit control of the temporal memory
range via spectral radius |λ| ∈ \[r\_min, r\_max\], while parameters ν
and θ are learned in log-space, guaranteeing gradient stability. The
cell solves the vanishing/exploding gradient problem of standard RNNs
while preserving long-term dependencies without LSTM/GRU-type
mechanisms.

== Key mechanism
```python
# u: [B, input_size]  h: [B, 2*hidden_size] — real + imag packed
def forward(self, u, h):
    h_re, h_im = h[:, :H], h[:, H:]
    lam_re, lam_im, gamma = self._lambda_gamma()   # derived from nu, theta

    # complex multiplication: Λ * h + γ * B * u
    new_re = lam_re * h_re - lam_im * h_im + gamma * self.B_re(u)
    new_im = lam_re * h_im + lam_im * h_re + gamma * self.B_im(u)
    h_n = torch.cat([new_re, new_im], dim=-1)      # [B, 2H]

    y = self.C(h_n)                                 # [B, H] — real output
    return y, h_n
```

`gamma = sqrt(1 - |λ|²)` normalizes the input so that signal energy is
independent of the chosen radius λ.

== Important implementation details
#strong[λ parameterization through log-space.] Spectral radius |λ| \=
exp(-exp(ν)) is always strictly less than 1 for any real ν, and phase θ
\= exp(θ\_param) ∈ (0, max\_phase) specifies the rotation:

```python
log_r     = -torch.exp(self.nu)                    # log(|lambda|) <= 0
lambda_re = torch.exp(log_r) * torch.cos(torch.exp(self.theta))
lambda_im = torch.exp(log_r) * torch.sin(torch.exp(self.theta))
gamma     = torch.sqrt((1.0 - torch.exp(2.0 * log_r)).clamp(min=1e-6))
```

#strong[Vectorized `forward_sequence`.] Projections B and D are computed
in a single call over the entire sequence `[T*B, ...]`, while the
recurrent loop iterates only over the time dimension:

```python
Bu_re = self.B_re(u_flat).view(T, B, -1)   # one matmul for entire sequence
for t in range(T):
    new_re = lam_re * h_re - lam_im * h_im + gamma * Bu_re[t]
```

#strong[`LRUBlock`.] Wrapper around `LRUCell` with normalization and
feed-forward:
`RMSNorm → LRUCell → GLU → residual → RMSNorm → PFFN → residual`.
Initialization of the last FF layer with small std (`0.01/sqrt(H)`)
suppresses the initial FF contribution.

== Hyperparameters
#table(
  columns: 2,
  inset: 6pt,
  [Parameter], [Description],
  [`r_min`, `r_max`],
  [Spectral radius range at initialization; ν is initialized uniformly
  in \[r\_min, r\_max\]],
  [`max_phase`],
  [Maximum initial phase θ (default 2π); limits the initial state
  rotation rate],
  [`use_d_feedthrough`],
  [If `True` — adds a direct path D·u to output y, improving
  approximation of high-frequency signals],
  [`ff_mult`],
  [(in `LRUBlock`) Feed-forward layer size multiplier relative to
  hidden\_size],
)

== Results
`LRUCell` is a component and is not tested directly. Results of models
using LRU as a recurrent cell are in:

- #link("grnn_lru.md")[grnn\_lru.md] — Grid RNN based on LRU (SDQ:
  Acc\=0.849, shakespeare Loss\=1.268)
- #link("hgrnn_lru.md")[hgrnn\_lru.md] — LRU + Hopfield attention (SDQ:
  Acc\=0.967 — best result by Acc++)
