#import "_template.typ": template
#show: template.with(title: "hgrnn")

= HopfieldGridRnn
The model investigates how well Modern Hopfield Networks (MHN) can
improve information exchange between Grid RNN columns compared to
standard multi-head attention. Unlike the base `GridRnn`, here GRUCell
is replaced by LSTM (as in the original associative memory paper) and
`HopfieldMessageLayer` is used — attention with a learnable scaling
parameter β per head. At large β the layer tends toward winner-take-all
mode, corresponding to Hopfield dynamics, while at small β it behaves
like standard softmax attention.

== Key mechanism
`HopfieldMessageLayer` replaces the fixed scaling `1/√d_k` with a
learnable `β = exp(log_β)`:

```python
# beta per head: (num_heads,) -> (num_heads, 1, 1, 1) for broadcast
beta = self.log_beta.exp().view(self.num_heads, 1, 1, 1)
# Hopfield energy score instead of standard 1/sqrt(d) scaling
scores = beta * torch.matmul(q, k.transpose(-2, -1))   # [heads, B, C, C]
attn = torch.softmax(scores, dim=-1)
out = torch.matmul(attn, v)                             # [heads, B, C, d_k]
```

Large β sharpens the attention, and the network operates as an
associative memory with a single key pattern. Initialized as
`log(1/√d_k)` — equivalent to standard scaling — and then learned.

== Important implementation details
#strong[LSTM instead of GRU.] State is stored as a pair `(h, c)`:

```python
h_ic, c_ic = cells[ic](x_list[ic], (hl[ic], cl[ic]))
```

Because of this `state` is a tuple `(h, c)` of shape
`(layers, cols, batch, hidden)` each, rather than a single tensor as in
`GridRnn`. Methods `reset_state`, `detach_state`, `init_state` operate
on both tensors.

#strong[Small out\_proj initialization.] So that the message has almost
no effect on state at the start of training, the output projection
weights are initialized with very small values:

```python
nn.init.normal_(self.out_proj.weight, 0.0, 0.001)
nn.init.zeros_(self.out_proj.bias)
```

#strong[LayerNorm on Hopfield layer output.]
`HopfieldMessageLayer.forward` returns `self.norm(out)` without a
residual connection — normalization stabilizes activations at large β.

== Hyperparameters
#table(
  columns: 2,
  inset: 6pt,
  [Parameter], [Description],
  [`log_beta`],
  [Logarithm of scaling coefficient β (one per head); the larger it is,
  the \"sharper\" the attention and the closer to Hopfield mode],
  [`n_attn_heads`],
  [Number of heads; `hidden_size` is truncated to the nearest multiple],
  [`messaging`],
  [`"post"` — attention after LSTM step; `"pre"` — before step, changes
  cell input dimension],
)

== Results
Configuration ~78K parameters (2 columns / 1 layer) in projects
`grid-rnn-sdq` and `grid-rnn-text`.

=== SDQ (Store-Distract-Query, hard)
#table(
  columns: 6,
  inset: 6pt,
  [Experiment], [H], [Acc], [Acc++], [Loss], [Steps],
  [hgrnn sdq ~78K (`grid-rnn-sdq`)],
  [128],
  [#strong[0.950]],
  [#strong[0.906]],
  [#strong[0.138]],
  [~81M],
)

This is substantially higher than the base `grnn` with the same topology
(Acc\=0.675, Acc++\=0.314 at 34M steps). The learnable β scaling
parameter in the Hopfield layer allows the model to focus attention more
sharply on the relevant columns, which is critical for the associative
retrieval task.

=== Text experiments
#table(
  columns: 7,
  inset: 6pt,
  [Experiment], [H], [Dataset], [Acc], [BPC], [PPL], [Steps],
  [hgrnn text8 ~78K (`grid-rnn-text`)],
  [110],
  [text8],
  [0.553],
  [2.090],
  [4.26],
  [~33M],
)

On text tasks the advantages of Hopfield attention over standard
attention are minimal: BPC\=2.090 is almost identical to the base grnn
(BPC\=2.088). Likely, \"soft\" attention is sufficient for
autoregressive prediction, while SDQ requires precise associative
retrieval.
