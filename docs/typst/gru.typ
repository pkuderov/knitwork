#import "_template.typ": template
#show: template.with(title: "gru")

= GruBaseline
`GruBaseline` is a standard multi-layer GRU recurrent network serving as
a baseline for comparison with GridRNN. The model solves sequential
prediction tasks (associative memory, language modeling): tokens pass
through an embedding, then through a stack of GRU layers, and finally
through a linear head that produces logits. The hidden size can be set
explicitly or computed automatically from `base_hidden_size` so that the
total parameter count matches a single-layer GRU reference.

== Key mechanism
The forward pass adds a sequence dimension, runs through `nn.GRU`, and
removes it again so the interface matches the other models:

```python
tokens = tokens.unsqueeze(0)           # [1, batch, 1]
x = self.embedding(tokens.view(-1))   # [batch, embed_dim]
x = x.view(seq_sz, bsz, -1)           # [1, batch, embed_dim]
y, hN = self.rnn(x, h0)
logits = self.head(y).squeeze(0)       # [batch, output_size]
```

This allows processing one token per step in autoregressive mode.

== Important implementation details
Hidden state reset is implemented via multiplication by a mask rather
than indexed assignment — this works correctly with autograd:

```python
def reset_state(self, state, reset_mask):
    if not torch.any(reset_mask) or state is None:
        return state
    keep_mask = torch.logical_not(reset_mask).to(self.head.weight.dtype)
    return state * keep_mask[None, :, None]   # [layers, batch, hidden]
```

Unlike GridRNN, here `state` is a dense tensor
`[layers, batch, hidden]`, so a single broadcast multiplication
suffices.

== Hyperparameters
#table(
  columns: 2,
  inset: 6pt,
  [Parameter], [Description],
  [`base_hidden_size`],
  [Reference hidden state size; if `hidden_size` is not set explicitly,
  the model selects `hidden_size` so that the parameter count matches a
  single-layer GRU with this size],
  [`dropout`],
  [Automatically zeroed when `n_layers == 1`, since `nn.GRU` does not
  apply dropout to the last layer],
)

== Results
GRU Baseline serves as the baseline for comparing all Grid RNN models.

=== SDQ (Store-Distract-Query, hard)
#table(
  columns: 7,
  inset: 6pt,
  [Configuration], [H], [Layers], [Acc], [Acc++], [Loss], [Steps],
  [rnn H\=128 SDQ (`sdq-gru`)],
  [128],
  [1],
  [0.689],
  [0.471],
  [0.847],
  [~100M],
)

LSTM variant from the same series: lstm H\=128 SDQ (`sdq-lstm`) —
Acc\=0.683, Acc++\=0.455, Loss\=0.847 at ~100M steps. GRU and LSTM
behave almost identically.

=== Text experiments (text8)
#table(
  columns: 8,
  inset: 6pt,
  [Configuration], [H], [Layers], [Dataset], [Acc], [BPC], [PPL],
  [Steps],
  [rnn H\=128 (`grid-rnn-text`)],
  [128],
  [1],
  [text8],
  [0.569],
  [2.024],
  [4.07],
  [~79M],
)

The baseline GRU sets the lower bound: Acc\=0.689 on SDQ, BPC\=2.024 /
PPL\=4.07 on text8. All Grid RNN models with 4+ columns / 3+ layers
outperform it on both tasks.
