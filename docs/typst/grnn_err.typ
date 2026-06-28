#import "_template.typ": template
#show: template.with(title: "grnn_err")

= GridRnn (with error signal)
This version of GridRnn solves the same sequential prediction task, but
explicitly introduces a prediction error signal into the grid. Instead
of a single input stream (the true token), column zero receives the
embedding of the predicted token from the previous step (`x_pred`), and
the first column receives the difference `x_true - x_pred`, representing
the prediction error. This allows the model to separate roles: one
column accumulates \"what I expected\", the other — \"how much I was
wrong\".

== Key Mechanism
The input tensor for the first layer is formed from two components —
prediction and error — instead of a single true embedding:

```python
def _prepare_grid_input(self, x_true, x_pred):
    xl = [x_pred, x_true - x_pred]   # [batch, embed_dim] each
    if self.n_columns > 2:
        dummy = torch.zeros(bsz, 1, device=x_true.device, dtype=x_true.dtype)
        for _ in range(1, self.n_columns):
            xl.append(dummy)
    return xl
```

Column 0 sees the previous step\'s prediction, column 1 sees the
difference (error); the rest receive a zero dummy input.

== Important Implementation Details
The prediction `x_pred` is computed from the previous step\'s logits via
softmax and a weighted sum of the embedding matrix rows:

```python
if self._y_last is not None:
    probs = torch.softmax(self._y_last, dim=-1)
    x_pred = torch.matmul(probs, self.embedding.weight)  # [batch, embed_dim]
else:
    x_pred = torch.zeros_like(x_true)
```

On the first episode step, `_y_last` is `None`, so `x_pred` is a zero
vector.

#line(length: 100%)

The `_y_last` state is reset together with the hidden state when a reset
mask is received:

```python
if self._y_last is not None:
    self._y_last = self._y_last.clone()
    self._y_last[:, ixs, :] *= 0.0   # reset finished episodes
```

Similarly to the hidden state, `_y_last` is detached from the graph on
`detach_state`.

== Hyperparameters
#table(
  columns: 2,
  inset: 6pt,
  [Parameter], [Description],
  [`n_columns`],
  [Minimum 2: column 0 — prediction, column 1 — error; the rest receive
  zero input],
  [`messaging`],
  [Only `"post"` is supported (attribute `use_postmsg`); `"pre"` is not
  overridden in this version],
)

== Results
Both experiments use the small configuration (~78K parameters, 2 columns
#"/" 1 layer).

=== SDQ (Store-Distract-Query, hard)
#table(
  columns: 6,
  inset: 6pt,
  [Experiment], [H], [Acc], [Acc++], [Loss], [Steps],
  [grnn\_err sdq ~78K (`grid-rnn-sdq`)],
  [110],
  [0.695],
  [0.434],
  [0.785],
  [~87M],
)

For comparison: the baseline `grnn` with an analogous configuration
(H\=115, 2/1) gives Acc\=0.734, Acc++\=0.494. The explicit prediction
error signal does not improve associative memory — the difference column
`x_true − x_pred` carries no information beyond the true token itself.

=== Text experiments
#table(
  columns: 7,
  inset: 6pt,
  [Experiment], [H], [Dataset], [Acc], [BPC], [PPL], [Steps],
  [grnn\_err text8 ~78K (`grid-rnn-text`)],
  [115],
  [text8],
  [0.547],
  [2.131],
  [4.38],
  [~34M],
)

On text8, the result is also below the baseline grnn (BPC\=2.088 over
the same 34M steps), confirming no advantage from the error signal on
the text prediction task.
