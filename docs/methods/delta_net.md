# DeltaNet baseline

Linear-attention recurrence with the delta rule (`baseline/delta_net.py`, `DeltaNetCore`). Each layer keeps a matrix memory `S` (`H x H`).
A new key/value pair is written as a correction of what the memory currently returns for that key, with a learned write gate.

## Key mechanism

```python
k = normalize(W_k(x)); v = W_v(x); b = sigmoid(W_b(x))
dv = v - S @ k                               # delta correction
S  = S + b * dv[:, :, None] * k[:, None, :]  # outer-product write
y  = S @ W_q(x)                              # read
```

State per sequence: `L * H^2` floats, which is much larger than the state of GRU/grid models of the same parameter count.

## Hyperparameters

`hidden_size`, `n_layers`; the feed-forward width is fixed at `2H`.
