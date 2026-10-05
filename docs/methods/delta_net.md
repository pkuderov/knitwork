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

## Memory: lazy reset

The text runner calls `reset_state` on every step. Multiplying the `[B, H, H]` state by the keep mask there created a full copy that autograd saved
for backward, in addition to the outer product and the new state: three large tensors per layer per step (about 56 GB for H=217, 3 layers,
512 streams, 64-step rollout). Now `reset_state` only records the 0/1 mask in the state (`state['keep']`) and the layer applies it through
per-environment scalars (`Sk = keep * (S k)`, `S = keep * S + (b dv) k^T`), with the write scaled before the outer product. Gradients are equal
up to rounding (relative 2e-5) and a step is slightly faster; peak activation memory drops about 2.8x (to roughly 19 GB). See
[baseline_memory](baseline_memory.md) for measurements and the optional `grad_checkpoint` runner option.
