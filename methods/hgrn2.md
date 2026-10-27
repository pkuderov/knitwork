# HGRN2 baseline

Gated linear RNN with state expansion (`baseline/hgrn2.py`, `HGRN2Core`). Each layer keeps a `H x (H * expand)` state updated with a
forget gate `f` and a coupled input gate `1 - f`; the outer product expands the state without adding recurrent parameters.

## Key mechanism

```python
f = sigmoid(W_f(x)); i = silu(W_i(x)) * (1 - f); g = silu(W_g(x))   # g: [B, H*expand]
h = f[..., None] * h + i[..., None] * g[:, None, :]                  # [B, H, H*expand]
y = norm(W_o(h.sum(1)))
```

State per sequence: `L * H^2 * expand` floats.

## Hyperparameters

`hidden_size`, `n_layers`, `expand` (4).

## Memory

The state `h` of shape `[B, H, H*expand]` is already the only large tensor saved per layer and step, so folding the reset (as done for
[DeltaNet](delta_net.md) and [mLSTM](mlstm.md)) gives no gain here. About 39 GB of activations for the mid config with 512 streams and a 64-step
rollout fall to roughly 9 GB with the runner option `grad_checkpoint=8` (activation checkpointing over 8-step segments, about 1.3x time,
identical gradients). See [baseline_memory](baseline_memory.md).
