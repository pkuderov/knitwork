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
