# Mamba baseline

Selective state-space model (Gu & Dao, arXiv 2312.00752) in its recurrent, one-token-per-step form (`baseline/mamba.py`,
`MambaCore`, registry key `mamba`). Each layer expands the input, applies a causal depthwise conv and an input-dependent diagonal SSM
(`B`, `C`, and a step size `dt` depend on the token), gated by a SiLU branch. This is a simplified variant: `dt` is a single
scalar per token projected to all channels.

## Key mechanism

```python
dA = exp(dt[..., None] * A)                  # A = -exp(log_A), [D, N]
h  = dA * h + (dt[..., None] * B[:, None]) * x_conv[..., None]   # [B, D, N]
y  = (h * C[:, None]).sum(-1) + D * x_conv
```

State per sequence: `L * d_inner * (d_state + d_conv - 1)` floats (SSM state plus conv buffer), `d_inner = expand * hidden_size`.

## Hyperparameters

`d_state=16`, `d_conv=4`, `expand=2`; `hidden_size` and `n_layers` set by the parameter budget.
Note: an earlier version of the layer read `dt` from the wrong slice of the projection; this was fixed when the core was added.
