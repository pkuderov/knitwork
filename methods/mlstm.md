# mLSTM baseline

Matrix LSTM with exponential gating from xLSTM (Beck et al., NeurIPS 2024; `baseline/mlstm.py`, `mLSTMCore`). The memory is an `H x H`
matrix written by outer products with input/forget gates kept in log space for stability.

## Key mechanism

```python
m_new = max(lf + m, li)                       # stabiliser
C = f * C + i * (v[:, :, None] * k[:, None, :])
n = f * n + i * k
y = (C @ q) / max(|n . q|, 1)
```

State per sequence: `L * (H^2 + H + 1)` floats (`C`, `n`, `m`).

## Hyperparameters

`hidden_size`, `n_layers`; forget-gate bias initialised to 3 (long memory).

## Memory: lazy reset

As in [DeltaNet](delta_net.md), the reset is no longer a multiplication of the `[B, H, H]` matrix state on every step. `reset_state` stores the
0/1 keep mask in the state; the layer folds it into the stabiliser (`m * keep`) and the forget gate (`f * keep`), and the write uses `(i v) k^T`.
This is exactly the reset semantics (a reset environment sees `C = n = m = 0`), with about 2.8x lower peak activation memory
and relative gradient differences of about 2e-5. Details and the optional `grad_checkpoint` runner option: [baseline_memory](baseline_memory.md).
