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
