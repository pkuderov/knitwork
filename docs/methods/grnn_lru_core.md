# GridRnn-LRU core (grnn_lru_core)

Simplified Grid RNN used by `--model=grnn_lru` (`knitwork/models/grnn_lru_core.py`). Compared to [grnn_core](grnn_core.md) it drops the
per-cell GRU, attention heads, bias and regularisers: each of the `L x C` cells is a **diagonal complex linear recurrent unit** and routing is a
learned, query-independent softmax table between columns.

## Key mechanism

```python
lam = exp(-exp(log_r)) * exp(i * theta)                 # |lam| < 1, per channel
h   = lam * h + gamma * (x @ W_in)                      # complex state, stored as [re, im] -> [C, B, 2H]
out = silu(h @ W_out) + x                               # real projection of the packed state, residual
msg = softmax(pi_route_logits) @ v                      # StaticMessagePassingLayer, [B, C, H]
```

State per sequence: `h` is `2 * L * C * H`; only the last-layer output (`C * H`) is fed back, so the carried state is `2LCH + CH`
(`knitwork.common.state_size` follows this convention; the per-layer `outs` tensor is not used by the static router and is not counted).
The step input is batch-first, `(B, n_inputs, H)` (`batch_first = True`).

## Hyperparameters

`horizon=[min, max]`: initial e-folding range in steps (log-uniform), controls the memory timescales.
Stability: the residual feedback of `out` into the next message pool had loop gain above 1, so an untrained model overflowed after about
1200 steps without a state reset (training resets average every ~1000 steps, which hid it). `fb_norm=true` applies a parameter-free RMS norm to the
fed-back output (also in `reset_state`); it adds no weights and keeps 20k-step rollouts finite. It is enabled in the text configs.

## Sparse LRU experiments

`grnn_lru_sparse` extends this core with star/ring routing, periodic column updates, block projections and inter-layer channel shuffle, static entmax routing, and dynamic reading by column 0. Standalone mid configurations preserve the first-column output bottleneck and share the current baseline's feedback normalization. See [Sparse LRU mid experiments](../experiments/lru_sparse_mid.md) for the eight configurations, launch commands, parameter/state accounting, and local checks.
