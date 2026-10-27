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

## Optional variants (POPGym RL diagnostics)

All are off by default; the defaults reproduce the plain core.

| Option | Effect |
|---|---|
| `x_add=true` | add the external input to every layer-0 column input (bypasses routing dilution) |
| `feedback=false` | drop the previous top outputs from the layer-0 routing sources |
| `routing='dynamic'` | per-step query-key attention between columns (`StochasticMessagePassingLayer`); the input is also a key/value token |
| `routing='table'` | control for `dynamic`: same value/output projections and LayerNorm, query-independent weights |
| `comm_rounds=n` | n - 1 further exchanges among the received messages (shared weights), residual |
| `raw_columns=k` | the last k layer-0 columns take only the external input (no routed message): a plain LRU stream that other columns read via routing |
| `cell_act='linear'` | column output `y + x` without the silu (linear state readout) |
| `backbone=true` | the input first passes one LRU of the same width; the grid routes on its output (LRU backbone + routing grid hybrid) |
| `share_columns=true` | one input/output projection per layer shared by all columns; columns differ by horizons and routing ids |
| `layer_strides=[1, s]` | clockwork: layer l updates state and output only every s_l steps, otherwise holds them (temporal hierarchy) |
| `mem_slots=M, mem_stride=S` | every S steps the core output is written to a ring of M snapshot slots; slots plus an age embedding are extra layer-0 routing sources |
| `route_query='state'` | dynamic routing queries are a projection of the column's LRU state (`h @ W`, 2H -> H): route by memory content |
| `write_alloc=true` | soft content-addressed write: `cell_in *= C * softmax_c(<q(x), k(out_prev_c) + id_c>)`, zero-initialised keys |
| `route_query='current'` | dynamic routing queries come from this step (input, then the layer below); with `feedback=false` routing is feed-forward in time |
| `route_topk=k` | dynamic routing reads only the k best sources per column and head |
| `input_all_layers=true` | the external input is a routing source in every layer |
| `column_horizons=true` | column c initialises its horizons in the c-th of C log-spaced bands of `horizon` |
| `read_attn=true` | core output = column 0 + attention from the external input over top-layer columns |
| `write_topk=k` | only the k best-scoring columns update their state; all are still read |
| `write_mode='state'` | write score `<q(column input), k(column output)>` with keys shared across columns |
| `write_balance=u` | aux-loss-free load balancing: selection-only bias moved by `u * sign(k/C - load)` |
| `write_free=a` | selection prefers columns with low state RMS |
| `selective_rank=r` | input-dependent step `lam_t = lam^softplus(x W1 W2 + b)` |
| `input_gate_rank=r` | per-channel input gate `2 sigmoid(x W1 W2)` with fixed `lam` |
| `gru_columns=k` | the first k columns of each layer are GRU cells on the real half of the state |

Training-side options of the fast PPO runner (`knitwork/rl/ppo_gpu.py`), usable with any core:

| Option | Effect |
|---|---|
| `rl_wrapper.aux_lags=[k, ...]` + `rl.aux_coef=a` | aux head predicts `obs[t-k]` from the policy features (MSE, same episode only) |
| `rl.route_entropy_coef=c` | adds `-c * H(routing)` (normalised attention entropy); `c > 0` softer, `c < 0` sharper routing |

```python
score = (cell_in @ Wq) . (outs_prev @ Wk) / sqrt(d) + b     # write_mode='state', [C, B, 1]
sel   = score.detach() + write_bias - write_free * rms(h)  # selection only
write = sigmoid(score) * (sel >= topk(sel, k)[-1])         # h = h + write * (h_new - h)
```

## Sparse LRU experiments

`grnn_lru_sparse` extends this core with star/ring routing, periodic column updates, block projections and inter-layer channel shuffle, static entmax routing, and dynamic reading by column 0. Standalone mid configurations preserve the first-column output bottleneck and share the current baseline's feedback normalization. See [Sparse LRU mid experiments](../experiments/lru_sparse_mid.md) for the eight configurations, launch commands, parameter/state accounting, and local checks.
