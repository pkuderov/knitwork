# Grid-LRU: detailed architecture

Grid-LRU keeps the grid of Grid-GRU but replaces every GRU cell with a **diagonal complex linear recurrent unit (LRU)**. The
implementation is `knitwork/models/grnn_lru_core.py` (`GridRnn`), selected with `--model=grnn_lru`. Two versions matter:

- **Language-modeling Grid-LRU** (AISTATS tables): static routing table, `fb_norm`, two to three layers.
- **RL Grid-LRU** (POPGym, October 2026): dynamic routing, `x_add`, per-column horizon bands, one layer. It is the best
  memory core found in the RL study, see [RL experiments](experiments/rl_2026-10.md).

The option-by-option reference is [grnn_lru_core](methods/grnn_lru_core.md); the parent architecture is
[Grid-GRU](methods/grid_gru.md).

## Motivation

The LRU simplification came out of the AAAI-27 review: the router regularizers were not ablated, and the GRU cell made it
hard to tell what the grid itself contributes. An LRU cell has explicit, interpretable memory timescales (the eigenvalue
modulus per channel), no gates and no saturating nonlinearity in the recurrence. It trains stably on long rollouts. The
question was whether the grid plus routing works on top of such a simple cell.

## Cell: complex diagonal LRU (`LruBank`)

Each of the `L x C` cells holds a complex state of width `H`, stored as `[re, im]`, so `2H` real floats:

```python
lam   = exp(-exp(log_r)) * exp(i * theta)      # |lam| < 1 per channel; horizon = 1 / exp(log_r)
gamma = sqrt(1 - |lam|^2)                      # input normalization, keeps the state variance constant
h     = lam * h + gamma * (x @ W_in)           # [C, B, 2H] complex update, no gates
out   = silu(h @ W_out) + x                    # real projection of the packed state, residual to the input
```

- `log_r` is initialized so that the horizons `1 / (1 - |lam|)` are log-uniform in `horizon = [1, 1000]` steps.
- `theta` is uniform in `[0, 2 pi)`: every channel is a damped oscillator.
- The cell output has a residual connection to the cell input. The GRU cell has none.

## Routing (static table, default)

```python
pi  = softmax(pi_route_logits)                 # [C, C + 1], diagonal initialized to 1, near zero elsewhere
msg = pi @ sources                             # sources = previous top outputs + external input
```

The static router has no query/key/value projections. It is a learned, input-independent mixing table, which is the
cheapest communication there is. The diagonal initialization starts every column reading mostly itself.

## One step and the feedback loop

```python
sources = cat([rms_norm(out_prev_top), x])     # fb_norm: parameter-free RMS norm on the fed-back output
for l in range(L):
    msg = route[l](sources)                    # [C, B, H]
    out, h[l] = lru_bank(l, msg, h[l])
    sources = out
y = out[:, 0]                                  # column 0 of the top layer
```

`fb_norm` was added after a stability bug: with the residual output, the loop through the top layer has gain above 1, and
an untrained model overflowed after about 1,200 steps without a state reset. Training reset the state on average every
~1,000 tokens, which hid the problem. The RMS norm adds no weights and keeps 20k-step rollouts finite.

## State and parameter accounting

| Config | `H` | Params | State floats | KiB |
|---|---:|---:|---:|---:|
| L1C8 | 176 | 1.004M | 4,224 | 16.5 |
| L2C4 | 180 | 1.049M | 3,600 | 14.1 |
| L2C8 | 128 | 1.060M | 5,120 | 20.0 |
| L3C4 | 148 | 1.063M | 4,144 | 16.2 |

State = `2LCH` (complex) + `CH` (fed-back output). The parameter budget per cell is `4H^2` (`W_in` and `W_out` are
`H x 2H`) against `6H^2` for a GRU cell. At the same total budget Grid-LRU columns are wider and carry about 3.3 times more
state than Grid-GRU.

## RL variant (best configuration, October 2026)

POPGym exposed a structural problem of the static grid: with a static softmax table the external input gets about 15% of
each column's input mass, so the observation is diluted before it reaches any memory. Three options fixed it:

```python
cell_in = route(q=out_prev, kv=[out_prev, x])          # routing='dynamic': query-key attention, 4 heads
cell_in = cell_in + x                                  # x_add: the observation reaches every layer-0 column directly
log_r[c] ~ band c of C log-spaced bands of [1, 1000]  # column_horizons: column c owns one range of timescales
```

| Setting | Value |
|---|---|
| model | `grnn_lru.L2C4` with `n_layers=1`, `hidden_size=224` (1.01M params with the actor-critic heads) |
| options | `routing=dynamic`, `x_add=true`, `column_horizons=true`, `feedback=true` (default) |
| AutoencodeEasy RL (seed 42) | 0.918 eval return (plain LRU: 0.38 over 3 seeds) |
| AutoencodeHard oracle accuracy @4.6M | 0.513 (LRU 0.338) |
| AutoencodeMedium, init from Easy | 0.29 (LRU 0.61): **transfer to longer episodes is the open problem** |

Single-layer, four-column grids won over two-layer grids, wider columns (C2) and more columns (C5 to C8). Removing the
recurrent feedback of the top outputs (`feedback=false`) broke learning completely (-0.16). The full list of tested
options and the verdict for each is on [RL experiments](experiments/rl_2026-10.md?id=hypotheses-tested).

## What the experiments say about Grid-LRU

| Claim | Status | Evidence |
|---|---|---|
| Grid-LRU matches Grid-GRU on Text8 at 1M | **rejected** | L3C4 1.6699 vs Grid-GRU L3C4 1.5918 BPC |
| Grid-LRU beats a single LRU early in training | **rejected (pilot)** | 67M-token rollout pilot: LRU 2.1575 vs Grid-LRU 2.4867 BPC; no 1B-token single-LRU run |
| Two read heads per column help | **supported (1 seed)** | -0.0185 BPC vs control |
| Gated writes / rotations help | **neutral** | -0.0023 / -0.0023 |
| Query-based reads / associative matrix memory help | **rejected** | +0.0199 / +0.0265 |
| A dynamic hub improves sparse routing | **supported (1 seed)** | -0.0102 vs dense control |
| Clockwork / block-sparse grids | **rejected** | +0.0323 / +0.0877 |
| Longer BPTT helps | **neutral** | rollout 512 vs 64: +0.0004 BPC |
| Dynamic routing + `x_add` + horizon bands make the grid beat LRU in RL | **supported on AutoencodeEasy**, not on other tasks | see the RL page |
