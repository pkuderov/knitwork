# grnn_delta — Grid RNN with Two-Scale Delta Memory

Grid RNN in which each layer is augmented with explicit associative memory based on the delta rule (Widrow-Hoff). The goal is to separate temporal dynamics (LRU state) from key-value associations (fast weight matrix): distractors do not overwrite stored pairs since they are held in a separate structure.

## Key mechanism

**TwoScaleMemLayer** — vectorized two-scale delta memory for all columns of a layer simultaneously:

```python
# for each layer l, all C columns are batched as [C*B, H]
k = normalize(Wk(h))            # write key
v = Wv(h)                        # value
q = normalize(Wq(h) + col_bias) # read key (per-col bias)
g = sigmoid(Wg(h))               # write gate

# delta rule: remove old association, write new one
v_old = W @ k                    # what is stored for key k
W ← decay * W + g * k ⊗ (v − v_old)

# read
m = W^T @ q
```

Two scales (fast: `dk_f=H//8`, slow: `dk_s=H//4`) with different per-layer decay. Layer 0: fast (decay_fast≈0.3, decay_slow≈0.95), layer L-1: slow (decay_fast≈0.7, decay_slow≈0.999).

Why the delta rule is better than Hebbian (`grnn_engram`):
- Hebbian: `W ← W + η·v⊗k` — accumulates interference between different keys
- Delta: explicitly removes the old entry for key k before the new one — exact overwrite without accumulated errors

**LRU with dual r_max hierarchy** — fast lower layers (r_max≈0.05) and slow upper layers (r_max≈0.999), plus per-column variation within a layer:

```python
r_max[l] = lerp(r_min_layers, r_max_layers, l / (L-1))
r_max[l, c] = r_max[l] * lerp(0.85, 1.0, c / (C-1))
```

**Cross-layer skip** (optional, `use_cross_layer_skip=True`): the top layer reads directly from the slow memory of the bottom layer, creating a "memory corridor":

```python
q_skip = normalize(Wq_skip(h_top))
m_skip = W_slow[layer=0].T @ q_skip
h_top ← h_top + Wo_skip(m_skip)
```

## Hyperparameters

| Parameter | SDQ default | Text default | Description |
|----------|-------------|--------------|----------|
| `dk_fast` / `dv_fast` | 16 | 16 | Fast memory dimensionality (H//8) |
| `dk_slow` / `dv_slow` | 32 | 32 | Slow memory dimensionality (H//4) |
| `mem_decay_fast` | [0.3, 0.5, 0.7] | [0.5, 0.7, 0.85, 0.9] | Fast memory decay per-layer |
| `mem_decay_slow` | [0.95, 0.98, 0.999] | [0.97, 0.985, 0.995, 0.999] | Slow memory decay per-layer |
| `r_min_layers` | 0.05 | 0.2 | r_max of the bottom LRU layer |
| `r_max_layers` | 0.999 | 0.9995 | r_max of the top LRU layer |
| `use_cross_layer_skip` | false | false | Memory corridor top→bottom |

## State

```python
class DeltaGridState(NamedTuple):
    h:      Tensor  # [L, C, B, 2H]          — LRU hidden states
    W_fast: Tensor  # [L, C, B, dk_f * dv_f] — fast delta matrices
    W_slow: Tensor  # [L, C, B, dk_s * dv_s] — slow delta matrices
```

## Results

### MIKASA / POPGym (preliminary, 22M/200M steps)

| Environment | Memory type | EpRet | Progress | FPS | Conclusion |
|---|---|---|---|---|---|
| RepeatFirstEasy | Object | **-0.48** | 22M/200M (11%) | 586 | Does not learn |
| HigherLowerEasy | Sequential | — | pending | — | — |

**RepeatFirstEasy** — at 22M steps (11% of training, LR already at 98%) EpRet is negative: the agent selects actions worse than random. The model is significantly slower: 586 fps vs ~1000 fps for grnn_ema_mem — the two-scale delta memory substantially increases computational load (~1.7× per step). Possible reasons for failure: (1) excessive model capacity for a simple Object task; (2) the delta rule handles exact key-value lookup well, but for RL tasks with sparse and noisy training signal it requires more warmup steps; (3) slow training speed due to high FLOPs per step limits the real number of updates per unit time.

> Results are preliminary — 11% of training. Final conclusion after a full run.
