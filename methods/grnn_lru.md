# GridLRU

GridLRU is a Grid RNN variant where standard GRU cells are replaced by LRU (Linear Recurrent Unit) blocks. LRU operates in a complex state space, allowing explicit control over the range of memorized temporal scales via the spectral radius parameter `r_max`. The model addresses the limited memory of GRU: thanks to a diagonal recurrent matrix in complex numbers, LRU stably retains long-range dependencies, while the column grid allows each column to specialize at its own temporal scale.

## Key Mechanism

```python
# each column gets its own r_max, linearly spaced across [r_min, r_max]
col_r_max = r_min + (r_max - r_min) * (icol + 1) / n_columns  # if lru_r_per_col

# LRUBlock returns two tensors: output activations and complex state
y_col, h_col_n = cells[icol](x_list[icol], hl[icol])
# y_col:   [batch, H]    — real-valued output
# h_col_n: [batch, 2*H]  — complex state (real + imag packed)
```

The network state has shape `[layers, cols, batch, 2*H]`, where the last dimension stores the real and imaginary parts of the LRU complex state. Output activations `[batch, H]` are formed separately and are not mixed with the state.

## Important Implementation Details

**Per-column r_max.** When `lru_r_per_col=True`, each column gets its own maximum memory radius, linearly increasing from `r_min` to `r_max`. Column 0 has short memory, the last column has long memory:

```python
col_r_max = r_min + (r_max - r_min) * (icol + 1) / n_columns
```

**Gated merge after message passing.** After message aggregation, a sigmoid gate controls the mixing of original and received activations:

```python
g      = torch.sigmoid(attn_gate(torch.cat([out_t, msg], dim=-1)))  # [cols, batch, 1]
merged = (1 - g) * out_t + g * msg                                  # [cols, batch, H]
```

**State initialization.** `init_state` returns a zero tensor with twice the size of the last dimension to store the complex state:

```python
# state: [layers, cols, batch, 2*hidden_size]
return torch.zeros(self.n_layers, self.n_columns, bsz, 2 * self.hidden_size, ...)
```

## Hyperparameters

| Parameter | Description |
|---|---|
| `r_min`, `r_max` | LRU spectral radius range. `r_max=0.999` approaches neutral memory (eigenvalues on the unit circle) |
| `lru_r_per_col` | If `True` — each column gets its own `r_max`, linearly increasing. Allows the grid to span different temporal scales |
| `ff_mult` | Feed-forward intermediate layer size multiplier inside LRUBlock |

## Results

### SDQ (Store-Distract-Query, hard)

Two runs in project `grid-rnn-sdq` with different model sizes:

| Configuration | H | Cols / Layers | r\_max | ff\_mult | Acc | Acc++ | Loss | Steps |
|---|---|---|---|---|---|---|---|---|
| grnn\_lru (`grid-rnn-sdq`) | 128 | 4 / 3 | 0.999 | 2 | 0.742 | 0.455 | 0.682 | ~50M |
| grnn\_lru\_wide (`grid-rnn-sdq`) | 256 | 4 / 4 | 0.9995 | 3 | **0.849** | **0.694** | **0.398** | ~87M |

The wide variant (H=256, 4 layers) significantly outperforms the base (H=128, 3 layers): Acc++ grows from 0.455 to 0.694. The larger r\_max (0.9995 vs 0.999) provides longer memory, which matters when distractors accumulate in SDQ.

### Text experiments

| Configuration | H | Cols / Layers | Dataset | Acc | BPC | PPL | Steps |
|---|---|---|---|---|---|---|---|
| grnn\_lru H=104 (`text-lru`) | 106 | 3 / 3 | shakespeare | **0.609** | **1.830** | **3.56** | ~46M |

On shakespeare, the LRU variant (BPC=1.830) falls behind grnn 4/3 (BPC=1.721 in 70M steps), but outperforms the baseline grnn 2/1 (BPC=1.954 in 146M steps). LRU's diagonal complex recurrence is competitive with GRU at fewer parameters and achieves good quality in 46M steps.

### MIKASA / POPGym (stopped at 15M/200M, ~7.5%)

| Environment | Memory type | EpRet | H | FPS | Outcome |
|---|---|---|---|---|---|
| RepeatFirstEasy | Object | ~−0.4…−0.5 | 1.07 | 338 | not learning, stopped |
| RepeatPreviousEasy | Object + Sequential | −0.246 | 0.64 | 337 | collapsed, stopped |

**RepeatPreviousEasy** — EpRet peaked at 0.93 by 3M steps, then entropy collapsed to H=0.19 and degraded. By 15M steps H=0.64, EpRet continues to fall — collapse is irreversible. Reason: LRU without explicit entropy regularization quickly determinizes upon first success.

**RepeatFirstEasy** — EpRet oscillates in −0.5…−0.4 with no trend. H=1.07 (healthy exploration), but no progress over 7.5% of training. LRU recurrence is insufficient for Object memory without cross-column attention (unlike hgrnn_lru).

Both runs stopped early to conserve resources.
