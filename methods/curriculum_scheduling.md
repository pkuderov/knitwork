# Curriculum Scheduling in SDQ

## General idea

In the SDQ experiment, task difficulty increases gradually: sequence length (`T`) grows, store/query operation probabilities decrease (more distractors). The curriculum scheduler decides when to advance to the next difficulty level based on the model's current performance.

The scheduler checks metrics every `schedule` steps (default 200k). It also adaptively adjusts the check frequency: speeds up (×0.97) with stable progress, slows down (×1.25) during stagnation. The adjustment range is 0.25× to 4× the base schedule.

At each accepted curriculum step:
```
T       += 0.1
p_store  = max(p_store - 0.00014, 0.10)
p_query  = max(p_query - 0.00005, 0.25)
```

Configuration in the config — `curriculum:` section in `extend_config.yaml`.

---

## Modes

### `mode: speed` (default)

Original mode. Accepts a curriculum step if the exponential moving average of the metric improvement rate (`avg_speed`) is positive — i.e., the metric is improving on average.

```python
speed = sign * (val - last_val)   # improvement over one interval
avg_speed += lr * (speed - avg_speed)
accept = avg_speed > 0.0
```

**Problem**: Loss almost always decreases at the start of training, so curriculum effectively ticks on schedule without looking at actual quality.

**When to use**: quick runs, debugging, baseline comparison.

```yaml
curriculum:
  schedule: 2e+5
  key: Loss
  mode: speed
```

---

### `mode: threshold`

Adds a minimum competence threshold: curriculum does not advance until the model reaches the specified metric level.

```python
at_threshold = (sign * val) >= (sign * threshold)
accept = avg_speed > 0.0 and at_threshold
```

**Effect**: The model must show Acc ≥ threshold at the current difficulty level before advancing to the next.

**When to use**: the primary mode for meaningful curriculum — eliminates the "proportional time advance" effect.

```yaml
curriculum:
  schedule: 2e+5
  mode: threshold
  key: Acc
  minimization: false
  threshold: 0.75
```

If `threshold` is too high and curriculum stalls — lower to 0.65. For difficult models, start at 0.60.

---

### `mode: plateau`

Advances curriculum only when the metric has stabilized near the threshold value. Instead of instant speed comparison, tracks the history of the last N checks.

```python
# history: last plateau_len metric values
std = sqrt(mean((x - mean(history))^2 for x in history))
is_plateau = std < plateau_tol
accept = is_plateau and (sign * val >= sign * threshold)
```

**Effect**: Advance happens only when the model operates stably at the current level (not just "improving on average"). Eliminates random metric spikes.

**When to use**: with unstable training or when it's important to confirm that progress is not accidental.

```yaml
curriculum:
  schedule: 2e+5
  mode: plateau
  key: Acc
  minimization: false
  threshold: 0.72
  plateau_len: 5      # number of checks in history (5 × 200k = 1M steps)
  plateau_tol: 0.02   # acceptable std (2% absolute Acc units)
```

Note: the first `plateau_len - 2` checks always return False (history not yet accumulated).

---

### `mode: multiaxis`

Each difficulty axis advances independently based on its own accuracy metric:

| Axis | Metric | Default threshold |
|---|---|---|
| `T` (sequence length) | `Acc` | 0.75 |
| `p_store` | `Acc/store` | 0.80 |
| `p_query` | `Acc/query` | 0.70 |

```python
axes = {
    'T':       metrics['Acc']       >= t_threshold,
    'p_store': metrics['Acc/store'] >= store_threshold,
    'p_query': metrics['Acc/query'] >= query_threshold,
}
```

**Effect**: If the model handles store well but query poorly — only p_store increases. Allows targeted pressure on the weak point.

**When to use**: when analyzing exactly where the model underperforms; when axes have different training dynamics.

```yaml
curriculum:
  schedule: 2e+5
  mode: multiaxis
  key: Acc            # used only for adaptive schedule tempo
  minimization: false
  t_threshold: 0.75
  store_threshold: 0.80
  query_threshold: 0.70
```

**Note**: the `key` parameter in multiaxis mode is still used for adaptive check frequency (adaptive schedule tempo), but does not affect the axis advance decision.

---

## Mode comparison

| Mode | Advance condition | Requires threshold | Per-axis |
|---|---|---|---|
| `speed` | avg speed > 0 | no | no |
| `threshold` | speed > 0 + metric ≥ threshold | yes | no |
| `plateau` | stability + metric ≥ threshold | yes | no |
| `multiaxis` | per-axis accuracy threshold | yes | yes |

## Experiment results (SDQ, grnn, 42M steps)

Three modes were compared on the `grnn` model in the SDQ experiment (hard: `count_queried=True`, `count_stored=True`, `n_keys=5`, `n_vals=10`). All runs for 42M steps, 64 envs, `schedule=2e5`.

| Mode | Acc ↑ | Acc/query ↑ | Acc/distract ↑ | Loss ↓ | curr_step |
|---|---|---|---|---|---|
| `speed` | 0.677 | 0.406 | 0.945 | 0.886 | 58 |
| `threshold` (0.75) | 0.786 | 0.543 | 0.965 | 0.639 | 18 |
| `plateau` (0.72, len=5) | **0.790** | **0.596** | 0.943 | **0.536** | 27 |

**Observations:**

- `speed` produces the highest number of advance steps (58) — curriculum ticks almost mechanically as Loss decreases, without guaranteeing real level mastery. Final Acc (0.677) is noticeably lower.
- `threshold` at 0.75 takes only 18 steps and achieves Acc 0.786 — the model reliably masters each level.
- `plateau` at threshold 0.72 and `plateau_len=5` shows the best final Acc (0.790) and Acc/query (0.596) with a moderate step count (27). Stabilization before advancing helps achieve deeper mastery of the current difficulty level.

**Conclusion:** `plateau` is the preferred mode for meaningful curriculum with unstable training. `threshold` is easier to tune and gives close results. `speed` is suitable only for quick smoke tests.

---

## Code

`knitwork/common/curriculum.py` — class `CurriculumScheduler`
