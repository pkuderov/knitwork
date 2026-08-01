# MIKASA / POPGym — Memory Benchmark

## Task formulation

POPGym (Partially Observable Process Gym, Morad et al. 2023) is a set of diagnostic environments for evaluating different memory types in recurrent RL agents. Unlike TreasureHunt (single task type) or SDQ (synthetic association test), POPGym covers four orthogonal memory aspects:

| Type | What is required | Example environments |
|---|---|---|
| **Object memory** | Retain properties of a specific object (object permanence) | RepeatFirst, HigherLower |
| **Sequential memory** | Remember the order of events | RepeatPrevious, Autoencode |
| **Capacity** | Hold multiple items simultaneously | MultiarmedBandit, CountRecall |
| **Spatial memory** | Navigate in space without a map | Battleship, MineSweeper |

All selected environments use **discrete actions** and **vector observations** — compatible with existing knitwork models without changes.

---

## Environments

### Object memory

**RepeatFirst** (`popgym-RepeatFirstEasy/Medium/Hard-v0`)

The agent receives a sequence of symbols from an alphabet of size N; at the end of the episode it must reproduce the **first** symbol of the sequence. Test for long-term storage of a single value.

- Obs: `Discrete(N)` → token `[B, 1]`
- Actions: `Discrete(N)`
- Difficulty Easy → Hard: sequence length increases

**HigherLower** (`popgym-HigherLowerEasy/Medium/Hard-v0`)

Card game: the agent sees a card and predicts whether the next one will be **higher** or **lower**. Requires holding in memory the distribution of cards already played.

- Obs: `Discrete(13)`, Actions: `Discrete(2)`

### Sequential memory

**RepeatPrevious** (`popgym-RepeatPreviousEasy/Medium/Hard-v0`)

At each step the agent sees a symbol and on the next step must reproduce it (i.e., always respond with a delay of k steps).

- Obs: `Discrete(N)`, Actions: `Discrete(N)`

### Capacity (working memory capacity)

**MultiarmedBandit** (`popgym-MultiarmedBanditEasy/Medium/Hard-v0`)

Multi-armed bandit: arm rewards are set randomly at the start of the episode. The agent must explore and memorize the reward of each arm.

- Obs: `Discrete(2)` (success/failure), Actions: `Discrete(K)`, where K = number of arms

**CountRecall** (`popgym-CountRecallEasy/Medium/Hard-v0`) — *continuous observations*

The agent counts symbol occurrences, then must reproduce the exact counter.

- Obs: `MultiDiscrete([2, 2])` → linear encoder `[B, 2] → [B, embedding_size]`
- Actions: `Discrete(26)`

### Spatial memory

**Battleship** (`popgym-BattleshipEasy/Medium/Hard-v0`) — *MultiDiscrete actions*

The agent sinks ships on a grid, receiving binary feedback (hit/miss). Requires a map of already explored cells.

- Obs: `Discrete(2)`, Actions: `MultiDiscrete([8, 8])` *(not yet supported)*

**MineSweeper** (`popgym-MineSweeperEasy/Medium/Hard-v0`) — *MultiDiscrete actions*

Classic minesweeper with partial visibility.

- Obs: `Discrete(3)`, Actions: `MultiDiscrete([4, 4])` *(not yet supported)*

> **Note:** environments with `MultiDiscrete` action space (Battleship, MineSweeper) require a separate action head and are not supported in the current implementation. Spatial memory is partially covered via HigherLower (remembering card distribution = implicit spatial statistics).

---

## Observation integration

### Discrete observations (`Discrete(n)`) — no model changes

```python
obs_in = to_torch(raw['obs'], device).view(-1, 1)   # [B, 1] int64
rnn_state = rnn.reset_state(rnn_state, reset_mask)
logits, rnn_state = rnn(obs_in, rnn_state)
```

The token is passed directly to `nn.Embedding(n_tokens, embedding_size)` of the model.

### Continuous observations (`MultiDiscrete` / `Tuple`) — wrappers from `rl_wrappers.py`

```python
obs_in = to_torch(raw['obs'], device).to(dtype)     # [B, obs_dim] float32
```

In `GridRnnContinuous` (and analogues) `nn.Embedding` is replaced with `nn.Linear(obs_dim, embedding_size)`:

```python
x = self.embedding(obs.float())   # [B, obs_dim] → [B, embedding_size]
h, extras = self.grid_step_postmsg(x, h=h)
```

The grid core (cells, attention, head) is inherited without changes.

---

## Training algorithm

**PPO + GAE** — identical to TreasureHunt:

| Parameter | Value |
|---|---|
| γ (discount) | 0.99 |
| λ (GAE lambda) | 0.95 |
| clip\_eps | 0.2 |
| value\_coef | 0.5 |
| entropy\_coef | 0.01 |
| max\_grad\_norm | 0.5 |
| PPO epochs | 4 |
| rollout\_len | 32 |
| n\_envs | 64 |
| LR | 8e-4 (warmup → decay) |

**Actor-critic:** recurrent model as actor + separate `nn.Linear(hidden, 1)` as critic on top of the upper hidden state layer.

---

## Metrics

| Metric | Description |
|---|---|
| **MeanReward** | Mean reward per step in the rollout |
| **ep\_return** | Total return per episode (EMA over the last n\_envs completed) |
| **ep\_length** | Mean length of completed episode |
| **PolicyLoss** | PPO clipped surrogate loss |
| **ValueLoss** | MSE critic |
| **Entropy** | Policy entropy |

The main metric is `ep_return`: how high a total reward the agent obtains per episode. For RepeatFirst this is proportional to the number of correct answers at the end.

---

## Running

```sh
# Object memory — RepeatFirst (starting test)
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatFirstEasy-v0 --model=grnn

# Sequential — RepeatPrevious
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatPreviousEasy-v0 --model=grnn_lru

# Capacity — MultiarmedBandit
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-MultiarmedBanditEasy-v0 --model=hgrnn

# Capacity with continuous obs — CountRecall (requires continuous wrapper)
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-CountRecallEasy-v0 --model=grnn

# Smoke-test (no AIM, 50k steps)
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatFirstEasy-v0 --model=grnn --n_steps=5e4 --log.enabled=false
```

### Running multiple environments for comparative analysis

```sh
# Parallel — no more than 3 simultaneously (project rule)
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatFirstEasy-v0 --model=grnn_lru --name="RepeatFirst grnn_lru" &

uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatPreviousEasy-v0 --model=grnn_lru --name="RepeatPrev grnn_lru" &

uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-MultiarmedBanditEasy-v0 --model=grnn_lru --name="Bandit grnn_lru" &
```

---

## Logging

AIM project: `grid-rnn-mikasa`. Default metrics: `MeanReward`, `ep_return`, `ep_length`, `PolicyLoss`, `ValueLoss`, `Entropy`, `fps`.

---

## Experiment results

### grnn_harmonic — MIKASA v6 (2026-06-17, running)

Model: `HarmonicGridRNN 3L×4C LRU + 0Res | hidden=128 heads=4 mem=(32×128) ema=0.9` (2.11M params)

Algorithm: PPO | n_envs=64 | rollout_len=32 | LR=8e-4 (warmup+cosine) | 200M steps

| # | Env | Progress | LR | PL | VL | H | R | EpRet | Status |
|---|---|---|---|---|---|---|---|---|---|
| #076 | RepeatFirstEasy | 20.5M/200M (10%) | 80% | -0.000 | 0.007 | 1.32 | -0.009 | -0.499 | RUNNING |
| #077 | RepeatFirstMedium | 20.5M/200M (10%) | 80% | -0.002 | 0.000 | 1.33 | -0.001 | -0.496 | RUNNING |
| #078 | MultiarmedBanditEasy | 20.0M/200M (10%) | 79% | -0.002 | 0.244 | **2.27** | **0.000** | **+0.021** | RUNNING |
| #079 | MultiarmedBanditMedium | — | — | — | — | — | — | — | PENDING |

**Observations (20M steps):**
- BanditEasy (#078) shows the best behavior: entropy 2.27 (active exploration), EpRet slightly positive (+0.021). Model is learning the exploration task.
- RepeatFirst Easy/Medium (#076, #077) are stuck at negative EpRet ≈ -0.5. Entropy is moderate (1.32–1.33). Long-term memory task — model has not learned it yet.

### grnn_harmonic — MIKASA v5 (2026-06-17, interrupted)

Progress to 29M/200M steps before interruption (exit=-15, SIGTERM):

| # | Env | Progress | H | R | EpRet |
|---|---|---|---|---|---|
| #074 | RepeatFirstEasy | 29M/200M | 0.78 | -0.009 | -0.555 |
| #075 | RepeatFirstMedium | 29M/200M | 0.22 | -0.001 | -0.529 |

**Observations v5:** entropy dropped sharply (0.78 and 0.22) — model collapses into a single strategy without learning the task. This motivated changes in v6 (entropy_coef, warmup, Hopfield gate init).
