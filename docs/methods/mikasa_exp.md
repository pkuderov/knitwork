# MIKASA / POPGym — Memory Benchmark for RL

## Description

POPGym (Partially Observable Process Gym, Morad et al. 2023) and MIKASA-Base (Cherepanov et al. 2025) are diagnostic benchmarks for evaluating the memory capabilities of recurrent agents in partially observable environments. Unlike SDQ (synthetic associative memory test) and TreasureHunt (navigation with memory), these environments cover four canonical memory types, each in isolation:

| Memory type | What is tested | Example environments |
|---|---|---|
| **Object** | Storing object properties (object permanence) | RepeatFirst, HigherLower, Passive-T-Maze |
| **Sequential** | Order of events over time | RepeatPrevious, Autoencode |
| **Capacity** | Working memory capacity (how many items are held simultaneously) | MultiarmedBandit, CountRecall |
| **Spatial** | Spatial layout and navigation maps | Battleship, MineSweeper |

## Environments used

The experiment uses POPGym environments with discrete observations and discrete actions — compatible with existing knitwork models without any changes.

**Object memory:**
- `popgym-RepeatFirstEasy/Medium/Hard-v0` — N symbols are shown, the agent must reproduce the first one
- `popgym-HigherLowerEasy/Medium/Hard-v0` — guess whether the next card will be higher or lower than the current

**Object + Sequential memory:**
- `popgym-RepeatPreviousEasy/Medium/Hard-v0` — reproduce the symbol shown k steps ago

**Object + Capacity:**
- `popgym-MultiarmedBanditEasy/Medium/Hard-v0` — multi-armed bandit, agent memorizes arm rewards

**Continuous observations** (require wrappers from `rl_wrappers.py`):
- `popgym-CountRecallEasy-v0` — counting occurrences; `MultiDiscrete([2, 2])` → linear encoder
- `popgym-AutoencodeEasy-v0` — reproduce a sequence; `Tuple` obs → linear encoder

## Key mechanism

For environments with discrete observation `Discrete(n)` — the token is passed directly to the existing model:

```python
obs_in = to_torch(raw['obs'], device).view(-1, 1)   # [B, 1] int64
rnn_state = rnn.reset_state(rnn_state, reset_mask)
logits, rnn_state = rnn(obs_in, rnn_state)          # standard call
```

For environments with continuous observations, wrappers from `knitwork/models/rl_wrappers.py` are used, replacing `nn.Embedding` with `nn.Linear(obs_dim, embedding_size)`:

```python
# GridRnnContinuous — inherits GridRnn, overrides forward
x = self.embedding(obs.float())   # [B, obs_dim] → [B, embedding_size]
h, extras = self.grid_step_postmsg(x, h=h)
```

Training is performed with PPO + GAE (γ=0.99, λ=0.95, 4 epochs, clip=0.2).

## Hyperparameters

Non-standard parameters:
- `rollout_len: 32` — longer than in TreasureHunt (16), since POPGym episodes are longer
- `n_envs: 64` — fewer than in TreasureHunt (128), since SyncVectorEnv is slower than GPU environments
- `env:` — gym environment identifier, switched via `--env=popgym-...-v0`

## Experiment results

### Iteration 1 — June 2026 (in progress)

Models `grnn_ema_mem`, `grnn_delta`, `hgrnn_lru` were tested on two environments: RepeatFirstEasy (Object memory) and HigherLowerEasy (Sequential memory). 200M PPO steps, 64 envs.

| Model | Environment | EpRet | Progress | FPS | Status |
|---|---|---|---|---|---|
| grnn_ema_mem | RepeatFirstEasy | **~0.95** | 74M/200M | ~1000 | running |
| grnn_ema_mem | HigherLowerEasy | ~0.41 | 73M/200M | ~1000 | running |
| grnn_delta | RepeatFirstEasy | -0.48 | 22M/200M | ~586 | running |
| grnn_delta | HigherLowerEasy | — | pending | — | pending |
| hgrnn_lru | RepeatFirstEasy | — | pending | — | pending |
| hgrnn_lru | HigherLowerEasy | — | pending | — | pending |

**Observations:**
- `grnn_ema_mem` — strong model for Object memory: EpRet ~0.95 already at 37% of training on RepeatFirst. Surprise-EMA effectively separates "important" tokens from routine ones. On HigherLower (Sequential) the result is moderate (~0.41).
- `grnn_delta` — substantially slower (~1.7× in FPS) due to dual-scale delta memory. At 11% of training EpRet is negative — model is not learning or requires significantly more warmup steps.
- `hgrnn_lru` — awaiting launch; showed the best result on SDQ among all models (Acc=0.967).

**Bugs fixed at launch:**
- `grnn_delta` was missing from the `run_mikasa.py` registry and config — added manually.
- `hgrnn_lru.reset_state` crashed with `TypeError` on float `reset_mask` — fixed by adding `.bool()` cast.

### Iteration 2 — June 2026 (in progress, 4M/200M, ~2%)

Testing `grnn_lru`, `hgrnn_lru` on RepeatFirstEasy and RepeatPreviousEasy.
In queue: `grnn_engram`, `grnn_fw` on RepeatFirstEasy.

| # | Model | Environment | EpRet @ stop | H | FPS | Result |
|---|--------|-------|--------------|---|-----|------|
| 21 | grnn_lru | RepeatFirstEasy | ~−0.4 @ 15M | 1.07 | 338 | stopped, no progress |
| 22 | grnn_lru | RepeatPreviousEasy | −0.246 @ 15M | 0.64 | 337 | stopped, entropy collapse |
| 23 | hgrnn_lru | RepeatFirstEasy | ~−0.5 @ 15M | 1.18 | 338 | stopped, no progress |
| 24 | hgrnn_lru | RepeatPreviousEasy | — | — | — | cancelled (pending) |
| 25 | grnn_engram | RepeatFirstEasy | — | — | — | pending |
| 26 | grnn_fw | RepeatFirstEasy | — | — | — | pending |

**Observations:**
- No model achieved positive EpRet within 7.5% of training on RepeatFirst. This is expected — Object memory requires 20-30M+ steps.
- `grnn_lru` on RepeatPrevious: early entropy collapse (H: 1.1 → 0.19 → 0.64). LRU without entropy regularization quickly determinizes at the first success — irreversibly.
- `hgrnn_lru`: highest H=1.18 among all (Hopfield attention slows determinization), but also the highest oscillation amplitude — unstable training.
- All three runs stopped early — results are non-final.

## Running

```sh
# Object memory — RepeatFirst (best starting test)
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatFirstEasy-v0 --model=grnn

# Sequential — RepeatPrevious
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatPreviousEasy-v0 --model=grnn_lru

# Capacity — MultiarmedBandit
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-MultiarmedBanditEasy-v0 --model=hgrnn

# Smoke-test (no AIM, 100k steps)
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatFirstEasy-v0 --model=grnn --n_steps=1e5 --log.enabled=false
```
