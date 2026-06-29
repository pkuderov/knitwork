# TreasureHunt

## Task formulation

TreasureHunt is an RL benchmark for evaluating associative memory in recurrent agents under partial observability. The agent operates in a 2D N×N grid with pairs of objects — keys and their corresponding doors, each of `n_colors` colors. To open a door of color c, the agent must first pick up the key of the same color c. The observation space is fundamentally limited: the agent only sees the central cell of its position.

The task requires multi-step associative memory: the agent must remember the binding (color → key location) and, upon finding a door, retrieve that memory for navigation back. The further the key is from the door and the more pairs there are, the longer the association must be held in hidden state.

## Environment

### Observation

All observable information is encoded into a single integer token:

```
token = central_cell_type × 2^n_colors + inventory_bitmask
```

- `central_cell_type` ∈ {empty, wall, key_0..7, door_0..7, agent} — type of cell under the agent
- `inventory_bitmask` — bitmask of collected keys (2^n_colors variants)

Input vocabulary size: `VOCAB_SIZE × 2^n_colors` = 19 × 2^n_colors tokens.

### Action space

Four discrete actions: UP (0), DOWN (1), LEFT (2), RIGHT (3).

### Reward structure

| Event | Reward |
|---|---|
| Opening a door | +1.0 |
| Each step | −0.01 (time penalty) |
| Wall collision | — (action is ignored) |

Episode ends when all doors are opened or `max_steps` is reached.

### Difficulty levels

| Mode | Grid | Pairs | max\_steps | min\_dist key–door | Memory load |
|---|---|---|---|---|---|
| **easy** | 8×8 | 2 | 100 | 2 | minimal |
| **medium** | 10×10 | 3 | 150 | 4 | moderate |
| **hard** | 12×12 | 4 | 200 | 6 | high |
| **nightmare** | 15×15 | 6 | 400 | 8 | extreme |

## Training algorithm

Training uses **PPO (Proximal Policy Optimization)** with **GAE (Generalized Advantage Estimation)**.

### Actor-critic architecture

- **Actor** — recurrent model (GridRNN and variants); takes observation token, returns logits over action space and updated hidden state.
- **Critic** — separate linear layer `nn.Linear(actor_hidden, 1)` on top of the upper hidden state layer of the actor. This is an intentional asymmetry: the critic has no own recurrent state and uses actor representations as features.

Feature extraction from heterogeneous states:

```python
def extract_h_top(state) -> torch.Tensor:
    h = state[0] if isinstance(state, tuple) else state
    if h.ndim == 3:              # GRU: [layers, B, H]
        return h[-1][:, :actor_hidden]
    return h[-1, 0, :, :actor_hidden]   # Grid: [layers, cols, B, H]
```

### PPO hyperparameters

| Parameter | Value |
|---|---|
| γ (discount) | 0.99 |
| λ (GAE lambda) | 0.95 |
| clip\_eps | 0.2 |
| value\_coef | 0.5 |
| entropy\_coef | 0.01 |
| max\_grad\_norm | 0.5 |
| PPO epochs | 4 |

### Trajectory collection

At each iteration a rollout of length `rollout_len = 16` steps is collected in `n_envs = 128` parallel environments (2048 transitions). The actor hidden state is preserved between iterations in TBPTT style: after PPO update the state is detached from the computation graph (`detach_state`).

### GAE

```
δ_t = r_t + γ · V(s_{t+1}) · (1−done_t) − V(s_t)
A_t = δ_t + γ · λ · (1−done_t) · A_{t+1}
```

Advantages are normalized per batch: `(A - mean) / (std + ε)`.

### LR schedule

Linear warmup (100 steps) from initial 1e-5·lr to target lr=8e-4, then exponential decay to 5e-3·lr.

## Metrics

| Metric | Description |
|---|---|
| **MeanReward** | Mean reward per step; becomes positive when doors are systematically opened |
| **OpenedFrac** | Mean fraction of doors opened per episode ∈ [0, 1]; main quality metric |
| **PolicyLoss** | PPO clipped surrogate loss |
| **ValueLoss** | MSE between predicted and actual return |
| **Entropy** | Policy entropy; low → deterministic policy |

## What counts as a good result

| Mode | OpenedFrac | Interpretation |
|---|---|---|
| easy | > 0.90 | agent reliably solves the task |
| easy | < 0.50 | model cannot hold the association |
| medium | > 0.70 | convincing result |
| hard | > 0.50 | significant; requires holding 4 associations |
| nightmare | > 0.30 | evidence of long-term associative memory |

Baseline GRU on easy typically achieves OpenedFrac ~ 0.3–0.5: without an explicit associative memory mechanism the model guesses rather than remembers. Positive MeanReward at −0.01/step penalty means at least one door reliably opened per episode.

## Running

```sh
# easy mode, grnn_lru model (default)
uv run knitwork/exps/treasure/run_treasure_hunt.py \
    knitwork/exps/treasure/config_treasure_hunt.yaml --model=grnn_lru

# hard mode, grnn model
uv run knitwork/exps/treasure/run_treasure_hunt.py \
    knitwork/exps/treasure/config_treasure_hunt.yaml --model=grnn --env=hard
```

## Logging

AIM project: `grid-rnn-treasure`. Main metrics: `MeanReward`, `OpenedFrac`, `PolicyLoss`, `ValueLoss`, `Entropy`, `env/mean_opened`.
