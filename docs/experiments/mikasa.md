# POPGym RL pipeline (run_fast)

The only RL pipeline for the POPGym / MIKASA memory tasks is the GPU pipeline `knitwork/exps/mikasa/run_fast.py`. It
replaced the earlier gymnasium-based runners (`run.py`, `run_mikasa.py`, `knitwork/rl/on_policy.py` and the
off-policy code). The new pipeline is 100 to 250 times faster (130 vs 10k-36k environment steps per second) and its card
tasks are tested to match the reference `popgym` implementation. Results: [RL experiments](experiments/rl_2026-10.md).

## Components

| File | Role |
|---|---|
| `knitwork/env/popgym_torch.py` | `CardMemoryEnv`: POPGym card-memory tasks as batched torch environments on the GPU, plus the deck-length curriculum |
| `knitwork/env/popgym_batched.py` | `PopgymBatchEnv`: original POPGym environments stepped on the CPU in a batch (MineSweeper, Concentration, Battleship, ...) |
| `knitwork/rl/ppo_gpu.py` | recurrent PPO: rollout collection, GAE, update, evaluation; CUDA graphs for the policy step and the minibatch replay |
| `knitwork/models/wrappers.py` | `RLVectorModel`: observation encoder + any recurrent core + policy/value heads |
| `knitwork/exps/mikasa/run_fast.py` | runner: config, checkpoints, resume, Comet logging |
| `knitwork/exps/mikasa/config/fast/base.yaml` | defaults and the ~1M-parameter model configs |
| `knitwork/exps/mikasa/run_fast_queue.sh`, `run_fast_pause.sh` | sequential queues with resource caps; pause and resume |

## Tasks

GPU (`CardMemoryEnv`): `RepeatPrevious{Easy,Medium,Hard}`, `RepeatFirst{Easy,Medium,Hard}`,
`Autoencode{Easy,Medium,Hard}`, `CountRecall{Easy,Medium,Hard}`. Any other POPGym id (for example `MineSweeperEasy`,
`ConcentrationEasy`, `BattleshipEasy`) runs through `PopgymBatchEnv` on the CPU.

The observation is a one-hot of the card plus the previous action and reward sign (`env_cfg.prev_action_reward`).

## Running

```sh
# one run; Comet project knitwork-aistat, checkpoint in runs/rl_fast/<run_id>.pt
uv run python -m knitwork.exps.mikasa.run_fast knitwork/exps/mikasa/config/fast/base.yaml \
  --model=lru.L2 --env=AutoencodeEasy --n_steps=20e6 --name=my/run

# best grid configuration from the October study (1.01M params)
uv run python -m knitwork.exps.mikasa.run_fast knitwork/exps/mikasa/config/fast/base.yaml \
  --model=grnn_lru.L2C4 --grnn_lru_L2C4.n_layers=1 --grnn_lru_L2C4.hidden_size=224 \
  --grnn_lru_L2C4.routing=dynamic --grnn_lru_L2C4.x_add=true --grnn_lru_L2C4.column_horizons=true \
  --env=AutoencodeEasy --n_steps=20e6

# smoke test without logging
uv run python -m knitwork.exps.mikasa.run_fast knitwork/exps/mikasa/config/fast/base.yaml \
  --model=lru.L2 --env=AutoencodeEasy --n_steps=2e5 --log.enabled=false
```

Model keys are `<registry name>.<config suffix>`: `--model=grnn_lru.L2C4` reads the `grnn_lru_L2C4:` section, and its
fields are overridden as `--grnn_lru_L2C4.<field>=...`. The base config has `rnn.L2`, `lru.L2`, `grnn.L2C4`,
`grnn_lru.L2C4`, `grnn_bank.L2C4`, `grnn_many.C16` and others, all sized to about 1M parameters with the heads.

### Three protocols

```sh
B="uv run python -m knitwork.exps.mikasa.run_fast knitwork/exps/mikasa/config/fast/base.yaml --model=lru.L2"
$B --env=AutoencodeEasy --n_steps=20e6 --run_id=ae_easy                       # 1. RL
$B --env=AutoencodeHard --n_steps=5e6 --rl.supervised=true                    # 2. oracle: supervised on the correct action
$B --env=AutoencodeMedium --n_steps=20e6 --init_from=runs/rl_fast/ae_easy.pt  # 3. transfer from the Easy checkpoint
```

The oracle (`rl.supervised=true`) trains the same network with cross-entropy on the rewarded action. It separates memory
capacity from the RL signal: if the oracle learns and PPO does not, the problem is credit assignment, not memory.

### Queues, pause and resume

```sh
# each line: <run_id> <overrides...>; logs go to logs/run_fast/<run_id>.log
bash knitwork/exps/mikasa/run_fast_queue.sh knitwork/exps/mikasa/config/fast/queue_p25_arch.txt
bash knitwork/exps/mikasa/run_fast_pause.sh      # SIGTERM: checkpoint at the end of the update, exit code 3
bash knitwork/exps/mikasa/run_fast_queue.sh <same list>   # resumes unfinished runs, skips finished ones
```

The queue runs each job under `systemd-run` caps (`RUN_MEM`, default 3G; `RUN_CPU`, default 200%). `gpu_mem_frac`
(default 0.3) caps the GPU memory per run; long BPTT, snapshot memory and many-column models need 0.45.

## Main options

| Option | Default | Meaning |
|---|---|---|
| `n_envs`, `rollout_len` | 256, 128 | parallel environments and BPTT window |
| `n_steps` | 20e6 | environment steps |
| `rl.lr`, `rl.epochs`, `rl.minibatches` | 3e-4, 4, 4 | PPO optimizer |
| `rl.gamma`, `rl.gae_lambda`, `rl.clip_eps`, `rl.entropy_coef` | 0.99, 0.95, 0.2, 0.01 | PPO loss |
| `rl.supervised` | false | oracle mode |
| `rl.aux_coef` + `rl_wrapper.aux_lags` | 0, [] | auxiliary prediction of past observations |
| `rl.route_entropy_coef` | 0 | routing entropy term for grid cores |
| `rl_wrapper.readout` | core | `columns` feeds all top-layer grid columns to the heads |
| `rl_wrapper.policy` | categorical | `ordinal` for counting tasks |
| `train_env_cfg.min_cards` | null | Autoencode: deck length per episode drawn from `[min_cards, full]` |
| `train_env_cfg.grow_cards`, `grow_acc` | null | adaptive curriculum: grow the deck when training accuracy reaches `grow_acc` |
| `init_from` | null | warm start: model weights from another checkpoint |
| `cuda_graphs` | true | capture the policy step and the update replay |
| `eval.n_envs`, `eval.every` | 512, 20 | greedy evaluation on a separate env seed |

## Output

Every `print_every` updates:

```text
[19.04M] 13452 sps Acc 0.962 H 0.530 KL 0.0302 EV 0.98 EpRet 0.929 Eval 0.933
```

`Acc`: accuracy of the sampled action on rewarded steps. `H`: policy entropy. `KL`: approximate KL of the update.
`EV`: explained variance of the value head. `EpRet`: training episode return. `Eval`: greedy evaluation return (the
metric used in the docs, maximum 1.0, random about -0.5).

## Tests

```sh
uv run --with pytest python -m pytest tests/test_popgym_torch.py tests/test_ppo_gpu.py
```

`test_popgym_torch.py` checks the GPU tasks step by step against `popgym` and the env state round trip;
`test_ppo_gpu.py` checks GAE resets, the CUDA-graph policy against eager mode, the curriculum and the Grid-LRU RL options.
Run the graphed tests in separate processes if the whole file crashes with an illegal-address error (known issue when the
clockwork and memory-slot graph tests share one process).
