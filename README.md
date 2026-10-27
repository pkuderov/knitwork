# knitwork

Research project for Grid RNN experiments — associative memory benchmarks and language modeling.

## Setup

```sh
uv sync
```

## Running experiments

All scripts use `uv run <script> <config> [overrides]`. Overrides use dot-notation: `key=value` or `--key=value`.

### SDQ (Store-Distract-Query associative memory)

```sh
uv run knitwork/exps/sdq/run_sdq.py knitwork/exps/sdq/config/extend_config.yaml --model=grnn
uv run knitwork/exps/sdq/run_sdq.py knitwork/exps/sdq/config/extend_config.yaml --model=grnn --name="my run"
uv run knitwork/exps/sdq/run_sdq.py knitwork/exps/sdq/config/extend_config.yaml --model=grnn_fusion --log.enabled=false
```

Available models: `rnn`, `grnn`, `grnn_err`, `hgrnn`, `grnn_fw`, `grnn_reservoir`, `grnn_hgrn`, `grnn2`, `grnn_engram`, `grnn_loss`, `grnn_eq`, `grnn_eq1`, `grnn_disc`, `grnn_adv_loss`, `grnn_fusion`, `grnn_lru`, `grnn_lru_wide`, `grnn_lru_hop`

### Text (shakespeare / text8 language modeling)

```sh
uv run knitwork/exps/text/run_text.py knitwork/exps/text/config/extend_config.yaml --model=grnn
uv run knitwork/exps/text/run_text.py knitwork/exps/text/config/extend_config.yaml --model=hgrnn --name="hgrnn text8 ~78K"
```

Available models: `rnn`, `grnn`, `grnn_err`, `hgrnn`, `grnn2`, `grnn_loss`, `grnn_res`, `grnn_engram`

### POPGym memory tasks (RL, fast GPU pipeline)

Recurrent PPO with GPU-vectorized POPGym card tasks and CUDA graphs (10k-36k env steps/s). Full guide:
[docs/experiments/mikasa.md](docs/experiments/mikasa.md); results: [docs/experiments/rl_2026-10.md](docs/experiments/rl_2026-10.md).

```sh
CFG=knitwork/exps/mikasa/config/fast/base.yaml

# one run (Comet project knitwork-aistat; checkpoint runs/rl_fast/<run_id>.pt, rerun with the same --run_id resumes)
uv run python -m knitwork.exps.mikasa.run_fast $CFG --model=lru.L2 --env=AutoencodeEasy --n_steps=20e6 --name=my/run

# best Grid-LRU configuration (1.01M params)
uv run python -m knitwork.exps.mikasa.run_fast $CFG --model=grnn_lru.L2C4 \
  --grnn_lru_L2C4.n_layers=1 --grnn_lru_L2C4.hidden_size=224 --grnn_lru_L2C4.routing=dynamic \
  --grnn_lru_L2C4.x_add=true --grnn_lru_L2C4.column_horizons=true --env=AutoencodeEasy --n_steps=20e6

# supervised oracle (memory capacity without the RL signal) and curriculum transfer
uv run python -m knitwork.exps.mikasa.run_fast $CFG --model=lru.L2 --env=AutoencodeHard --n_steps=5e6 --rl.supervised=true
uv run python -m knitwork.exps.mikasa.run_fast $CFG --model=lru.L2 --env=AutoencodeMedium --init_from=runs/rl_fast/<easy_run>.pt

# smoke test
uv run python -m knitwork.exps.mikasa.run_fast $CFG --model=lru.L2 --env=AutoencodeEasy --n_steps=2e5 --log.enabled=false

# queue of runs (one "<run_id> <overrides...>" per line), pause and resume
bash knitwork/exps/mikasa/run_fast_queue.sh knitwork/exps/mikasa/config/fast/queue_p25_arch.txt   # logs/run_fast/<run_id>.log
bash knitwork/exps/mikasa/run_fast_pause.sh   # checkpoint and stop; rerun the same queue to resume
```

GPU tasks: `RepeatPrevious*`, `RepeatFirst*`, `Autoencode*`, `CountRecall*` (`Easy`/`Medium`/`Hard`); any other POPGym id
(e.g. `MineSweeperEasy`) runs on the CPU through `PopgymBatchEnv`. Models: any core from `knitwork/models/utils.py` with
a section in the config (`rnn.L2`, `lru.L2`, `grnn.L2C4`, `grnn_lru.L2C4`, `grnn_bank.L2C4`, `grnn_many.C16`).
Printed metrics: `Acc` (sampled-action accuracy), `EpRet` (train return), `Eval` (greedy eval return, max 1.0).
Tests: `uv run --with pytest python -m pytest tests/test_popgym_torch.py tests/test_ppo_gpu.py`.

### TreasureHunt (RL with PPO)

```sh
uv run knitwork/exps/treasure/run_treasure_hunt.py knitwork/exps/treasure/config_treasure_hunt.yaml --model=grnn
uv run knitwork/exps/treasure/run_treasure_hunt.py knitwork/exps/treasure/config_treasure_hunt.yaml --model=grnn_lru
```

Available models: `rnn`, `grnn`, `grnn_lru`, `grnn_lru_wide`

## Common overrides

```sh
--model=<name>          # select model
--name="<run name>"     # Comet run name
--device=cuda|cpu       # default: cuda
--n_steps=1e9
--n_envs=128
--seed=42
--log.enabled=false     # disable Comet logging
--visualize=false       # disable visualizations
```

## Count model parameters

```sh
uv run python -m knitwork.common.count_params --model grnn --input_size 27 --output_size 27
```

## Methods documentation

Docs are in `docs/methods/` and served via GitHub Pages (Docsify).  
To view locally: open `docs/index.html` in a browser or run `npx serve docs`.

## Project structure

```text
knitwork/
  common/         # shared utilities: config, logging, scheduler, tracker, entrypoint
  gens/           # data generators: sdq.py, text.py, periodic.py
  env/            # RL environments: popgym_torch.py (GPU POPGym), popgym_batched.py, treasure_hunt.py
  rl/             # ppo_gpu.py: recurrent PPO with CUDA graphs
  models/         # model implementations
  exps/
    sdq/          # Store-Distract-Query: run_sdq.py + _viz.py
    text/         # text8/shakespeare: run_text.py
    mikasa/       # POPGym RL: run_fast.py, config/fast/, queue scripts
    treasure/     # TreasureHunt PPO RL: run_treasure_hunt.py
  visualization/  # CKA, attention flow
docs/
  methods/        # model documentation (.md per model)
  index.html      # Docsify GitHub Pages site
  _sidebar.md     # navigation
```
