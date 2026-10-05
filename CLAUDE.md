# knitwork

Research project for Grid RNN experiments — associative memory benchmarks and language modeling.

## Package manager

Always use `uv`. Never use `pip` or `python` directly.

```sh
uv run <script>       # run script
uv run python -m <module>  # run module
uv sync               # install/update dependencies
```

## Project structure

```
knitwork/
  common/         # shared utilities: config, logging, scheduler, tracker, entrypoint
  gens/           # data generators: sdq.py, text.py, periodic.py
  env/            # RL environments: treasure_hunt.py
  models/         # model implementations (see Models section)
  exps/
    sdq/          # Store-Distract-Query experiment
    text/         # text8/shakespeare language modeling
    treasure/     # TreasureHunt RL benchmark
  visualization/  # CKA, attention flow
```

## Experiment constraints

- Always run on **CUDA**: pass `--device=cuda` or ensure `device: cuda` in config (it is the default)
- No fixed limit on simultaneous experiments on the server: `server/queue_daemon.py` keeps GPU memory use under 79 GiB (`--mem_cap_gb`) and never starts jobs while a foreign process is on the GPU

## Running experiments

All scripts use `uv run <script> <config> [overrides]`. Config overrides use `key=value` or `--key=value` dot-notation.

```sh
# SDQ (Store-Distract-Query) — unified, all models
uv run knitwork/exps/sdq/run_sdq.py knitwork/exps/sdq/config/extend_config.yaml --model=grnn
uv run knitwork/exps/sdq/run_sdq.py knitwork/exps/sdq/config/extend_config.yaml --model=grnn --name="my run"

# Text (shakespeare / text8) — unified, all models
uv run knitwork/exps/text/run_text.py knitwork/exps/text/config/extend_config.yaml --model=grnn
uv run knitwork/exps/text/run_text.py knitwork/exps/text/config/extend_config.yaml --model=hgrnn --name="hgrnn text8 ~78K"

# TreasureHunt (PPO RL) — all models
uv run knitwork/exps/treasure/run_treasure_hunt.py knitwork/exps/treasure/config_treasure_hunt.yaml --model=grnn_lru

# Count model parameters
uv run python -m knitwork.common.count_params --model grnn --input_size 27 --output_size 27
```

### Common config overrides

```sh
--model=<name>          # select model (see Models section)
--name="<run name>"     # Comet run name
--device=cuda|cpu
--n_steps=1e9
--n_envs=128
--seed=42
--log.enabled=false     # disable Comet logging
```

## Models

All models are configured in the `models:` section of the config file and selected via `--model=<name>`.

| Model | Description |
|---|---|
| `rnn` / `gru` | GRU baseline |
| `grnn` | Grid RNN (base) |
| `grnn2` | Grid RNN v2 with time gate and VAE latent |
| `grnn_err` | Grid RNN with error signal |
| `grnn_eq` | Grid RNN with equilibrium iterations |
| `grnn_lru` | Grid RNN with Linear Recurrent Units |
| `grnn_lru_wide` | Wide LRU variant |
| `hgrnn` | Hierarchical Grid RNN |
| `hgrnn_lru` | Hierarchical Grid RNN + LRU |
| `hgrn_grnn` | HGRN cell in Grid RNN |
| `grnn_fw` | Grid RNN with Fast Weights |
| `grnn_reservoir` | Grid RNN with frozen reservoir columns |
| `grnn_fusion` | Grid RNN with HGRN + reservoir + cross-attention + diversity loss |
| `grnn_engram` | Grid RNN with Hebbian engram memory slots |
| `grnn_loss` | Grid RNN with auxiliary losses |
| `grnn_disc` | Grid RNN discriminator variant |
| `grnn_adv_loss` | Grid RNN with adversarial loss |
| `engram_grnn` | Engram-based Grid RNN |

## Methods documentation

For every model file in `knitwork/models/` there must be a corresponding `.md` file in `docs/methods/` with a brief explanation of the approach and short code excerpts.

Structure:
```
docs/
  methods/
    grnn.md
    grnn_lru.md
    ...
  index.html    # Docsify site (GitHub Pages)
  _sidebar.md   # navigation with categories
```

Each `docs/methods/<name>.md` should contain:
1. **One-paragraph summary** — what problem the method solves and the core idea
2. **Key mechanism** — the most important part of the implementation with a short inline code snippet
3. **Hyperparameters** — the non-obvious ones worth noting
4. Write in English. (The 37 pre-existing `docs/methods/*.md` written before this rule remain in Russian and are not retro-translated; all new docs are English.)

When adding a new model file, always create the corresponding `docs/methods/` doc alongside it.

## Remote experiment server

GPU-сервер — **только `aicenter3`, только H100 (GPU 3)**. Подробности
подключения, ограничений и типичных проблем — в `setup.md` в корне
репозитория, не дублируются здесь. Короткая выжимка:

```sh
ssh aicenter3 'nvidia-smi -i 3 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv'
ssh aicenter3 'cd /storage/annenkov_vd/knitwork && CUDA_VISIBLE_DEVICES=3 python <script>'
```

**Разрешена только GPU 3**, остальные пять H100 на сервере — не трогать.
GPU общая (не выделена под проект) — перед запуском всегда проверять
занятость, не запускать задачу и не завершать чужие процессы, если карта
занята. Нет очереди/daemon'а — планирование параллельных запусков вручную,
по одной команде за раз (см. `setup.md`).

Любые упоминания прежнего сервера (`knitwork-server`, RTX 3050, демон
`knitwork-queue`, AIM-логирование) в истории документации проекта —
устарели и не описывают текущую инфраструктуру; не использовать как
инструкцию.

## Experiment tracking (Comet ML)

Experiments are logged to **Comet ML only** (`log.logger: comet` in
config), workspace `team-rl-exp`. No other logger (AIM, wandb, tensorboard)
is used for this project going forward — historical runs logged to AIM
before the switch to Comet remain in place and are not deleted; they are
simply not the source for current/future comparisons. Active projects:
- `knitwork-sdq` — SDQ experiments
- `knitwork-text` / `knitwork-text-debug` — text8/Shakespeare experiments
- `knitwork-mikasa` — MIKASA/POPGym RL experiments (separate research thread)
- `knitwork` — earlier architecture-search runs (pre-dates the final paper
  configs; useful for history, not for current comparisons)



## Code style

Follow the style established in `knitwork/models/grnn.py`:

- Keep code **short and direct** — no unnecessary abstractions or wrapper layers
- No emojis anywhere in code or comments
- Comments only where non-obvious; write them **in English**, briefly
- Use comments to annotate **tensor shapes**, e.g. `# [B, T, H]`
- No multi-line docstrings; one short line maximum if needed
- Keyword-only arguments (`*,`) for constructors with many params (see `GridRnn.__init__`)

## Git

- Commit author: **Vladimir <aberay89@bk.ru>**
- Remotes: `origin` → GitHub (`github.com/pkuderov/knitwork`), `gitea` → self-hosted
- Branch: **main**

```sh
git commit --author="Vladimir <aberay89@bk.ru>" -m "..."
```

**Make and push commits only after explicit user permission.**

## Security

### Protected files — do not modify

The following core model files define the foundational architecture and must not be changed without explicit instruction:

```
knitwork/models/grnn.py       # GridRNN base — reference implementation
knitwork/models/grnn_err.py   # GridRNN with error signal
knitwork/models/gru.py        # GRU baseline
knitwork/config/base.yaml
```

### Protected directories — do not modify or delete

```
.env      # environment variables and secrets
.aim/     # AIM experiment database — modification corrupts run history
```
