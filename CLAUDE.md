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
- Run **no more than 3 experiments simultaneously**

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
--name="<run name>"     # AIM run name
--device=cuda|cpu
--n_steps=1e9
--n_envs=128
--seed=42
--log.enabled=false     # disable AIM logging
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

GPU-сервер с RTX 3050. Все эксперименты запускаются там.

### SSH подключение

```sh
# Локальная сеть (быстро, всегда)
ssh knitwork-server          # 192.168.1.58:2222

# Глобальная сеть (через WireGuard → playit.gg UDP)
ssh knitwork-server-global   # 10.8.0.1:2222
```

`~/.ssh/config` уже настроен, ключ: `~/.ssh/knitwork_server`.

**После перезагрузки клиента** — WireGuard поднимается автоматически (`systemctl enable wg-quick@wg-knitwork`). Если нет:
```sh
sudo wg-quick up wg-knitwork
```

**После перезагрузки сервера** — очередь и playit стартуют автоматически (systemd user services с lingering). Проверить:
```sh
ssh knitwork-server "systemctl --user status knitwork-queue playit"
```

### Синхронизация кода

Перед запуском экспериментов код нужно синхронизировать. Исключаются: `.aim/`, `.git/`, `docs/`, `article/`, `__pycache__/`, `.venv/`.

```sh
bash server/sync_to_server.sh
```

Или сразу sync + enqueue одной командой:
```sh
bash server/knitwork_run.sh <script> <config> [overrides] [-- --name "run name"]
```

### Запуск экспериментов (очередь)

На сервере работает демон `knitwork-queue.service` — запускает не более 3 экспериментов одновременно, остальные ждут в очереди.

```sh
# Добавить эксперимент в очередь (выполняется локально, запускает sync + enqueue):
bash server/knitwork_run.sh knitwork/exps/sdq/run_sdq.py \
    knitwork/exps/sdq/config/extend_config.yaml --model=grnn -- --name "grnn baseline"

# Добавить несколько — выполнятся по 3 параллельно:
bash server/knitwork_run.sh knitwork/exps/sdq/run_sdq.py \
    knitwork/exps/sdq/config/extend_config.yaml --model=grnn_lru -- --name "grnn_lru"
```

Напрямую на сервере (без sync):
```sh
ssh knitwork-server "cd ~/knitwork && PYTHON=/opt/uv-envs/knitwork/.venv/bin/python && \
    \$PYTHON server/enqueue.py 'КОМАНДА' --name='название'"
```

### Статус очереди и логи

```sh
# Статус: running / pending / completed
ssh knitwork-server "cd ~/knitwork && /opt/uv-envs/knitwork/.venv/bin/python server/queue_status.py"

# Лог конкретного эксперимента (ID из queue_status):
ssh knitwork-server "tail -50 ~/knitwork_logs/0001_имя.log"

# Сброс pending (отменить все ожидающие):
ssh knitwork-server "cd ~/knitwork && /opt/uv-envs/knitwork/.venv/bin/python server/enqueue.py --clear-pending"
```

### Результаты экспериментов (AIM)

AIM-база хранится на сервере в `~/.aim/` (корень home). Читать через SSH:

```sh
ssh knitwork-server "cd ~/knitwork && /opt/uv-envs/knitwork/.venv/bin/python server/query_results.py --last 10"
ssh knitwork-server "cd ~/knitwork && /opt/uv-envs/knitwork/.venv/bin/python server/query_results.py --model grnn --sort val_acc"
```

### Серверное окружение

| Что | Где |
|---|---|
| Python venv | `/opt/uv-envs/knitwork/.venv/` |
| Python binary | `/opt/uv-envs/knitwork/.venv/bin/python` |
| Проект | `~/knitwork/` |
| Логи экспериментов | `~/knitwork_logs/` |
| AIM база | `~/.aim/` |
| Очередь (JSON) | `~/knitwork_queue.json` |

Команды нужно запускать через полный путь к python вenv или с `LD_LIBRARY_PATH` (очередь прокидывает его автоматически через systemd-сервис).

### Частые ошибки и решения

**`ImportError: libcusparseLt.so.0`**
Torch не видит CUDA-библиотеки. Эксперименты через очередь (`queue_runner.py`) это исправляют автоматически — сервис экспортирует `LD_LIBRARY_PATH`. При ручном запуске:
```sh
export LD_LIBRARY_PATH=/opt/uv-envs/knitwork/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:...
```

**`CUDNN_STATUS_NOT_INITIALIZED` для RNN/GRU моделей**
Пакет `nvidia-cudnn-cu13` (CUDA 13) конфликтует с драйвером, поддерживающим только CUDA 12.8. Решение — заменить на cu12:
```sh
ssh knitwork-server "/opt/uv-envs/knitwork/.venv/bin/pip uninstall nvidia-cudnn-cu13 -y && \
    /opt/uv-envs/knitwork/.venv/bin/pip install nvidia-cudnn-cu12==9.10.2.21 --force-reinstall"
```

**`ssh knitwork-server-global` не подключается**
WireGuard не поднят. Локально: `sudo wg-quick up wg-knitwork`. Если playit упал на сервере: `ssh knitwork-server "sudo systemctl restart playit"`.

**`Connection closed by 10.8.0.1 port 2222`**
MTU-проблема (сработает при обновлении SSH). В `~/.ssh/config` для `knitwork-server-global` прописан `KexAlgorithms curve25519-sha256` — решает проблему.

**Очередь не запускает эксперименты**
```sh
ssh knitwork-server "systemctl --user restart knitwork-queue"
ssh knitwork-server "systemctl --user status knitwork-queue"
```

**`uv run` не видит пакеты**
Venv не слинкован. На сервере: `ln -sfn /opt/uv-envs/knitwork/.venv ~/knitwork/.venv`

**Мало места на диске сервера**
Очистить кэши (безопасно): `ssh knitwork-server "rm -rf ~/.cache/uv/ ~/.cache/pip/"`

## Experiment tracking (AIM)

Experiments are logged to AIM. Projects:
- `grid-rnn-sdq` — SDQ experiments
- `grid-rnn-text` — text experiments  
- `grid-rnn-treasure` — TreasureHunt experiments



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
