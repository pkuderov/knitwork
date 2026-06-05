# MIKASA / POPGym — Бенчмарк памяти для RL

## Описание

POPGym (Partially Observable Process Gym, Morad et al. 2023) и MIKASA-Base (Cherepanov et al. 2025) — диагностические бенчмарки для оценки способностей рекуррентных агентов к запоминанию информации в частично наблюдаемых средах. В отличие от SDQ (синтетический тест на ассоциативную память) и TreasureHunt (навигация с памятью), эти среды охватывают четыре канонических типа памяти, каждый изолированно:

| Тип памяти | Что тестируется | Примеры сред |
|---|---|---|
| **Object** | Сохранение свойств объектов (object permanence) | RepeatFirst, HigherLower, Passive-T-Maze |
| **Sequential** | Порядок событий во времени | RepeatPrevious, Autoencode |
| **Capacity** | Объём рабочей памяти (сколько элементов удерживается одновременно) | MultiarmedBandit, CountRecall |
| **Spatial** | Пространственное расположение и навигационные карты | Battleship, MineSweeper |

## Используемые среды

Эксперимент использует среды из POPGym с дискретными наблюдениями и дискретными действиями — они совместимы с существующими моделями knitwork без каких-либо изменений.

**Object memory:**
- `popgym-RepeatFirstEasy/Medium/Hard-v0` — показывается N символов, агент должен воспроизвести первый
- `popgym-HigherLowerEasy/Medium/Hard-v0` — угадать, будет ли следующая карта выше или ниже текущей

**Object + Sequential memory:**
- `popgym-RepeatPreviousEasy/Medium/Hard-v0` — воспроизвести символ, показанный k шагов назад

**Object + Capacity:**
- `popgym-MultiarmedBanditEasy/Medium/Hard-v0` — многорукий бандит, агент запоминает доходность рычагов

**Непрерывные наблюдения** (нужны обёртки из `rl_wrappers.py`):
- `popgym-CountRecallEasy-v0` — подсчёт вхождений; `MultiDiscrete([2, 2])` → linear encoder
- `popgym-AutoencodeEasy-v0` — воспроизвести последовательность; `Tuple` obs → linear encoder

## Ключевой механизм

Для сред с дискретным наблюдением `Discrete(n)` — токен передаётся напрямую в существующую модель:

```python
obs_in = to_torch(raw['obs'], device).view(-1, 1)   # [B, 1] int64
rnn_state = rnn.reset_state(rnn_state, reset_mask)
logits, rnn_state = rnn(obs_in, rnn_state)          # стандартный вызов
```

Для сред с непрерывным наблюдением используются обёртки из `knitwork/models/rl_wrappers.py`, которые заменяют `nn.Embedding` на `nn.Linear(obs_dim, embedding_size)`:

```python
# GridRnnContinuous — наследует GridRnn, переопределяет forward
x = self.embedding(obs.float())   # [B, obs_dim] → [B, embedding_size]
h, extras = self.grid_step_postmsg(x, h=h)
```

Обучение выполняется PPO с GAE (γ=0.99, λ=0.95, 4 эпохи, clip=0.2).

## Гиперпараметры

Нестандартные параметры:
- `rollout_len: 32` — длиннее чем в TreasureHunt (16), так как эпизоды в POPGym длиннее
- `n_envs: 64` — меньше чем в TreasureHunt (128), так как SyncVectorEnv медленнее GPU-среды
- `env:` — gym-идентификатор среды, переключается через `--env=popgym-...-v0`

## Запуск

```sh
# Object memory — RepeatFirst (лучший стартовый тест)
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatFirstEasy-v0 --model=grnn

# Sequential — RepeatPrevious
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatPreviousEasy-v0 --model=grnn_lru

# Capacity — MultiarmedBandit
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-MultiarmedBanditEasy-v0 --model=hgrnn

# Smoke-test (без AIM, 100k шагов)
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatFirstEasy-v0 --model=grnn --n_steps=1e5 --log.enabled=false
```
