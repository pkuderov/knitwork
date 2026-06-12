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

## Результаты экспериментов

### Итерация 1 — June 2026 (в процессе)

Тестировались модели `grnn_ema_mem`, `grnn_delta`, `hgrnn_lru` на двух средах: RepeatFirstEasy (Object memory) и HigherLowerEasy (Sequential memory). 200M шагов PPO, 64 envs.

| Модель | Среда | EpRet | Прогресс | FPS | Статус |
|---|---|---|---|---|---|
| grnn_ema_mem | RepeatFirstEasy | **~0.95** | 74M/200M | ~1000 | running |
| grnn_ema_mem | HigherLowerEasy | ~0.41 | 73M/200M | ~1000 | running |
| grnn_delta | RepeatFirstEasy | -0.48 | 22M/200M | ~586 | running |
| grnn_delta | HigherLowerEasy | — | pending | — | pending |
| hgrnn_lru | RepeatFirstEasy | — | pending | — | pending |
| hgrnn_lru | HigherLowerEasy | — | pending | — | pending |

**Наблюдения:**
- `grnn_ema_mem` — сильная модель для Object memory: EpRet ~0.95 уже к 37% обучения на RepeatFirst. Surprise-EMA хорошо отделяет «важные» токены от рутинных. На HigherLower (Sequential) результат умеренный (~0.41).
- `grnn_delta` — существенно медленнее (~1.7× по FPS) из-за двухмасштабной delta-памяти. На 11% обучения EpRet отрицательный — модель не обучается или требует значительно большего числа шагов прогрева.
- `hgrnn_lru` — ожидает запуска; на SDQ показывала лучший результат среди всех моделей (Acc=0.967).

**Исправленные баги при запуске:**
- `grnn_delta` отсутствовал в реестре `run_mikasa.py` и конфиге — добавлен вручную.
- `hgrnn_lru.reset_state` падал с `TypeError` на float `reset_mask` — исправлен каст `.bool()`.

### Итерация 2 — June 2026 (в процессе, 4M/200M, ~2%)

Тестируются `grnn_lru`, `hgrnn_lru` на RepeatFirstEasy и RepeatPreviousEasy.
В очереди: `grnn_engram`, `grnn_fw` на RepeatFirstEasy.

| # | Модель | Среда | EpRet @ стоп | H | FPS | Итог |
|---|--------|-------|--------------|---|-----|------|
| 21 | grnn_lru | RepeatFirstEasy | ~−0.4 @ 15M | 1.07 | 338 | остановлено, нет прогресса |
| 22 | grnn_lru | RepeatPreviousEasy | −0.246 @ 15M | 0.64 | 337 | остановлено, коллапс энтропии |
| 23 | hgrnn_lru | RepeatFirstEasy | ~−0.5 @ 15M | 1.18 | 338 | остановлено, нет прогресса |
| 24 | hgrnn_lru | RepeatPreviousEasy | — | — | — | отменено (pending) |
| 25 | grnn_engram | RepeatFirstEasy | — | — | — | pending |
| 26 | grnn_fw | RepeatFirstEasy | — | — | — | pending |

**Наблюдения:**
- Ни одна модель не вышла в положительный EpRet за 7.5% обучения на RepeatFirst. Это ожидаемо — для Object memory нужно 20-30M+ шагов.
- `grnn_lru` на RepeatPrevious: ранний коллапс энтропии (H: 1.1 → 0.19 → 0.64). LRU без entropy regularization быстро детерминизируется при первом успехе — необратимо.
- `hgrnn_lru`: наибольшая H=1.18 среди всех (Hopfield-attention тормозит детерминизацию), но и наибольшая амплитуда осцилляций — нестабильное обучение.
- Все три запуска остановлены досрочно — результаты неокончательные.

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
