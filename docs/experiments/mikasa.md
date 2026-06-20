# MIKASA / POPGym — Бенчмарк памяти

## Постановка задачи

POPGym (Partially Observable Process Gym, Morad et al. 2023) — набор диагностических сред для оценки различных типов памяти в рекуррентных RL-агентах. В отличие от TreasureHunt (один тип задачи) или SDQ (синтетический тест ассоциации), POPGym охватывает четыре ортогональных аспекта памяти:

| Тип | Что требуется | Примеры сред |
|---|---|---|
| **Object memory** | Удерживать свойства конкретного объекта (object permanence) | RepeatFirst, HigherLower |
| **Sequential memory** | Запоминать порядок событий | RepeatPrevious, Autoencode |
| **Capacity** | Удерживать несколько элементов одновременно | MultiarmedBandit, CountRecall |
| **Spatial memory** | Ориентироваться в пространстве без карты | Battleship, MineSweeper |

Все выбранные среды используют **дискретные действия** и **векторные наблюдения** — они совместимы с существующими моделями knitwork без изменений.

---

## Среды

### Object memory

**RepeatFirst** (`popgym-RepeatFirstEasy/Medium/Hard-v0`)

Агент получает последовательность символов из алфавита размером N; в конце эпизода должен воспроизвести **первый** символ последовательности. Тест на долгосрочное хранение одного значения.

- Obs: `Discrete(N)` → токен `[B, 1]`
- Actions: `Discrete(N)`
- Сложность Easy → Hard: увеличивается длина последовательности

**HigherLower** (`popgym-HigherLowerEasy/Medium/Hard-v0`)

Карточная игра: агент видит карту и предсказывает, будет ли следующая **выше** или **ниже**. Требует удерживать в памяти распределение уже сыгранных карт.

- Obs: `Discrete(13)`, Actions: `Discrete(2)`

### Sequential memory

**RepeatPrevious** (`popgym-RepeatPreviousEasy/Medium/Hard-v0`)

На каждом шаге агент видит символ, а в следующем должен его воспроизвести (т.е. всегда отвечать с задержкой k шагов).

- Obs: `Discrete(N)`, Actions: `Discrete(N)`

### Capacity (объём рабочей памяти)

**MultiarmedBandit** (`popgym-MultiarmedBanditEasy/Medium/Hard-v0`)

Многорукий бандит: награды рычагов задаются случайно в начале эпизода. Агент должен исследовать и запомнить доходность каждого рычага.

- Obs: `Discrete(2)` (успех/неуспех), Actions: `Discrete(K)`, где K = количество рычагов

**CountRecall** (`popgym-CountRecallEasy/Medium/Hard-v0`) — *непрерывные наблюдения*

Агент считает вхождения символов, затем должен воспроизвести точный счётчик.

- Obs: `MultiDiscrete([2, 2])` → linear encoder `[B, 2] → [B, embedding_size]`
- Actions: `Discrete(26)`

### Spatial memory

**Battleship** (`popgym-BattleshipEasy/Medium/Hard-v0`) — *MultiDiscrete actions*

Агент топит корабли на сетке, получая бинарную обратную связь (попал/промах). Требует карту уже исследованных клеток.

- Obs: `Discrete(2)`, Actions: `MultiDiscrete([8, 8])` *(пока не поддерживается)*

**MineSweeper** (`popgym-MineSweeperEasy/Medium/Hard-v0`) — *MultiDiscrete actions*

Классический сапёр с частичной видимостью.

- Obs: `Discrete(3)`, Actions: `MultiDiscrete([4, 4])` *(пока не поддерживается)*

> **Примечание:** среды с `MultiDiscrete` action space (Battleship, MineSweeper) требуют отдельного action head и в текущей реализации не поддерживаются. Spatial memory частично покрывается через HigherLower (запоминание распределения карт = имплицитная пространственная статистика).

---

## Интеграция наблюдений

### Дискретные наблюдения (`Discrete(n)`) — без изменений в моделях

```python
obs_in = to_torch(raw['obs'], device).view(-1, 1)   # [B, 1] int64
rnn_state = rnn.reset_state(rnn_state, reset_mask)
logits, rnn_state = rnn(obs_in, rnn_state)
```

Токен передаётся напрямую в `nn.Embedding(n_tokens, embedding_size)` модели.

### Непрерывные наблюдения (`MultiDiscrete` / `Tuple`) — обёртки из `rl_wrappers.py`

```python
obs_in = to_torch(raw['obs'], device).to(dtype)     # [B, obs_dim] float32
```

В `GridRnnContinuous` (и аналогах) `nn.Embedding` заменён на `nn.Linear(obs_dim, embedding_size)`:

```python
x = self.embedding(obs.float())   # [B, obs_dim] → [B, embedding_size]
h, extras = self.grid_step_postmsg(x, h=h)
```

Ядро сетки (cells, attention, head) наследуется без изменений.

---

## Алгоритм обучения

**PPO + GAE** — идентично TreasureHunt:

| Параметр | Значение |
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

**Actor-critic:** рекуррентная модель как актор + отдельный `nn.Linear(hidden, 1)` как critic поверх верхнего слоя скрытого состояния.

---

## Метрики

| Метрика | Описание |
|---|---|
| **MeanReward** | Среднее вознаграждение за шаг в ролауте |
| **ep\_return** | Полный возврат за эпизод (EMA по последним n\_envs завершённым) |
| **ep\_length** | Средняя длина завершённого эпизода |
| **PolicyLoss** | PPO clipped surrogate loss |
| **ValueLoss** | MSE critic |
| **Entropy** | Энтропия политики |

Главная метрика — `ep_return`: насколько высокую суммарную награду получает агент за эпизод. Для RepeatFirst это пропорционально числу правильных ответов в конце.

---

## Запуск

```sh
# Object memory — RepeatFirst (стартовый тест)
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatFirstEasy-v0 --model=grnn

# Sequential — RepeatPrevious
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatPreviousEasy-v0 --model=grnn_lru

# Capacity — MultiarmedBandit
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-MultiarmedBanditEasy-v0 --model=hgrnn

# Capacity с непрерывными obs — CountRecall (нужен continuous wrapper)
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-CountRecallEasy-v0 --model=grnn

# Smoke-test (без AIM, 50k шагов)
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatFirstEasy-v0 --model=grnn --n_steps=5e4 --log.enabled=false
```

### Запуск нескольких сред для сравнительного анализа

```sh
# Параллельно — не более 3 одновременно (правило проекта)
uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatFirstEasy-v0 --model=grnn_lru --name="RepeatFirst grnn_lru" &

uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-RepeatPreviousEasy-v0 --model=grnn_lru --name="RepeatPrev grnn_lru" &

uv run knitwork/exps/mikasa/run_mikasa.py knitwork/exps/mikasa/config_mikasa.yaml \
  --env=popgym-MultiarmedBanditEasy-v0 --model=grnn_lru --name="Bandit grnn_lru" &
```

---

## Логирование

AIM проект: `grid-rnn-mikasa`. Метрики по умолчанию: `MeanReward`, `ep_return`, `ep_length`, `PolicyLoss`, `ValueLoss`, `Entropy`, `fps`.

---

## Результаты экспериментов

### grnn_harmonic — MIKASA v6 (2026-06-17, запущено)

Модель: `HarmonicGridRNN 3L×4C LRU + 0Res | hidden=128 heads=4 mem=(32×128) ema=0.9` (2.11M params)

Алгоритм: PPO | n_envs=64 | rollout_len=32 | LR=8e-4 (warmup+cosine) | 200M шагов

| # | Env | Прогресс | LR | PL | VL | H | R | EpRet | Статус |
|---|---|---|---|---|---|---|---|---|---|
| #076 | RepeatFirstEasy | 20.5M/200M (10%) | 80% | -0.000 | 0.007 | 1.32 | -0.009 | -0.499 | RUNNING |
| #077 | RepeatFirstMedium | 20.5M/200M (10%) | 80% | -0.002 | 0.000 | 1.33 | -0.001 | -0.496 | RUNNING |
| #078 | MultiarmedBanditEasy | 20.0M/200M (10%) | 79% | -0.002 | 0.244 | **2.27** | **0.000** | **+0.021** | RUNNING |
| #079 | MultiarmedBanditMedium | — | — | — | — | — | — | — | PENDING |

**Наблюдения (20M шагов):**
- BanditEasy (#078) показывает лучшее поведение: энтропия 2.27 (активное исследование), EpRet чуть положительный (+0.021). Модель осваивает задачу исследования.
- RepeatFirst Easy/Medium (#076, #077) застряли в отрицательных EpRet ≈ -0.5. Энтропия умеренная (1.32–1.33). Задача на долгосрочную память — модель пока не освоила.

### grnn_harmonic — MIKASA v5 (2026-06-17, прервано)

Прогресс до 29M/200M шагов перед прерыванием (exit=-15, SIGTERM):

| # | Env | Прогресс | H | R | EpRet |
|---|---|---|---|---|---|
| #074 | RepeatFirstEasy | 29M/200M | 0.78 | -0.009 | -0.555 |
| #075 | RepeatFirstMedium | 29M/200M | 0.22 | -0.001 | -0.529 |

**Наблюдения v5:** энтропия резко упала (0.78 и 0.22) — модель схлопывается в одну стратегию, не выучив задачу. Это мотивировало правки в v6 (entropy_coef, warmup, Hopfield gate init).
