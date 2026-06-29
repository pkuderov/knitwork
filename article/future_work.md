# Future Work — Что нужно доделать для AAAI 2026

Этот файл отслеживает недостающие эксперименты, данные и улучшения, необходимые
для доведения черновика до уровня сильной AAAI-публикации.

---

## Критически важно (блокируют таблицы сравнения)

### 1. Завершить MIKASA эксперименты до 200M шагов

| Модель | Среда | Статус | Что нужно |
|--------|-------|--------|-----------|
| grnn_ema_mem | RepeatFirstEasy | running (74M/200M) | дождаться финала |
| grnn_ema_mem | HigherLowerEasy | running (73M/200M) | дождаться финала |
| grnn_delta | RepeatFirstEasy | running (22M/200M) | дождаться финала или убить |
| hgrnn_lru | RepeatFirstEasy | pending | запустить |
| hgrnn_lru | HigherLowerEasy | pending | запустить |
| grnn (GRU) | RepeatFirstEasy | not started | **нужен baseline** |
| grnn (GRU) | HigherLowerEasy | not started | **нужен baseline** |

**Без GRU baseline Table 2 пустая и статья не может быть принята.**

Команда запуска GRU baseline на MIKASA:
```sh
bash server/knitwork_run.sh knitwork/exps/mikasa/run_mikasa.py \
    knitwork/exps/mikasa/config_mikasa.yaml \
    --model=grnn --env=popgym-RepeatFirstEasy-v0 -- --name "grnn RepeatFirstEasy baseline"

bash server/knitwork_run.sh knitwork/exps/mikasa/run_mikasa.py \
    knitwork/exps/mikasa/config_mikasa.yaml \
    --model=grnn --env=popgym-HigherLowerEasy-v0 -- --name "grnn HigherLowerEasy baseline"
```

### 2. Добавить published POPGym baselines из Morad et al. 2023

В статье Morad et al. 2023 (Table 2) приведены результаты GRU и FFM на всех
средах. Нужно:
- Прочитать статью POPGym из Zotero или arxiv
- Перенести числа GRU/LSTM в нашу таблицу
- Убедиться, что наша конфигурация (rollout_len=32, n_envs=64, PPO) совместима
  с их протоколом обучения

### 3. Более широкое покрытие MIKASA сред

Для убедительного RL-раздела нужно минимум 3 среды:
- RepeatFirst (object) ✓ in progress
- HigherLower (sequential) ✓ in progress  
- MultiarmedBandit (capacity) — **не запущен**
- RepeatPrevious (object + sequential) — был остановлен досрочно

Добавить MultiarmedBandit:
```sh
bash server/knitwork_run.sh knitwork/exps/mikasa/run_mikasa.py \
    knitwork/exps/mikasa/config_mikasa.yaml \
    --model=grnn_ema_mem --env=popgym-MultiarmedBanditEasy-v0 \
    -- --name "grnn_ema_mem MultiarmedBanditEasy"
```

---

## Важно (усиливают доказательную базу)

### 4. Ablation: число колонок на SDQ

Текущая таблица ablation показывает только 2 точки (2 cols / 4 cols).
Нужен полный sweep при фиксированном числе параметров ~2.1M:

| Конфигурация | H | Cols | Layers |
|---|---|---|---|
| 2 cols / 1 layer | 115 | 2 | 1 |
| 3 cols / 2 layers | ~120 | 3 | 2 |
| 4 cols / 3 layers | 128 | 4 | 3 |
| 5 cols / 3 layers | 116 | 5 | 3 |

Уже есть результаты для 2 и 4 cols (и 5 cols на shakespeare). Нужен 3 cols.

Команда:
```sh
bash server/knitwork_run.sh knitwork/exps/sdq/run_sdq.py \
    knitwork/exps/sdq/config/extend_config.yaml \
    --model=grnn models.grnn.n_columns=3 models.grnn.n_layers=2 \
    -- --name "grnn 3cols-2layers sdq ablation"
```

### 5. Ablation: post vs pre-messaging

Текущие все модели используют post-messaging (GRU сначала, потом attention).
Нужно сравнение с pre-messaging (attention до GRU) хотя бы на SDQ.

```sh
bash server/knitwork_run.sh knitwork/exps/sdq/run_sdq.py \
    knitwork/exps/sdq/config/extend_config.yaml \
    --model=grnn models.grnn.messaging=pre \
    -- --name "grnn pre-messaging sdq"
```

### 6. GRU baseline на SDQ (точный)

GRU baseline на SDQ-Hard показан как "~0.50" — нужен точный запуск.

```sh
bash server/knitwork_run.sh knitwork/exps/sdq/run_sdq.py \
    knitwork/exps/sdq/config/extend_config.yaml \
    --model=gru -- --name "gru sdq-hard baseline"
```

### 7. grnn_harmonic на MIKASA

HarmonicGridRNN (grnn_harmonic) протестирован на SDQ и text8, но не на MIKASA.
v3.1 запускался на 14.5M шагов — убит досрочно, результат неинформативный.
Нужен полный запуск v5.

```sh
bash server/knitwork_run.sh knitwork/exps/mikasa/run_mikasa.py \
    knitwork/exps/mikasa/config_mikasa.yaml \
    --model=grnn_harmonic --env=popgym-RepeatFirstEasy-v0 \
    -- --name "grnn_harmonic RepeatFirstEasy v5"
```

---

## Желательно (усиливают статью, но не блокируют)

### 8. Treasure Hunt результаты

Эксперимент TreasureHunt (навигация с памятью) — хорошее дополнение к MIKASA.
Нужны результаты grnn и grnn_lru на easy/medium difficulty.

```sh
bash server/knitwork_run.sh knitwork/exps/treasure/run_treasure_hunt.py \
    knitwork/exps/treasure/config_treasure_hunt.yaml \
    --model=grnn -- --name "grnn treasure easy"
```

### 9. Исправить Acc/store = NaN в grnn_harmonic

В grnn_harmonic метрика Acc/store = NaN на всём протяжении обучения SDQ.
Причина: маска sq_gaps не срабатывает. Это значит мы не знаем насколько хорошо
модель *сохраняет* пары (только *читает*). Нужно диагностировать и исправить
перед финальной публикацией.

### 10. Сравнение с HGRN2 на text8

HGRN2 (Qin et al. 2024) — один из ключевых contemporary baselines.
Нужны их числа при сопоставимом (≤3M) числе параметров. По публичным результатам
HGRN2 при ~2M params на text8 ≈ 1.76 BPC — но нужно найти точную цитату.

### 11. Medium/Hard MIKASA environments

Для более убедительных RL результатов нужно оценить на Medium/Hard complexity:
- RepeatFirstMedium, RepeatFirstHard
- HigherLowerMedium

---

## Технические долги

### 12. Нормализация нормы колонок в grnn_harmonic

На SDQ v3.1 наблюдается неограниченный рост нормы col0 на L2 (324→∞).
Hopfield attention с высоким β входит в attractor mode. Варианты fix:
- Pre-attention LayerNorm (добавлен в v4)
- Gradient clipping per-column
- Ограничение β per head

### 13. Entropy regularization для LRU в RL

grnn_lru и hgrnn_lru быстро детерминизируются под PPO (entropy collapse).
Нужно: увеличить entropy_coef (0.01 → 0.05), добавить warmup без LRU spectral
constraints на первые 5M шагов, или использовать entropy-regularized PPO.

### 14. Проверить совместимость rollout_len с оценкой POPGym

В Morad 2023 используется rollout_len=128 (или episodic truncation), у нас 32.
Несовместимость может занижать наши результаты. Нужно убедиться что протоколы
идентичны или явно указать разницу в статье.

---

## Цитирование

### Статьи из Zotero, которые нужно перечитать для Related Work:
- `/mnt/c/Users/master/Zotero/storage/2EQNWW7N/Chung и др. - 2017 - Hierarchical Multiscale Recurrent Neural Networks.pdf`
- `/mnt/c/Users/master/Zotero/storage/2GR8JM3G/Schlag и др. - 2021 - Linear Transformers Are Secretly Fast Weight Programmers.pdf`
- `/mnt/c/Users/master/Zotero/storage/7BOSCAG.../Boscaglia и др. - 2023 - A dynamic attractor network model.pdf`

### Найти и добавить в Zotero (пока нет):
- Morad et al. 2023 — POPGym (arxiv:2303.01859)
- Cherepanov et al. 2025 — MIKASA (arxiv:2501.14346)
- Orvieto et al. 2023 — LRU (arxiv:2303.06349)
- Qin et al. 2024 — HGRN2 (arxiv:2404.07904)

---

## Deadline / Timeline оценка

Для AAAI 2026 (дедлайн обычно август–сентябрь 2026):
- До 2026-07-01: завершить все RL эксперименты, получить GRU baselines
- До 2026-07-15: добавить ablations, исправить grnn_harmonic oscillation
- До 2026-08-01: финальный polishing, figures, review
- До 2026-08-15: submission buffer
