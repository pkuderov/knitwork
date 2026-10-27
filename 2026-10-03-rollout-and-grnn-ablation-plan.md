# Экономная проверка rollout и абляций Grid-GRU

Дата: 03.10.2026. [Обновлённый анализ Comet](2026-10-03-lru-comet-analysis.md) показывает преимущество two_reads в раннем обучении, но пока не отвечает на вопрос о длинном BPTT и вкладе регуляризаторов Grid-GRU. Этот план содержит **готовые конфигурации**, без запуска длительного обучения.

## Что подготовлено

- [rollout_mid](knitwork/exps/text/config/rollout_mid): 16 файлов, четыре модели × rollout 64/256/512/1024.
- [rollout_mid/low_memory](knitwork/exps/text/config/rollout_mid/low_memory): 8 файлов, те же четыре модели × rollout 64/512, вдвое меньше токенов на update.
- [grnn_ablation_mid](knitwork/exps/text/config/grnn_ablation_mid): все 8 комбинаций noise/communication/entropy при одной архитектуре Grid-GRU L2C4/H136.

Под «grnn-rnn» здесь понимается текущий **Grid-GRU**, registry `grnn`, bank2, stochastic MHA2, четыре attention heads и message LayerNorm. Это не LRU-вариант. Для каждого эксперимента отдельный YAML, без необходимости редактировать общую `mid.yaml`.

| Имя семейства в файлах | Registry / конфигурация | Размер | Параметры с Text8 V27 | Переносимое состояние на поток, floats |
| --- | --- | --- | ---: | ---: |
| `grnn_gru` | `grnn.L2C4` | L2C4/H136 | 1,050,499 | 1,088 |
| `grnn_lru` | `grnn_lru.L2C4` | L2C4/H180 | 1,049,463 | 3,600 |
| `gru` | `rnn.L2` | L2/H288 | 1,014,363 | 576 |
| `lru` | `lru.L2` | L2/H358 | 1,046,103 | 1,432 |

Для обычной LRU добавлен [lru_core.py](knitwork/models/lru_core.py): две последовательные ячейки из **того же LruBank**, что в Grid-LRU, с теми же complex recurrence, gamma и SiLU/residual output. Здесь нет межколоночного роутера и обратной связи от emitted messages. Это контроль из одинаковых ячеек, а не перенос другой исторической реализации `LRUCell`. Таблица сопоставляет веса, но не уравнивает размер состояния или compute.

## Минимальный содержательный бюджет

По умолчанию все новые YAML используют **67,108,864 обучающих токена**, то есть 2048 updates при 32,768 токенах/update. Это примерно 6.7% прежнего бюджета 1B. LR warmup занимает 600 updates / 19.66M токенов; reset warmup — 1500 updates / 49.15M. Поэтому пилот включает 548 updates после reset warmup; запуск на 5–20M в основном измерял бы начальную фазу обучения.

Пилот проверяет раннюю сходимость и устойчивость. Отсутствие выигрыша на 67M не исключает пользу длинного BPTT позднее: штатная reset probability ещё сокращает многие истории. Для следующей проверки предусмотрен бюджет **201,326,592** токена / 6144 updates. Warmup и LR/reset schedules в коротком пилоте не ускоряются и не сжимаются; результат соответствует раннему префиксу исходного протокола, а не специально подобранному короткому schedule.

| Rollout | Обучающие потоки | Токенов/update | Updates в пилоте |
| ---: | ---: | ---: | ---: |
| 64 | 512 | 32,768 | 2048 |
| 256 | 128 | 32,768 | 2048 |
| 512 | 64 | 32,768 | 2048 |
| 1024 | 32 | 32,768 | 2048 |

Внутри семейства одинаковы seed, модель, optimizer, число tokens/updates и LR schedule. Состояние переносится между роллаутами; detach задаёт горизонт градиента, а не обязательный сброс памяти. При изменении `n_envs` меняются стартовые позиции потоков и прочитанные подотрезки текста, поэтому это не строго один и тот же упорядоченный поток данных. Длинный rollout также уменьшает параллелизм и увеличивает число последовательных операций: равное число токенов/update не гарантирует одинаковое время или память.

У Grid-GRU сохраняются исходные `noise_std=0.05`, `loss_weight=0.05`, `entropy_weight=0.005`; прочие три модели обучаются по task loss. Эффект rollout оценивается **внутри каждой архитектуры**. Чтобы отдельно оценить вклад регуляризации в межархитектурное различие, служит серия ablation ниже.

## Что оптимизировано в runner

1. Отключены дорогие CKA/attention visualizations и периодические inspection вызовы для пилотов.
2. `eval.n_envs: 512` фиксирует потоки валидации независимо от обучающих. Это сохраняет одни и те же стартовые позиции и стоимость оценки при всех rollout.
3. `eval.max_tokens: 524288` ограничивает каждую оценку первыми 1024 шагами каждого из 512 validation-потоков, равномерно стартующих на полном validation split. Train/val/test остаются 90M/5M/5M; словарь строится только по train. При cap не читается весь validation split.
4. Каждые 16,777,216 обучающих токенов выполняется одна непрерывная оценка: всего четыре validation-прохода на пилот. При cap1024 отдельный проход с window1024 не проверяет сокращение контекста, поэтому `eval.window: null` исключает эту дополнительную оценку. Best выбирается по пилотной `val/Loss`. Повторная финальная validation на том же шаге исключена. При финальном подтверждении возвращаются полные continuous и window1024 проходы.
5. `eval.test_on_finish: false` отключает тест при screening. Test не расходуется на выбор вариантов и не добавляет прежние четыре полных прохода в конце.
6. Comet logging выполняется каждые 4,194,304 токена; checkpoints/метрики выбираются обычным текущим runner. Runner не реализует resume обучения; увеличение бюджета после завершённого пилота означает новый запуск с нуля.

Пилотные validation numbers получены на меньшем фиксированном наборе и не должны напрямую сравниваться с опубликованными/старыми full-validation результатами. Новые параметры runner опциональны: без них сохраняются прежние потоки оценки, полные проходы и тест в конце.

## Сохранение reset-протокола

Текущий runner делит `reset_prob.val/tar` на rollout. В готовых конфигурациях исходные значения умножены на `rollout/64`: например, для 512 стоят 1.6 и 0.008, а после деления становятся прежними 0.003125 и 0.000015625. Значение 1.6 — промежуточная конфигурационная величина, не фактическая вероятность.

`warmup.val=0.1` сохранён: runner его не делит. Schedule и число warmup events при основном батче одинаковы. Не следует просто менять rollout в старой YAML без этой коррекции: тогда одновременно увеличится горизонт градиента и уменьшится частота сбросов памяти.

## Команды: сначала четыре запуска, затем необходимые контроли

Все команды выполняются из корня репозитория. `uv run python` можно заменить на `.venv/bin/python`. Укажите путь к настоящему Text8 и нужный GPU; без `--device` действует обычный автоматический выбор устройства.

```bash
# Первый screening: Grid-GRU и Grid-LRU, rollout 64 против 512.
for family in grnn_gru grnn_lru; do
  for rollout in 64 512; do
    uv run python -m knitwork.exps.text.run \
      "knitwork/exps/text/config/rollout_mid/${family}_r${rollout}.yaml" \
      --n_steps=67108864 --device=cuda:0 \
      --gens.text8.path=/absolute/path/to/text8.txt
  done
done

# Затем обычные GRU/LRU: дополняют сравнение модульных моделей с одиночными.
for family in gru lru; do
  for rollout in 64 512; do
    uv run python -m knitwork.exps.text.run \
      "knitwork/exps/text/config/rollout_mid/${family}_r${rollout}.yaml" \
      --n_steps=67108864 --device=cuda:0 \
      --gens.text8.path=/absolute/path/to/text8.txt
  done
done
```

Начальные четыре запуска суммарно используют 268.4M токенов, вся минимальная четырёхмодельная серия — 536.9M. Не запускать сразу все 16 rollout-конфигов. Если 512 нестабилен или эффект неясен, сначала промежуточный 256; 1024 оставить для модели с обоснованным направлением улучшения.

```bash
# Промежуточный rollout, тот же минимальный бюджет.
uv run python -m knitwork.exps.text.run knitwork/exps/text/config/rollout_mid/grnn_lru_r256.yaml \
  --n_steps=67108864 --gens.text8.path=/absolute/path/to/text8.txt

# Следующий бюджет: запуск заново, а не resume предыдущих 67M.
uv run python -m knitwork.exps.text.run knitwork/exps/text/config/rollout_mid/grnn_lru_r512.yaml \
  --n_steps=201326592 --name=rollout_mid/grnn_lru/r512/confirm201M \
  --gens.text8.path=/absolute/path/to/text8.txt

# Проверка 1024, когда есть основания расширить горизонт.
uv run python -m knitwork.exps.text.run knitwork/exps/text/config/rollout_mid/grnn_lru_r1024.yaml \
  --n_steps=67108864 --gens.text8.path=/absolute/path/to/text8.txt
```

Для подтверждения повторить **обе** длины выбранного семейства с seeds 1337 и 31337, а не только понравившийся результат. Сопоставлять BPC на тех же training tokens, поздние две-три validation-точки, skipped/nonfinite gradients, время/update и память. Разницу около 0.001–0.003 BPC по одному seed не считать убедительным выигрышем. На большем бюджете сравнивать несколько последних точек; это правило screening, не статистический тест значимости.

## Если основной батч не помещается

Использовать готовую **отдельную** серию `low_memory`, где rollout64 имеет 256 потоков, rollout512 — 32: оба дают 16,384 токена/update. Это приблизительно сокращает объём сохранённых активаций, без гарантии конкретного GPU peak. Число updates удвоено до 4096; LR/reset event schedules и warmup schedules тоже удвоены, чтобы сохранять момент изменения по обучающим токенам. Числа linear warmup events не меняются.

```bash
for rollout in 64 512; do
  uv run python -m knitwork.exps.text.run \
    "knitwork/exps/text/config/rollout_mid/low_memory/grnn_lru_r${rollout}.yaml" \
    --n_steps=67108864 --gens.text8.path=/absolute/path/to/text8.txt
done
```

В smaller-batch серии иной размер optimizer batch и больше апдейтов. Сравнивать её контроль64 с её rollout512, а не смешивать с основным батчем. Не уменьшать `n_envs` только у одного варианта и не менять LR вручную без парного контроля.

## Grid-GRU: абляции регуляризаторов

Все YAML используют одинаковые L2C4/H136, MHA, LayerNorm, веса, число потоков, rollout64 и пилотный протокол. Меняются только три фактора. Learning-rate tuning для каждого варианта отдельно не выполняется: сначала измеряется эффект при общем schedule.

| YAML | Routing noise | Communication weight | Entropy weight |
| --- | ---: | ---: | ---: |
| `full.yaml` | 0.05 | 0.05 | 0.005 |
| `none.yaml` | 0 | 0 | 0 |
| `no_noise.yaml` | 0 | 0.05 | 0.005 |
| `no_comm.yaml` | 0.05 | 0 | 0.005 |
| `no_entropy.yaml` | 0.05 | 0.05 | 0 |
| `noise_only.yaml` | 0.05 | 0 | 0 |
| `comm_only.yaml` | 0 | 0.05 | 0 |
| `entropy_only.yaml` | 0 | 0 | 0.005 |

**Минимальный порядок:** full/none → три одиночных выключения → три оставшихся combination только при необходимости разобрать взаимодействия. `full.yaml` и `rollout_mid/grnn_gru_r64.yaml` совпадают во всём, кроме имени; не запускать один и тот же контроль дважды. Для проверки полного вклада регуляризации после rollout-серии сначала нужен лишь **один дополнительный** запуск none.

```bash
# Только если общий Grid-GRU r64 контроль ещё не запускался.
uv run python -m knitwork.exps.text.run knitwork/exps/text/config/grnn_ablation_mid/full.yaml \
  --n_steps=67108864 --gens.text8.path=/absolute/path/to/text8.txt

# Первый обязательный новый контроль: task loss без трёх регуляризаторов.
uv run python -m knitwork.exps.text.run knitwork/exps/text/config/grnn_ablation_mid/none.yaml \
  --n_steps=67108864 --gens.text8.path=/absolute/path/to/text8.txt

# Распределить вклад отдельных компонентов.
for variant in no_noise no_comm no_entropy; do
  uv run python -m knitwork.exps.text.run \
    "knitwork/exps/text/config/grnn_ablation_mid/${variant}.yaml" \
    --n_steps=67108864 --gens.text8.path=/absolute/path/to/text8.txt
done
```

Если full/none существенно различаются, посмотреть направления одиночных выключений и повторить наиболее важную пару с дополнительными seeds/бюджетом. Если отдельные выключения не объясняют full/none, доступны `noise_only`, `comm_only`, `entropy_only`, но автоматически запускать их не нужно. Убрать auxiliary losses из архитектуры и оценить результат — полноценная абляция; она не требует добавления новых лоссов или усложнения модели.

## Финальное подтверждение

Когда вариант выбран по пилотной validation, вернуть полную оценку и один нетронутый test этап. Сохранять фиксированные eval streams при сравнении rollout. Следующая команда запускает новый полный эксперимент, не продолжает pilot:

```bash
uv run python -m knitwork.exps.text.run knitwork/exps/text/config/rollout_mid/grnn_lru_r512.yaml \
  --n_steps=1000013824 --eval.max_tokens=None --eval.window=1024 --eval.test_on_finish=true \
  --eval.schedule=20000000 --name=rollout_mid/grnn_lru/r512/full1B \
  --gens.text8.path=/absolute/path/to/text8.txt
```

1,000,013,824 токена — 30,518 полных updates по 32,768, без потерянного последнего частичного optimizer batch. Для регуляризаторного подтверждения аналогично изменить бюджет и eval-параметры у выбранной пары ablation. Размер split, LR и reset schedules сохраняются.

## Проверки реализации

Тесты [test_rollout_pilots.py](tests/test_rollout_pilots.py) проверяют одинаковые per-token reset trajectories, точные параметры и бюджеты, все восемь уникальных combinations, повторное использование общего контроля, градиенты через rollout512 у всех четырёх моделей, plain-LRU cell equivalence, reset/detach/compile, фиксированные eval streams, cap, отсутствие test в pilot и сохранение прежнего full-eval режима по умолчанию.

```bash
uv run python -m unittest discover -s tests -p test_rollout_pilots.py -v

# Два update настоящей конфигурации для проверки запуска и памяти.
# Это технический smoke, не бюджет для оценки качества.
uv run python -m knitwork.exps.text.run knitwork/exps/text/config/rollout_mid/grnn_lru_r512.yaml \
  --n_steps=65536 --eval.enabled=false --log.logger=None \
  --gens.text8.path=/absolute/path/to/text8.txt
```

Проверено: 60 тестов общей серии прошли; четыре модели полной ширины mid прошли по два CPU-update с rollout512 через настоящий CLI на синтетическом 27-символьном тексте. Это проверка запуска, не оценка качества или GPU-памяти. Для CLI отсутствие значения записывается как `None`, а в YAML — как `null`: парсер аргументов использует Python literals.

Долгие GPU-прогоны и новые Comet-запуски в рамках подготовки не выполнялись. Будущие архитектурные контроли из отчёта — fixed gates, residual query-read и fixed second query — остаются предложениями; эта серия готовит именно rollout и Grid-GRU regularizer ablation.
