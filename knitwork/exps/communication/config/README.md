# Причинные контроли коммуникации MoSAIC, около 1M параметров

Основная матрица: [matrix.yaml](matrix.yaml). Полные конфиги задач: [text8.yaml](text8.yaml) и [sdq.yaml](sdq.yaml). Запуск: `knitwork.exps.communication.run`; подготовка и последовательный запуск матрицы: `knitwork.exps.communication.suite`; сбор результатов: `knitwork.exps.communication.summarize`.

Пакет подготовлен для главного замечания независимого аудита от 2026-10-07: отделить эффект коммуникации и зависимости маршрутов от содержимого от эффекта большего состояния и вычислительной схемы. Это новые эксперименты, а не результаты статьи. По выбору пользователя основная матрица содержит обе задачи, только форму L2C4 и один seed 42: **6 условий × 2 задачи × 1 форма × 1 seed = 12 обучений**. Два условия `dynamic_column0` уже запущены локально; L3C4 исключена из активного плана 2026-10-10 по запросу пользователя. Определения L3C4 в конфигурациях сохранены как доступный вариант, но в очереди они не используются.

## Шесть условий

| Условие (`--model=grnn_communication.<условие>`) | Межколоночный обмен | Зависимость маршрутов от содержимого | Readout |
|---|---|---|---|
| `dynamic_column0` | Да | Да | Верхняя колонка 0 |
| `dynamic_mean` | Да | Да | Среднее всех верхних колонок |
| `static_column0` | Да | Нет | Верхняя колонка 0 |
| `static_mean` | Да | Нет | Среднее всех верхних колонок |
| `none_column0` | Нет | Только локальное смешивание собственного feedback и ввода | Верхняя колонка 0 |
| `none_mean` | Нет | Только локальное смешивание собственного feedback и ввода | Среднее всех верхних колонок |

`dynamic_column0` использует исходный `StochasticMessagePassingLayer` без изменения формул. `static` сохраняет все Q/K/V/O-проекции, четыре головы, SiLU, identities, temperature и LayerNorm, но Q и K вычисляет только из обучаемых identities. Содержимое V остаётся текущим. Проекции пересчитываются на каждом шаге, без кэширования статических маршрутов. Это обучаемый content-independent router, а не замороженная случайная матрица и не прежний одноголовый `StaticMessagePassingLayer`.

В `none` вычисляется та же полная attention-схема, после чего маска запрещает межколоночные маршруты. Внизу каждой колонке доступны **собственное задержанное верхнее состояние и внешний ввод**; выше — только собственная колонка слоя ниже. Собственные GRU-состояния сохраняются на всех слоях. Общие проекции разделяют параметры между колонками, но состояния разных колонок друг другу не передаются. Маска ставится после temperature scaling, чтобы не получить `NaN` при дифференцировании.

Readout `mean` — без параметров; размер входа prediction head остаётся H. Смешивание для предсказания выполняется после recurrent updates и не возвращается в сетку как общее состояние. `none_mean` поэтому оставляет независимые колонки, но позволяет предсказанию использовать всю их память.

У `none_column0` остальные колонки намеренно не имеют пути к предсказанию. Этот вариант служит диагностикой удаления доступа к памяти. Его проигрыш **сам по себе не доказывает** пользу attention. В маршрутах с единственным разрешённым источником Q/K имеют нулевой task gradient. Поэтому отчёт различает номинальные обучаемые параметры и state с путём к readout; равенство номинального числа параметров не объявляется равенством эффективной ёмкости. Неиспользуемые параметры не добавляются специально для подгонки бюджета: сохранены исходные модули и dense execution.

## Что должно быть одинаковым

Внутри каждой задачи/формы/recipe/seed все шесть условий имеют одинаковые GRU, H, L, C, adapters, головы, LayerNorm и полный начальный `state_dict`. Различаются режим маршрутизации и readout. Seed модели и seed данных заданы отдельно; в основной матрице оба равны 42. Поток обучения, random resets, curriculum, batch, BPTT, RMSprop, clipping, LR, расписание оценки и оценочные примеры совпадают.

| Задача | Форма | H | Параметров, включая adapters | State-скаляров | Dense forward FLOPs/token |
|---|---|---:|---:|---:|---:|
| text8 | L2C4 | 136 | 1,050,499 | 1,088 | 3,060,272 |
| text8 | L3C4 | 112 | 1,068,743 | 1,344 | 3,090,080 |
| SDQ | L2C4 | 136 | 1,053,338 | 1,088 | 3,055,648 |
| SDQ | L3C4 | 112 | 1,071,078 | 1,344 | 3,086,272 |

Это бюджет **около 1M**, с исходными ширинами 1M MoSAIC; ширины не увеличиваются для более дешёвого контроля. Учёт включает vocabulary-dependent embedding/head, поэтому text8 и SDQ немного различаются. State одинаков внутри шести условий, кроме доступности дополнительных колонок в специально отключённом `none_column0`.

`suite --check --profile` проверяет равенство параметров, state, хэша начальных весов и измеренных dense forward и forward/backward FLOPs. Forward формула согласована с матричным учётом статьи. Forward/backward измеряется на трёх синтетических шагах с batch 2 и включает `mm/bmm/addmm`; это проверка арифметики реализации, а не обучение на задаче. Elementwise операции, softmax, нормализация и RMSprop не включены; равенство этих чисел не означает равенства полной стоимости или времени. Runner дополнительно логирует синхронизированное время тренировочного цикла, оценки, checkpoint I/O, число пропущенных updates и peak CUDA memory.

### Recipe

По умолчанию `recipe: clean`: у всех условий нет routing noise, communication penalty и entropy bonus. Так loss состоит только из задачи, и контраст не смешивается с различной реакцией масок на вспомогательные штрафы.

Для связи с исходным рецептом статьи предусмотрен `--recipe=paper`. Он возвращает одинаковые для шести условий noise/penalty/entropy: text8 `0.05/0.05/0.005`, SDQ `0.02/0.02/0.002`. Это отдельная проверка устойчивости при том же числе архитектурных условий. Чтобы запустить обе версии, указать `recipes: [clean, paper]` в копии `matrix.yaml`; получится 48 запусков при одном seed. В `paper` маршруты static могут шуметь на обучении, но остаются независимыми от содержимого; в evaluation noise выключен. Сравнивать clean-абляцию с историческим paper-результатом как чистый причинный контраст нельзя.

## Проверка и запуск

Все команды выполняются из корня репозитория. Используйте имеющееся окружение `.venv/bin/python` либо `uv run python`.

Сначала проверить всю матрицу, не читая датасет и не открывая Comet:

```bash
.venv/bin/python -m knitwork.exps.communication.suite --check --profile --report /tmp/communication-accounting.json
```

Показать все 12 команд без запуска обучения:

```bash
.venv/bin/python -m knitwork.exps.communication.suite
```

Для text8 задать существующий датасет через `MY_HOME` либо изменить `dataset.path`. Comet использует `COMET_API_KEY` и проект `knitwork-aistat`; секреты в YAML добавлять не требуется. При необходимости рабочее пространство задаётся обычным полем `log.comet_workspace`.

Последовательно запустить матрицу на локальном GPU с Comet и локальными артефактами:

```bash
.venv/bin/python -m knitwork.exps.communication.suite --execute --device cuda:0
```

Запустить только SDQ L2C4, с локальным логированием без подключения Comet:

```bash
.venv/bin/python -m knitwork.exps.communication.suite --execute --task sdq --topology L2C4 --logger local --device cuda:0
```

Один конкретный запуск:

```bash
.venv/bin/python -m knitwork.exps.communication.run knitwork/exps/communication/config/text8.yaml --model=grnn_communication.static_mean --topology=L2C4 --seed=42 --data_seed=42 --device=cuda:0 --checkpoint.every_tokens=100000000
```

Чтобы писать только локальные логи, к одиночной команде добавить `--log.logger=None`. Для CPU или диагностики можно задать `--device=cpu --compile=False`. Смена backend/compile требует отдельного output directory для корректного сравнения.

Runner использует ровно целое число optimizer updates: из цели 1B токенов text8 обучает **999,981,056**, SDQ **999,997,440**. Внутри каждой задачи это одинаково для всех условий. Остаток меньше одного batch явно записан через requested/effective budget; нет незавершённого BPTT rollout в финальном checkpoint.

## Сохранение и возобновление

`checkpoint.every_tokens` — N обработанных токенов, как `n_steps` в существующих экспериментах. По умолчанию N = 100,000,000. При пересечении каждой кратной N границы сохранение выполняется **после ближайшего полного optimizer update**. Это необходимо, чтобы сохранять полностью обработанный BPTT batch; при N меньше batch checkpoint будет после каждого update. В конце всегда сохраняется последний checkpoint. Расписание validation остаётся каждые 20M токенов: изменение checkpoint не меняет training/evaluation protocol и разрешено при resume.

Директория запуска:

```text
runs/communication_1m/<task>/<topology>/<recipe>/<condition>/seed42-data42/
```

`latest.pt` записывается через временный файл и atomic replace; предыдущая версия остаётся в `previous.pt`. `checkpoint.keep_history: true` дополнительно сохраняет `checkpoint-<tokens>.pt` для каждого события. Включение history потребует больше дискового места.

Checkpoint содержит веса, весь RMSprop state, recurrent state, позиции текстовых потоков либо полный генератор SDQ, его отдельный RNG, Python/NumPy/Torch/CUDA RNG, LR и reset schedules, curriculum, evaluation schedule, счётчики updates/skips, EMA/logger trackers и лучший validation checkpoint. Счётчики Numba schedules сериализуются как числа. Накопление градиентов внутри rollout не сохраняется, поскольку checkpoint всегда на границе update.

Возобновить одиночный запуск с теми же параметрами:

```bash
.venv/bin/python -m knitwork.exps.communication.run knitwork/exps/communication/config/text8.yaml --model=grnn_communication.static_mean --topology=L2C4 --seed=42 --data_seed=42 --device=cuda:0 --resume=auto
```

`--resume=auto` выбирает `latest.pt`, при его отсутствии — `previous.pt`; если сохранений ещё нет, начинает новый запуск. Можно передать конкретный путь: `--resume=runs/.../latest.pt`. Checkpoint восстанавливается с `weights_only=False`, поскольку содержит training state; используйте файлы, созданные этим runner.

Повторный `suite --execute` автоматически возобновляет незавершённые запуски и пропускает завершённые. Runner отказывается перезаписывать существующий run без resume. Он проверяет config fingerprint, хэш датасета и training/model source; изменения модели, данных, seed, recipe, бюджета или расписаний не допускаются для точного продолжения. Интервал checkpoint, logger и output directory можно изменить. Локальные метрики после восстановленного шага удаляются, чтобы не учитывать повторно потерянный участок. В Comet возобновление создаёт новый tracking run с тем же `run_id`; локальные файлы служат непрерывным источником данных для анализа.

CPU-тест проверяет побитовое совпадение непрерывного и возобновлённого обучения, в том числе с routing noise. Это не обещание побитовой воспроизводимости между разными устройствами, версиями PyTorch или compiled/eager backend.

## План запуска на RTX 5060 в течение нескольких дней

На проверенной локальной машине установлена RTX 5060 Laptop с 8 GiB VRAM. Память параметров 1M сама по себе мала, но выбор параллелизма должен учитывать BPTT activations, optimizer, compiled workspaces и другие процессы. Скорость на полном batch этих экспериментов ещё не измерена; предыдущие короткие GPU smoke checks подтверждают корректность, а не дневную производительность. Полная матрица требует около 12B training tokens плюс validation/test/probes.

Предлагаемые режимы:

1. **Одна последовательная очередь, рекомендованный стартовый режим.** Один GPU training process; завершённые запуски пропускаются, незавершённый продолжается через `--resume=auto`. Сначала получить dynamic/static контрасты в одной задаче/форме, затем независимые колонки и второй readout. Это сохраняет VRAM для полного исходного batch и даёт воспроизводимые измерения времени без конкурирующих обучений.
2. **Два процесса после короткого сравнительного замера.** Рассмотреть пару text8 + SDQ, если один process оставляет GPU простаивать и оба полного batch помещаются в VRAM с запасом для desktop/evaluation. Сравнивать суммарные training tokens/s и время на фиксированный общий token budget после компиляционного прогрева. Практический критерий выбора — устойчивый выигрыш порядка 10–15% и отсутствие OOM; это критерий планирования, а не измеренный результат. Не уменьшать batch/BPTT, не менять precision и recipe только ради размещения двух процессов: это изменит протокол. Два CUDA processes не гарантируют параллельного выполнения kernels: [NVIDIA описывает time slicing отдельных contexts](https://docs.nvidia.com/deploy/mps/architecture.html).
3. **Один процесс с чередованием условий блоками по 100M.** Последовательно обрабатывать очередной блок каждой конфигурации, сохранять и возобновлять её полный state. Так прогресс всех условий распределяется по дням, но возрастает число загрузок и компиляционных стартов. Текущий `suite` исполняет запуск до конца; такой round-robin требует дополнительного управления бюджетом одного сеанса, сохраняя неизменный общий training target.

Для нескольких дней использовать persistent compiler cache на том же диске/окружении; не очищать его между дневными сеансами. [PyTorch документирует compile-time caching](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html). Оценивать скорости после warmup, а не по первой компиляции. До сравнения измерить один и два процесса на одинаковых shapes, полном исходном batch и одинаковом token budget. Продолжительность матрицы оценивать как `12B / measured aggregate tokens_per_second`, с отдельной поправкой на evaluation и startup.

**Локальная очередь реализована:** [local_queue.yaml](local_queue.yaml), runner `knitwork.exps.communication.queue`. Сейчас предусмотрены два процесса, задачи чередуются: сначала `dynamic_column0` Text8/SDQ L2C4, затем `static_column0`, `none_column0`, варианты mean readout. Первый полный optimizer update прогревается до запуска второго процесса. При ошибке очередь останавливает остальные процессы с сохранением, чтобы не запускать следующие условия при системной проблеме. Batch, precision, общий budget и scientific recipe не меняются ради параллелизма. Фактический выигрыш скорости двух процессов пока не установлен; timings таких совместных запусков нельзя использовать как изолированные latency measurements архитектур.

**Остановка:** по указанию пользователя от 2026-10-12 ограничение времени отменено: `daily_cutoff: false`. Очередь продолжает работать после 23:00 и через полночь до завершения заданий или ручной остановки; deadline у workers отсутствует, процессы guardian не запускаются. Регулярные checkpoints сохраняются каждые 100M токенов. При ручной остановке процессы завершают текущий optimizer update и сохраняют полный checkpoint вне зависимости от кратности 100M. Оценки можно прерывать на границе recurrent step; незавершённая очередная evaluation выполняется заново после resume. Финальное обучение не считается завершённым, пока не закончены все финальные оценки. Ручная команда `queue --stop` отправляет `SIGUSR1` самому supervisor, а тот отправляет workers `SIGTERM`, который перехватывается Python-обработчиком и запрашивает cooperative pause на границе optimizer update. При реальном GPU-запуске `SIGUSR1` у workers не инициировал pause; `SIGTERM` проверен с успешным сохранением и exit code 0 на обеих задачах.

Для возврата прежнего ограничения после остановки очереди задайте `daily_cutoff: true`: в 22:55 Europe/Moscow запрашивается checkpoint и остановка, в 23:00 supervisor и независимые guardians принудительно завершают оставшиеся process groups. В аварийном случае возможен откат к последнему успешно записанному checkpoint. Если поле `daily_cutoff` отсутствует, прежнее ограничение действует по умолчанию.

**После остановки или перезагрузки запуск только вручную, согласно выбору пользователя.** Никаких cron/timer или утреннего автозапуска нет. При отключённом ограничении времени работающая очередь продолжает следующий день без перезапуска. Фоновый supervisor заканчивает сеанс после остановки workers. Закрытие терминала не останавливает `--detach`; выключение компьютера прекращает процессы. После перезагрузки запускать ту же команду: очередь восстанавливается из файлов, завершённые задания пропускаются, paused продолжаются с `resume=auto`. Если ключевая часть training/model кода или данных изменилась, точное возобновление отклоняется; до изменения исходников завершите очередь или используйте новый output directory.

```bash
# Начать / вручную возобновить после остановки; выполнять из корня репозитория.
.venv/bin/python -m knitwork.exps.communication.queue --detach

# Показать активные и ожидающие задания.
.venv/bin/python -m knitwork.exps.communication.queue --status

# Остановить оба обучения с checkpoint сейчас.
.venv/bin/python -m knitwork.exps.communication.queue --stop

# После устранения причины ошибки разрешить повтор failed задания.
.venv/bin/python -m knitwork.exps.communication.queue --detach --retry-failed
```

Конфиг очереди задаёт локальный абсолютный путь Text8 `/home/master/data/text/text8.txt`, устройство `cuda:0`, `workers: 2`, Comet, `daily_cutoff: false` и ручной restart. Для другой машины замените путь/устройство. Для последовательного режима измените только `workers: 1` после остановки supervisor. Один supervisor на output directory обеспечивается файловой блокировкой; живой осиротевший worker блокирует повторный запуск. Дополнительный конфиг передаётся позиционным аргументом.

Состояние очереди: `runs/communication_1m/_queue/status.json`; события: `_queue/events.jsonl`; supervisor log: `_queue/supervisor.log`. Отдельные stdout/stderr и материализованные YAML каждой работы: `_queue/jobs/<run_id>/train.log` и `config.yaml`. Оперативный progress каждого обучения: `<run_id>/progress.json` (tokens, updates, phase, PID, timestamp и peak CUDA memory), обновляется примерно раз в 10 секунд на границах optimizer updates; первый compile может длиться дольше. Сохранённые scientific checkpoints и метрики находятся в `<run_id>/`, отдельно от административных файлов очереди. Persistent compiler cache общий в `_compiler_cache`; одновременно допускается один compile worker на training process, старты разнесены минимум на 30 секунд.

## Дополнительные none-контроли от 2026-10-12

По запросу пользователя в `local_queue.yaml` добавлены четыре задания через `additional_configs`: две задачи для расширенного единственного столбца и две задачи для обучаемой головы над конкатенацией. Они идут после исходных 12 заданий; общий план очереди — **16 запусков**, каждый с одним seed 42 и data_seed 42. Основная матрица шести matched условий остаётся в `matrix.yaml`. Дополнительные результаты сохраняются в `runs/communication_extensions_1m`, их stdout и управляющие файлы — в той же общей очереди. Два одновременно работающих процесса и режим без остановки по времени сохраняются.

| Задача / контроль | Полный конфиг | Форма / ширина | Всего параметров | State, весь доступен предсказанию |
|---|---|---|---:|---:|
| Text8 / один столбец | [text8_none_single_column.yaml](text8_none_single_column.yaml) | L2C1 / H228 | 1,057,493 | 456 |
| SDQ / один столбец | [sdq_none_single_column.yaml](sdq_none_single_column.yaml) | L2C1 / H228 | 1,062,264 | 456 |
| Text8 / concat | [text8_none_concat.yaml](text8_none_concat.yaml) | L2C4 / H136 | 1,061,515 | 1,088 |
| SDQ / concat | [sdq_none_concat.yaml](sdq_none_concat.yaml) | L2C4 / H136 | 1,057,418 | 1,088 |

**Единственный столбец (`none_column0`, L2C1).** Две исходные GRU-ячейки остаются в замкнутом локальном цикле: нижняя читает внешний токен и задержанное верхнее состояние, верхняя читает обновлённое нижнее состояние. Остальных столбцов физически нет, а их параметры и вычисления не создаются. Ширина 228 выбрана как ближайшая кратная четырём ширина к фактическому номинальному бюджету исходного L2C4 в каждой задаче: 1,050,499 и 1,053,338 параметров соответственно. Отклонение составляет около 0.67% и 0.85%. Все 2 × 228 = 456 значений recurrent state имеют путь к предсказанию. Это контроль близкого числа параметров, а не равенства state или вычислений: у четырёхколоночных вариантов state 1088. Сохранены четыре attention heads, Q/K/V/O, LayerNorm и GRU bank=2. На верхнем слое единственный источник имеет вес 1; его entropy определена как 0, без деления на log(1). Верхние Q/K остаются номинальными параметрами с нулевым task gradient; отсутствие неиспользуемых столбцов не означает ненулевого градиента у каждого параметра.

**Четыре независимых столбца (`none_concat`, L2C4).** GRU-сетка, маска запрета межколоночного обмена, ширина и state полностью сохраняются как у `none_mean`. Вместо среднего формируется `z = concat(h_top[0], h_top[1], h_top[2], h_top[3])` длиной 544. Единственный линейный prediction head вычисляет `logits = W_head @ z + b_head`: размер матрицы 27 × 544 для Text8 и 10 × 544 для SDQ. Дополнительного скрытого слоя, нелинейности между concat и logits или общего состояния, возвращаемого в сетку, нет. Каждый блок из 136 входов получает свои обучаемые веса головы. Для равного начального предсказания веса исходной mean-головы копируются в четыре блока с множителем 1/4, bias копируется без изменения; в дальнейшем блоки обучаются независимо. Голова добавляет 11,016 параметров для Text8 и 4,080 для SDQ, что явно включено в полный count и FLOPs. State при этом остаётся ровно 1088. Сравнение с `none_mean` проверяет ограничение усредняющего readout; это не контроль строго одинакового числа параметров.

Датасеты, batch, BPTT, оптимизатор, LR, resets/curriculum, validation и fixed-gap probes, Comet и локальное логирование совпадают с исходным `clean`-протоколом соответствующей задачи. Бюджет — 999,981,056 токенов для Text8 и 999,997,440 для SDQ. Checkpoints сохраняются каждые 100M токенов и при cooperative pause; полный resume поддерживается.

Новые runners `run_single_column` и `run_concat` используют исходный training/evaluation/checkpoint цикл, регистрируя дополнительную модель только внутри своего процесса. Исходные файлы runner/protocol/model, хэши которых уже содержатся в checkpoints текущих обучений, не изменены. Хэши новых реализаций добавлены в поле `implementation` конфигурации и участвуют в checkpoint fingerprint: изменение этих файлов также блокирует точный resume. Для добавления заданий работающий supervisor перезапускается после сохранения workers, затем те продолжаются с `resume=auto`.

Запуск всей обновлённой очереди — прежняя команда `queue --detach`. Отдельные новые задания запускаются соответствующим runner:

```bash
.venv/bin/python -m knitwork.exps.communication.run_single_column knitwork/exps/communication/config/text8_none_single_column.yaml --device=cuda:0 --dataset.path=/home/master/data/text/text8.txt
.venv/bin/python -m knitwork.exps.communication.run_single_column knitwork/exps/communication/config/sdq_none_single_column.yaml --device=cuda:0
.venv/bin/python -m knitwork.exps.communication.run_concat knitwork/exps/communication/config/text8_none_concat.yaml --device=cuda:0 --dataset.path=/home/master/data/text/text8.txt
.venv/bin/python -m knitwork.exps.communication.run_concat knitwork/exps/communication/config/sdq_none_concat.yaml --device=cuda:0
```

Не запускайте эти команды параллельно с общей очередью для тех же run IDs. Обычный runner `communication.run` предназначен для исходных шести условий; для новых конфигов нужен указанный отдельный runner. Проверки: `test_communication_extensions.py` — эквивалентность одноколоночного контроля активному столбцу исходного none, начальное совпадение concat/mean, градиенты всех блоков головы без межколоночных state derivatives, counts/FLOPs, полный train/eval/checkpoint/resume на обеих задачах и фактический запуск новых runners через очередь. Запись: [communication_extensions_1m_validation.json](../../../../docs/experiments/communication_extensions_1m_validation.json).

## Оценки и артефакты

Локальные артефакты сохраняются независимо от Comet: `config.yaml`, `metadata.json`, `metrics.jsonl`, `summary.json`, `latest.pt`, `previous.pt`, `best.pt`, `final.pt`. Metadata содержит точные counts, hashes кода/датасета/initialization, seed/config identity, версию PyTorch и GPU. `metrics.jsonl` содержит Loss, BPC или Acc/Acc+/Acc++/Acc-, gradient norm, LR, routing cost/entropy и ресурсы. SDQ онлайн-метрики сохраняют отдельный статус training-generator diagnostic.

Text8: contiguous 90M/5M/5M при стандартном 100M-файле; vocabulary строится по train. Все условия оцениваются на одинаковых непрерывных потоках без случайных resets. Primary endpoint — **final-checkpoint continuous validation BPC**. Best checkpoint выбирается по этой же validation loss; после обучения отдельно оцениваются final и best на test. Этот выбор отличается от исторического reset-1024 checkpoint selector и одинаков для всех новых условий. Test используется только после обучения; его прежнее использование при разработке статьи не делает его новым нетронутым test. Финальные оценки сохраняют aligned segment loss sums/counts: `val_final_segments.npz`, `test_final_segments.npz`, `test_best_segments.npz`. Segment grouping не сбрасывает state и предназначен для анализа на одинаковых сегментах.

SDQ: кроме онлайн EMA есть независимый генератор с фиксированными финальными параметрами curriculum, отдельным seed и абсолютными gap bins. Primary memory endpoints — **held-out query accuracy** и **macro accuracy фиксированных gaps**. Финальная оценка сохраняет `val_final_gaps.npz`. Фиксированные probes с gaps 8/32/128/256/512 сохраняются в `fixed_gap_final.npz`. В них принудительно задаются store и query одной ассоциации на расстоянии ровно gap; промежуточные stores/queries других ключей и distractors следуют финальным вероятностям, учитываются distractor accumulator и store/query counters исходной задачи. Это отдельная интервенционная диагностика без overwrites выбранного ключа, а не замена исходного распределения SDQ. Примеры одинаковы между моделями; evaluation не изменяет training generator или его RNG.

Собрать завершённые результаты и заранее определённые контрасты:

```bash
.venv/bin/python -m knitwork.exps.communication.summarize runs/communication_1m --output /tmp/communication-results.json
```

Сборщик проверяет совпадение training protocol, source/data identity и initialization внутри каждого seed. Пропуски и skipped updates остаются явными. Незавершённые обучения не считаются результатами.

## Как отвечать на главное замечание ревью

Главные контрасты нужно интерпретировать отдельно для задач и форм:

1. `dynamic_column0 − static_column0`: полезна ли зависимость маршрутов от содержимого в исходном single-column readout, при том же state и dense execution.
2. `dynamic_mean − static_mean`: сохраняется ли этот эффект, когда вся верхняя память непосредственно доступна readout.
3. `dynamic_mean − none_mean`: полезен ли межколоночный обмен при доступе ко всем колонкам; локальное input/feedback gating сохранено в обоих.
4. `none_mean − none_column0`: насколько само открытие дополнительной памяти readout меняет результат. Это диагностический контраст, а не доказательство пользы коммуникации.

Для BPC отрицательная разница означает преимущество первой модели; для accuracy — положительная. Не объявлять необходимость attention, если static сопоставим с dynamic; не объявлять ненужность attention только из-за неразрешённой разницы. Если independent columns с общим readout сопоставимы с communicating grid, это свидетельство в пользу объяснения организацией/доступностью state, требующее соответствующего сужения claim.

Дизайн устраняет смешение количества state и dominant dense arithmetic в этих причинных сравнениях на 1M и сохраняет одинаковые cells/normalization. **Результат нельзя гарантировать кодом.** Один выбранный seed даёт описательные эффекты, но не междуобученческую оценку неопределённости; отдельные queries, сегменты и checkpoints её не заменяют. `summarize` при одном seed не выдаёт SD, CI или p-value. Для полного статистического ответа нужен заранее фиксированный набор ≥3, предпочтительно 5 seed; расширение выполняется только изменением `seeds` в матрице. Если добавлены seed, сборщик выдаёт номинальные paired-seed t intervals с явными допущениями и без multiplicity adjustment. Контроли на 1M не доказывают такой же механизм на 10M; претензии к силе внешних baselines и общая compute efficiency остаются отдельными вопросами.

## Проверки реализации

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_communication*.py' -v
.venv/bin/python -m unittest discover -s tests -p test_rollout_pilots.py -v
```

Тесты проверяют production equivalence dynamic/column0, независимость static routes от contents, нулевые межколоночные state derivatives в none, доступ ввода каждой колонке, readout/reset/detach/backprop/compile, matching для активных 12 конфигураций и профилированный dense arithmetic, точные native SDQ targets probes, aligned segments, полный цикл train/eval/checkpoint всех шести условий на обеих задачах и точное возобновление с отказом при изменении данных/модели. Дополнительные короткие GPU smoke checks относятся к реализации и не являются экспериментальными доказательствами для статьи. Запись проверок: [communication_1m_validation.json](../../../../docs/experiments/communication_1m_validation.json); полный учёт: [communication_1m_accounting.json](../../../../docs/experiments/communication_1m_accounting.json). На GPU с настоящим `torch.compile` отдельно проверены resume и побитовое совпадение весов/recurrent state непрерывного и возобновлённого обучения с routing noise.
