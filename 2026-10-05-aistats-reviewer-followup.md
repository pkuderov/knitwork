# Повторная сверка MoSAIC с reviewer и AI researcher — 5 октября 2026

Проверены [текущая статья](article/latex/paper.tex), её [PDF](article/latex/paper.pdf), сохранённая карта замечаний u43j/S6rY/Dpwf/AI-Reviewer в [плане пересдачи](2026-09-30-aistats-resubmission-plan.md), [текстовые замечания](2026-10-01-aistats-text-and-format-fixes.md), [августовский аудит](2026-08-10-mosaic-full-audit.md) и [AI-разбор архитектуры и логов](docs/reviews/grnn_core_analysis.md). Отдельных полных оригиналов трёх Official Review и AI Review в найденных файлах нет: сверка использует зафиксированные цитаты и карту претензий, а не новую выгрузку отзывов. Старые количественные сводки AI-разбора относятся к другому срезу и не заменяют текущий numeric provenance.

Итог: прежние проблемы несопоставимых горизонтов, отсутствующих новых baseline и неполного описания метода существенно исправлены. Главное остающееся возражение — причинный вывод: дополнительное состояние, иной readout, регуляризация и оптимизация ещё не отделены от эффекта модульной организации. Новая редакция этого не скрывает.

## Что исправлено в этом проходе

1. **Уточнено родство с BRIMs.** Bottom-up и delayed top-down flow названы общей структурной чертой, а вклад сформулирован через комбинацию независимых GRU, синхронного обновления всех столбцов и single-column readout. Это точнее, чем представлять layered feedback сам по себе как отличие. Позиционирование сверено с [исходной работой BRIMs](https://proceedings.mlr.press/v119/mittal20a.html).
2. **Topology claim стал описательным.** Увеличение числа столбцов сопровождается увеличением состояния и уменьшением ширины столбца. Наблюдаемый порядок результатов больше не сформулирован как изолированный причинный эффект объёма памяти.
3. **Уточнено parameter matching.** В основном тексте прямо указан диапазон новой серии **0.973M–1.069M** с embeddings/head. Это приблизительное совпадение числа весов; compute и state не совпадают.
4. **Пояснён readout confound.** Для исторических L2C4 → L2C16 ширина столбца и единственного readout уменьшается **424 → 224**. Это самостоятельное возможное объяснение ухудшения Text8, наряду с input-route cost; ни одно объяснение не объявлено доказанным.
5. **Снято неподтверждённое утверждение о независимости исторических seed.** При `seed: null` можно описывать разные запуски, но нельзя проверить независимость seed или воспроизвести их. Соответствующие предложения исправлены в протоколе и uncertainty appendix.
6. **Добавлено точное описание диагностик.** `col_sim` сравнивает усреднённые по батчу top-layer activation vectors через cosine similarity. Это не проверка специализации, дублирования информации или качества условного поиска; entropy не заменяет измерение градиентов.
7. **Исправлены ссылки на результаты.** Figure 3 теперь явно обсуждается в тексте приложения; добавлены ссылки на таблицы rollout, MQAR и RL. Все три рисунка имеют ссылки из прозы.

Показатели качества, состав cohort, H100 labels, AI disclosure и выбор seed/checkpoint в этом проходе не менялись. Новое обучение не запускалось.

## Десять основных претензий reviewer / AI Reviewer

«Исправлено» относится к конкретному дефекту отчётности или текста. «Частично» не означает, что нужный эксперимент можно заменить оговоркой.

| № | Претензия | Текущий ответ | Статус и оставшаяся работа |
|---|---|---|---|
| 1 | Parameter matching оставляет больше состояния и другой routing compute | §3.4, §4.6, §5, Tables 6–7 раскрывают оба бюджета; topology/readout confounds названы явно | **Не закрыта экспериментально:** нужен законченный state-matched контроль; не монотонный результат внутри Grid не опровергает альтернативу «Grid против GRU выигрывает из-за большего state» |
| 2 | Нет component ablations noise/communication/entropy | Table 3 содержит пять условий, §4.4 описывает качество и изменение route cost | **Частично:** общий checkpoint 50.33M, один seed, evaluation cap, конец reset warmup; нет полной проверки SDQ и projections/readout |
| 3 | Нет прямых RIMs/BRIMs/RMC references | RIMs/BRIMs есть в основной Table 2, по два seed; протокол реализации в Appendix D | **Частично:** RMC отсутствует; общая схема без полноценного подбора не воспроизводит лучшие результаты авторских реализаций |
| 4 | Современным baseline дано меньше токенов | Новые Mamba/HGRN2/DeltaNet/mLSTM имеют тот же поздний шаг основной таблицы; старые короткие результаты вынесены отдельно | **Исправлено для новой Text8-серии:** соответствующие полноценные old-scale и SDQ сравнения ещё отсутствуют |
| 5 | Text8 validation использовалась для tuning | Исторический 90M/10M и новый 90M/5M/5M явно разделены; все числа названы internal validation | **Не закрыта независимость оценки:** новая test-часть пересекается с исторической development validation; красивые test-числа не устранят этот факт |
| 6 | Только одна синтетическая задача и Text8 на одном масштабе | Добавлены второй parameter scale, MQAR и две POPGym-задачи; отрицательные результаты показаны | **Частично:** качество переноса слабое, пилоты одно-seed; подтверждённой широкой применимости нет |
| 7 | Нет throughput/latency/peak-memory trade-off | Table 4 и Appendix J дают интеграл logger fps для 48 H100-запусков; storage посчитан отдельно | **Частично:** это elapsed-loop proxy с overhead/contention, не controlled inference throughput, выделенные GPU-hours, FLOPs или peak memory |
| 8 | SDQ Acc++ зависит от плавающего batch gap threshold | Точное определение, EMA и curriculum описаны; единое окно показателей для всех запусков | **Не закрыта экспериментально:** нужен held-out fixed-gap retrieval test |
| 9 | Качество сравнивалось на разных endpoints/windows | Исторический Text8 переагрегирован на 960 036 864; SDQ использует одинаковые пять counters; Transformer interpolation раскрыта | **Исправлена отчётность:** интерполяция не равна повторной оценке сохранённого checkpoint; отдельные новые горизонты подписаны в Table 2 |
| 10 | Недостаточное позиционирование и пропущенные близкие работы | Goyal factorizing, Rahaman spatial, RIMs/BRIMs/RMC и workspace цитируются; общий паттерн BRIMs явно признан | **Исправлено в тексте:** оценка существенности архитектурного вклада остаётся предметом review |

## Все пункты прежнего аудита C/M/m

| Пункт | Проверка полученной статьи |
|---|---|
| C-1: token budget mismatch | Исправлен переагрегацией и явными отдельными горизонтами |
| C-2: floating Acc++ | Определён и ограничен как training diagnostic; независимый тест остаётся необходим |
| M-1: asymmetric tuning | Регуляризаторы указаны, короткий pilot есть; полноценный tuning record и repeated full-budget ablations отсутствуют |
| M-2: висящий Figure 2 | Старые learning curves имеют ссылки; дополнительно исправлен новый Figure 3 |
| M-3: нет modular baseline | Добавлены RIMs/BRIMs; RMC остаётся пробелом |
| M-4: нет чисел современных baseline | Новые поздние baseline включены в основной текст; reduced-token context отделён |
| M-5: SDQ нельзя воспроизвести по описанию | Appendix B задаёт vocabulary, target modulo, counters, missing queries, resets и curriculum; кодовый архив пока отсутствует |
| M-6: удобный GRU в abstract | Abstract содержит depth-matched и strongest GRU для SDQ, strongest GRU для Text8 |
| M-7: не объяснён column 0 | Объяснены намеренный bottleneck, отдельная input-cost asymmetry и изменение readout width |
| M-8: слишком оборонительный тон | Удалены полемические определения, вклад сформулирован положительно; квалификация pilots/local baselines сохраняется рядом с соответствующими числами |
| M-9: пропущенные citations | MoE, NTM, MQAR, induction, SiLU, RMSNorm, RoPE, RMSprop и две novelty references присутствуют |
| M-10: нестандартный Text8 split | Описаны оба split и validation reuse; сравнений с published test BPC нет |
| M-11: availability/resources/seeds | Есть release after acceptance, H100 subset, recorded new seeds и честное отсутствие historical seed replay; нет anonymous archive |
| m-1: размеры проекций | Явно заданы $d=H/R$, Q/K/V и output projection |
| m-2: индексация beta/identities | Есть индексы слоя и столбца |
| m-3: noise до temperature | Явно сказано, что effective logit noise scale равен beta × sigma |
| m-4: SiLU и value identities | Описано SiLU на Q/K/V, identities только Q/K; отсутствие изолированной проверки признано |
| m-5: какие конфигурации full | Full MoSAIC, ablation и LRU objectives явно различаются |
| m-6: лишнее обобщение числа inputs | Метод использует один input stream и bank C+1 |
| m-7: что означает Self-Attentive | В introduction явно описан fixed message bank, независимый от token history |
| m-8: неопределённый logging convention | Его заменяет конкретный common-budget reporting с counters и interpolation rules |
| m-9: RL появляется только в limitations | Есть отдельная main subsection и Appendix I с tasks, PPO, оценкой и отрицательными результатами |
| m-10: разные captions budget | Различаются target, actual logged horizon и reduced-token context |
| m-11: consistently improves без статистики | Итог ограничен конкретными internal protocols; SD не названа significance/CI |
| m-12: AI disclosure в conclusion | Отдельный AI Use Statement перед references |
| m-13: нештатная сноска AAAI | Официальный неизменённый AISTATS submission style |
| m-14: вклад — лишь свойство controlled topology | Сформулирован конкретный наблюдаемый topology result, без причинного вывода о state capacity |

## Что из AI researcher нельзя переносить в статью как установленный факт

| Старый вывод / гипотеза | Повторная оценка |
|---|---|
| «У GRU глубина ломает SDQ, а Grid даёт градиенту обход» | Порядок SDQ результатов наблюдается; механизм обхода градиента не измерен и остаётся гипотезой |
| «Большое beta и низкая entropy означают исчезновение градиента» | Это возможная saturation-гипотеза, но beta и усреднённая entropy не измеряют градиент через каждый route |
| «Рост col_sim до 0.2 означает ансамбль одинаковых столбцов» | Cosine batch-mean vectors не доказывает дублирование информации, тем более одинаковые функции столбцов; нужен conditional probe/intervention |
| «Низкий cosine доказывает специализацию» | Непохожие activation vectors не задают семантическую роль и не доказывают независимую полезную память |
| «Корреляция col_sim–BPC примерно 0.9 объясняет ухудшение» | В старом срезе только пять topology configurations; H, C, стоимость input route и readout меняются вместе. Корреляция описательна, причинного объяснения нет |
| «Линейные baseline не работают на SDQ» | Старые 125–250M budgets и меньшие batches не позволяют делать family-level conclusion |
| «Grid имеет recurrent цену и поэтому эффективнее Transformer» | Фиксированный state/step geometry — свойство архитектуры. Исторические loop rates ниже некоторых GRU/Transformer references; controlled deployment сравнение отсутствует |
| «Отсутствие skipped updates означает устойчивое обучение» | Это узкий diagnostic; MQAR gradient spikes и слабое RL quality не позволяют объявить проблему решённой |
| «LRU против GRU — одна заменённая клетка» | В tested Grid-LRU одновременно изменены recurrence, routing, projections и auxiliary terms; статья корректно сравнивает альтернативные архитектуры |

AI-разбор также содержит замечания к dead helper methods, `n_outputs`, округлению H и старым `mha=0/1` интерфейсам. По текущему коду dead helpers и общий `n_outputs` интерфейс остаются вопросами подготовки релиза; опубликованные GRU-прогоны используют `n_outputs=1`, округлённые кратные четырём H и `mha=2`. Эти замечания не доказывают некорректность измеренных запусков. Текущие `mha=0/1` классы уже имеют `n_q/n_kv`; старое сообщение о неизбежном TypeError нельзя повторять как актуальный факт без отдельной runtime-проверки. Код архитектуры в этой редакционной задаче не менялся.

## Что ещё улучшать: порядок по ценности для статьи

1. **Законченный state-matched GRU контроль** на том же позднем token budget и нескольких seed. Показать рядом parameter-matched и state-matched сравнения с честными counts; одним подбором H оба бюджета не выровнять.
2. **Held-out fixed-gap SDQ**, сначала evaluation-only при наличии подходящих checkpoints. Заранее выбрать набор gaps, генератор и aggregation; не выбирать лучший порог после просмотра результатов.
3. **Full-budget full/no-aux MoSAIC** на нескольких seed, затем отдельная input-cost/projection/readout ablation. Изменять один механизм при фиксированных widths и остальных настройках; сначала проверить GRU-версию основной статьи.
4. **Честный tuning/implementation comparison** с наиболее сильным Mamba и ближайшим модульным baseline. Добавление RMC полезно, но менее приоритетно, чем первые три причинные проверки.
5. **Измерения inference latency, throughput и peak memory** при фиксированных hardware/batch/dtype/context, отделённые от historical logger fps. Только после них можно обсуждать практический quality–cost trade-off.
6. **Anonymous reproducibility archive**, если авторы решат разрешить его до acceptance: configs, точные версии/commit, training/eval commands и license inventory. В текущей согласованной политике публичный release остаётся после принятия; никаких обещаний о наличии архива не добавлено.

До дедлайна полезнее завершить одну существенную причинную проверку, чем дополнять статью ещё несколькими одно-seed LRU-идеями. Запуски и внешняя инфраструктура в этой задаче не использовались. Незапущенные эксперименты не внесены как результаты.

## Дедлайн по Москве

По [официальному CFP AISTATS 2027](https://virtual.aistats.org/Conferences/2027/CallForPapers), full paper и все supplementary materials должны быть поданы **6 октября 2026, 23:59 AoE**. AoE — UTC−12, Москва — UTC+3; разница 15 часов. Следовательно, крайний срок — **7 октября 2026, 14:59 МСК**. Отдельного supplementary deadline нет. Конверсия дополнительно проверена через `datetime`/`ZoneInfo('Europe/Moscow')`.

Abstract deadline был 29 сентября 23:59 AoE, то есть 30 сентября 14:59 МСК. Сохранённый план пересдачи отмечает подтверждение его подачи пользователем. Внешний OpenReview статус здесь не проверялся. Название/список авторов не менялись; существенные изменения зарегистрированного abstract после его дедлайна ограничены правилами конференции.

## Проверка результата

После правок PDF сохраняет **8 страниц основного текста**. Полный объём — **21 страница**: main 1–8, AI Use Statement/references 9–10, checklist 11, appendices 12–21. Дополнительная страница относится к приложениям и не нарушает лимит. Нет неопределённых references/citations; все три рисунка имеют ссылки из текста. Числовая evidence не менялась, включая 18 main baseline groups, 41 основной новый запуск и 124 ранее выполненные pilot scalar checks. Официальные style-файлы сохранены, metadata анонимны, Type 3 fonts отсутствуют. Результат текущей проверки записан в [validation JSON](docs/experiments/aistats2027_revision_validation.json).
