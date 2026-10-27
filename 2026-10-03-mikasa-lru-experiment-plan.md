# Mikasa / POPGym: результаты, аудит PPO и план Grid-LRU

Дата: 03.10.2026. Проверены [run_mlspace.yaml](aicenter3/run_mlspace.yaml), текущий [runner](knitwork/exps/mikasa/run.py), [общий PPO](knitwork/rl/on_policy.py), [RL-обёртки](knitwork/models/wrappers.py) и read-only данные Comet. **Активная очередь сокращена до ровно десяти запусков, включая baseline, без дополнительных sweep и повторов seeds.** Ниже приведены отдельные конфиги и команды; длительное обучение и MLspace-задания в ходе подготовки не запускались.

## Что действительно указано в run_mlspace.yaml

Файл находится в **aicenter3/**, а не в корне. Это отдельная копия проекта. Все три строки Mikasa закомментированы; активная `container.command` запускает **Text8 Grid-GRU L3C4, 2B tokens, rollout128**. `cloud.description: mikasa 0` не определяет запускаемый эксперимент.

| env-id | Среда из adjacent large.yaml | Модели в строке | Бюджет на запуск | Количество команд |
| ---: | --- | --- | ---: | ---: |
| 0 | RepeatFirstEasy | GRU L2; Grid-GRU L2C4 | 30M | 2 |
| 7 | HigherLowerMedium | GRU L2; Grid-GRU L2C4 | 30M | 2 |
| 1 | RepeatFirstMedium | GRU L2; Grid-GRU L2C4 | 40M | 6, три повторения пары |

Повторные команды не задают разные seeds: в large.yaml `seed: null`. Протокол файла: 512 envs × rollout32 = 16,384 vector slots/rollout, PPO4, RMSprop, gamma0.995, lambda0.95, clip0.1, entropy0.005; предусмотрены Grid-GRU noise0.02 и auxiliary weights 1e−5/1e−6. **Grid-LRU в этой YAML не задан**, поэтому простая подстановка `--model=grnn_lru.L2C4` не достаточна.

Зарегистрированный исторический протокол отличается от нынешней large.yaml: LR5e−4, warmup50×24 rollout updates, Grid-GRU noise0.05 и communication/entropy weights0.05/0.005. Не считать команды файла точной репродукцией старых результатов только по названиям задач.

Новые команды ниже выполняются **из корня текущего репозитория**. Исправления внесены в корневой `knitwork/`; копия `aicenter3/knitwork/` не обновлялась. Для MLspace подготовлен отдельный [run_mlspace_mikasa_lru.yaml](run_mlspace_mikasa_lru.yaml): ровно десять последовательных запусков из новой очереди `rl_shortlist_mid`. Его нужно запускать с рабочей копией обновлённого корня. Существующий Text8 job-файл не переключался.

## Есть ли результаты

Срез [knitwork-mikasa](https://www.comet.com/team-rl-exp/knitwork-mikasa/view/new/panels) сохранён в [JSON](docs/experiments/mikasa_2026-10-03_snapshot.json). В проверенном проекте **13 запусков**, все остановлены и дошли до примерно 30,015,488 vector slots. Это те же исторические запуски, что в июльском [results_aaai.md](docs/experiments/results_aaai.md); новых Grid-LRU прогонов в данном проекте нет.

Для каждого запуска усреднены последние пять фактических `env/EpRet` точек, затем рассчитаны mean ± sample standard deviation между запусками. Это **онлайн training return**, не независимая evaluation. Seeds были null, поэтому n обозначает отдельные запуски, а не воспроизводимые фиксированные seed-пары.

| Среда | Модель | n | Final-five online EpRet | Средний peak, только диагностика |
| --- | --- | ---: | ---: | ---: |
| RepeatFirstEasy | GRU L2, около10M weights | 3 | 0.9962 ± 0.0037 | 0.9997 |
| RepeatFirstEasy | Grid-GRU L2C4, около10M weights | 4 | 0.6921 ± 0.2428 | 0.9962 |
| HigherLowerMedium | GRU L2 | 3 | 0.2968 ± 0.0363 | 0.4898 |
| HigherLowerMedium | Grid-GRU L2C4 | 3 | 0.3577 ± 0.0718 | 0.3969 |

**Вывод:** RepeatFirstEasy выявляет деградацию Grid-GRU после высоких peaks; GRU держится около единицы. HigherLowerMedium даёт небольшой и шумный перевес Grid-GRU в конце, хотя GRU имела более высокие peaks. Это аргумент проверять устойчивость, а не выбирать лучший случайный checkpoint. RepeatFirstMedium из MLspace-команд в проверенном проекте не найден. Доказательства превосходства нового Grid-LRU в RL пока отсутствуют.

В старых июньских заметках `mikasa.md`/`mikasa_exp.md` описаны неудачные короткие прогоны legacy `grnn_lru` через `run_mikasa.py`. У legacy config logger по умолчанию Aim; при переключении на Comet helper жёстко пишет в общий проект `knitwork`, независимо от log.project. Дополнительно проверены env/exp параметры86 запусков `team-rl-exp/knitwork` — Mikasa совпадений нет; локальный Aim metadata index, открытый SQLite mode=ro, также не дал Mikasa name matches. Эти проверки не покрывают другие аккаунты/удалённые Aim repos. Поэтому июньские LRU числа остаются заметками, а не обновлённой tracker evidence. Они не устанавливают причины нестабильности текущего `grnn_lru_core`: implementation, budget и logger protocol различались. Исходный commit не зарегистрирован у исторических запусков; найденные ниже проблемы текущего кода нельзя автоматически объявлять причинами старых кривых.

## Аудит реализации: что исправлено и проверено

| Проверка | Результат / изменение |
| --- | --- |
| Вход RLToken/RLVector | Исправлено: обёртки учитывают `core.batch_first`. Ранее `[1,B,H]` был несовместим с текущими Grid-LRU/plain-LRU, требующими `[B,1,H]`. |
| Recurrent PPO | Один backward по целому rollout на PPO-эпоху; состояние отсоединяется на границах rollout. Legacy `run_mikasa.py` делает шаг optimizer и detach после каждого timestep; для новых экспериментов он не используется. |
| Autoreset | Явно установлен NEXT_STEP, другие режимы sampler отклоняет. Reset-only слоты исключены из PPO/GAE; конечное наблюдение сначала оценивается со старым состоянием, затем состояние очищается перед новым эпизодом. Перестановка reset перед этим forward сломала бы truncation bootstrap. |
| GAE | Terminal не bootstrap-ится; truncation bootstrap-ится по финальному observation. Перенос GAE обрывается на обеих границах. Численные тесты проверяют обе ситуации и переход через границу rollout. |
| PPO replay | До optimizer step повторный проход воспроизводит action log-probs. Независимый routing noise/dropout нарушает это условие: активные stochastic layers теперь отклоняются явно. Exploration остаётся через Categorical actions. |
| Маски и loss | PPO losses усредняются по общему числу валидных переходов, вместо среднего per-timestep с неодинаковым числом валидных envs. Полностью reset-only batch безопасно пропускается. |
| Episode return | Завершённый return регистрируется на terminal/truncation transition, включая его награду; reset-only slot не увеличивает длину. Финальные завершённые эпизоды больше не ждут следующего rollout для регистрации. |
| Seeds | `torch.manual_seed` задаёт инициализацию и выбор действий; env RNG задаётся отдельно. Тест повторного запуска проверяет воспроизводимость на CPU. Это не гарантия побитовой CUDA-детерминированности. |
| После optimizer | Добавлен опциональный no-grad replay rollout для обновления переносимого состояния. Начальное состояние rollout всё ещё вычислено прежними весами; это уменьшает локальную stale-state проблему, но не заменяет replay всей истории эпизода. |
| Независимая оценка | Добавлены свежие fixed-seed evaluation episodes, отдельный state, изоляция torch RNG и восстановление train/eval mode. Их return логируется как `eval/EpRet`, отдельно от `env/EpRet`. |
| Наблюдения | Опциональный [PreviousActionReward](knitwork/env/mikasa_observations.py) даёт one-hot дискретные компоненты, предыдущее действие и reward; при reset всё очищается. Bandit/Battleship иначе не сообщали агенту, какое случайно выбранное действие дало feedback. |
| Spatial actions | Опциональный DiscreteCoordinates переводит Battleship MultiDiscrete coordinates в один Categorical index по той же bijective схеме, что POPGym DiscreteAction. Старый официальный wrapper требует голый POPGymEnv и несовместим с outer wrappers от gym.make; новый adapter сохраняет их. Проверены крайний индекс и observation shape. |
| Диагностика / ресурсы | Логируются валидные transitions и nominal vector slots отдельно; KL, clip fraction, skipped gradients, state RMS/max и pole radius. Дорогие CKA-визуализации в новых YAML выключены. Env/logger закрываются и при ошибке. |

NEXT_STEP semantics соответствуют [документации Farama](https://farama.org/Vector-Autoreset-Mode). Нормализация valid losses и seed фиксация меняют протокол, поэтому новые результаты получают metadata `mikasa_ppo_v2`. При старых nonzero communication weights core возвращает auxiliary scalar, уже усреднённый по envs, включая reset-only envs: полной per-env маски у него нет. Новая серия использует **оба auxiliary coefficients=0**, эта проблема на её objective не влияет.

Важно: отключение recurrent noise является обязательным для данного PPO replay. Нынешние legacy `base.yaml`/`large.yaml` Grid-GRU noise>0 больше не проходят silent replay с независимым шумом; для новых запусков использовать подготовленные YAML. Архитектурная абляция стохастического роутинга потребовала бы сохранения/replay RNG или явной marginal policy, что здесь не реализуется.

## Активная очередь: десять запусков вместе с baseline

Выбраны две дополняющие задачи: **RepeatPreviousHard** проверяет последовательную память с `k=64`, **AutoencodeEasy** — запоминание и обратное воспроизведение 52 карточных мастей. В установленном POPGym максимальная длина эпизода соответственно155 и103 шага. Сложность Easy у Autoencode выбрана ради достижимого сигнала при ограниченном бюджете; это полноценная sequence-memory задача с отдельной фазой воспроизведения. Оба семейства входят в memory-intensive POPGym сравнения [SHM, ICLR2025](https://github.com/DA2I2-SLM/SHM#-benchmarks), где также есть GRU, Mamba, mLSTM и FFM; опубликованные результаты используют другой training protocol.

Шесть сравнений на первой задаче отделяют тип ячейки, наличие столбцов, топологию коммуникации и второе чтение. На второй повторяются четыре наиболее полезных участника; dense Grid-LRU и Grid-GRU там не включены, поэтому отдельного вывода о dense→star на Autoencode эта очередь не даст. Все YAML самостоятельные и находятся в [rl_shortlist_mid](knitwork/exps/mikasa/config/rl_shortlist_mid).

| № | Среда | Модель / конфиг | Что узнаём |
| ---: | --- | --- | --- |
| 1 | RepeatPreviousHard | [GRU](knitwork/exps/mikasa/config/rl_shortlist_mid/01_RepeatPreviousHard_gru.yaml) | Сильный обычный recurrent baseline. |
| 2 | RepeatPreviousHard | [Plain LRU](knitwork/exps/mikasa/config/rl_shortlist_mid/02_RepeatPreviousHard_lru.yaml) | Что умеет линейная память без столбцов и роутинга. |
| 3 | RepeatPreviousHard | [Grid-GRU](knitwork/exps/mikasa/config/rl_shortlist_mid/03_RepeatPreviousHard_grnn_gru.yaml) | Контроль той же L2C4 организации с GRU-ячейками. |
| 4 | RepeatPreviousHard | [Dense Grid-LRU](knitwork/exps/mikasa/config/rl_shortlist_mid/04_RepeatPreviousHard_grnn_lru.yaml) | Исходная столбцовая LRU; контроль для изменения коммуникации. |
| 5 | RepeatPreviousHard | [Dynamic hub](knitwork/exps/mikasa/config/rl_shortlist_mid/05_RepeatPreviousHard_dynamic_hub.yaml) | Помогает ли разреженная star-коммуникация с адаптивным чтением. |
| 6 | RepeatPreviousHard | [Two reads](knitwork/exps/mikasa/config/rl_shortlist_mid/06_RepeatPreviousHard_two_reads.yaml) | Улучшает ли второе чтение обновлённых столбцов извлечение памяти. |
| 7 | AutoencodeEasy | [GRU](knitwork/exps/mikasa/config/rl_shortlist_mid/07_AutoencodeEasy_gru.yaml) | Baseline на обратном воспроизведении последовательности. |
| 8 | AutoencodeEasy | [Plain LRU](knitwork/exps/mikasa/config/rl_shortlist_mid/08_AutoencodeEasy_lru.yaml) | Достаточна ли одна линейная recurrent память. |
| 9 | AutoencodeEasy | [Dynamic hub](knitwork/exps/mikasa/config/rl_shortlist_mid/09_AutoencodeEasy_dynamic_hub.yaml) | Даёт ли распределённое хранение с одним выходным столбцом преимущество на этой задаче. |
| 10 | AutoencodeEasy | [Two reads](knitwork/exps/mikasa/config/rl_shortlist_mid/10_AutoencodeEasy_two_reads.yaml) | Переносится ли польза второго чтения на последовательный доступ в обратном порядке. |

**Главный кандидат на улучшение — two_reads; основной архитектурный контроль — dynamic_hub.** В [Text8-срезе](2026-10-03-lru-comet-analysis.md) two_reads улучшил BPC относительно dynamic_control примерно на0.027 в общем checkpoint280M, с преимуществом на всех11 общих точках после100M. Это один training seed на другой задаче, поэтому перенос в RL — гипотеза. Dynamic hub выбран как более простая разреженная архитектура, способная уменьшать взаимное вмешательство столбцов; это также ожидаемый механизм, не доказанная RL-выгода. Второе чтение не добавляет весов относительно matched hub-control, но добавляет вычисления, в прежнем разборе примерно25% cell-projection arithmetic. Полезная информация и actor/value по-прежнему выходят через **первый столбец**; auxiliary losses не добавляются.

Сравнение dense→dynamic одновременно меняет топологию и adaptive readout, поэтому оно оценивает весь вариант коммуникации, а не изолированный эффект разреженности. Plain LRU→Grid-LRU и GRU→Grid-GRU при близком числе весов также меняют размер и организацию state. Их результаты показывают пользу архитектурного пакета при таком parameter budget, без утверждения о строгом state-matched выигрыше.

RepeatFirstEasy исключён из активной очереди: исторический GRU уже почти решал его, а previous action/reward создают путь восстановить cue через feedback. HigherLowerMedium имеет сильную стратегию по текущему наблюдению. Battleship/Concentration, Mamba, RIMs/BRIMs, остальные matrix-memory модели, отдельные stability-ablation и rollout512 sweep отложены, чтобы сохранить лимит. Существующие [30 конфигов lru_mid](knitwork/exps/mikasa/config/lru_mid) сохранены как библиотека; **ни один не запускается сверх десяти выбранных**.

## Общий протокол и стабильность

Все десять запусков используют один protocol: seed42, примерно1M weights, float32, **32 среды × rollout128 =4096 vector slots/update**, PPO2, gamma0.995, **GAE lambda0.99**, LR2e−4→1e−4 с прежним64-update warmup, gradient clip0.5, PPO clip0.1, entropy0.005, target KL0.02, state refresh, small actor-head initialization и RMS-нормализацию выхода перед actor/value heads. Previous action/reward доступны всем участникам; recurrent noise и оба auxiliary coefficients равны нулю. Feedback normalization включена у Grid-LRU вариантов, а bottleneck первого столбца сохранён.

По отношению к старым pilot YAML общий rollout увеличен64→128 при уменьшении envs64→32, чтобы BPTT мог охватить задержку `k=64` и весь103-шаговый Autoencode эпизод, хотя часть эпизодов пересекает границы rollout. State переносится между rollout, gradient отсоединяется на их границе. **PPO4→2** сокращает максимум optimizer updates вдвое при том же interaction budget и уменьшает повторное смещение policy на одном rollout. **Lambda0.95→0.99** расширяет характерную шкалу GAE weights примерно18→66 шагов, помогая delayed credit, но увеличивает variance. Эти настройки одинаковы у baseline и вариантов; это общий экономный старт, не отдельная абляция и не доказанное лечение collapse.

`radius=exp(-exp(log_r))` ограничивает собственные значения отдельной LRU-ячейки, но не гарантирует стабильность всей сети с trainable projections и обратной связью. Поэтому оценивать одновременно `eval/EpRet`, entropy, KL, clip fraction, skipped gradients и raw hidden RMS/max. Нормализация readout может скрыть рост raw state; диагностика состояния сохранена. KL-stop при KL>0.03 останавливает последующие PPO-эпохи, но не отменяет уже сделанный шаг. Refresh обновляет state только replay последнего rollout; его начальное состояние всё ещё получено прежними весами.

Полные параметры actor-critic на выбранных задачах измерены отдельно и записаны в [JSON](docs/experiments/mikasa_rl_shortlist_sizes.json). Это близкие mid-размеры, а не точное равенство веса/state/compute. Dynamic hub и two_reads имеют одинаковое число весов на одной задаче. GPU throughput/peak и wall time нужно брать из новых запусков; CPU parameter count не даёт этих величин.

## Бюджет и команды запуска

**Один эксперимент здесь — один training run модели на одной среде с seed42. Ровно десять, включая все baseline.** Стандартный бюджет каждого —8,388,608 vector slots, сумма83,886,080. Каждый run имеет2048 rollout updates и максимум4096 optimizer updates; KL-stop может снизить вторую цифру. Warmup262,144 slots одинаковый для всех. Slots включают NEXT_STEP reset-only шаги; фактические interactions отдельно логируются в `env/Transitions`.

Для минимального screening можно **сразу выбрать2,097,152 slots/run**, суммарно20,971,520:512 rollout updates и максимум1024 optimizer updates на модель. Это проверка ранней обучаемости и стабильности, особенно недостаточная для отрицательного вывода на RepeatPreviousHard. **Выбрать одну из двух команд до старта:** runner не поддерживает resume, поэтому последовательный запуск2M и8M создаст двадцать runs и нарушит лимит.

Из корня репозитория:

```bash
# Основная очередь: ровно десять запусков по8.39M slots.
bash knitwork/exps/mikasa/run_rl_shortlist.sh
```

Или вместо основной очереди:

```bash
# Минимальная очередь: те же десять запусков по2.10M slots.
KNITWORK_RL_STEPS=2097152 bash knitwork/exps/mikasa/run_rl_shortlist.sh
```

Скрипт проверяет наличие ровно десяти YAML, запускает их по порядку и останавливается при ошибке. Для другого GPU задаётся `KNITWORK_RL_DEVICE=cuda:1`. Budget в имени каждого Comet run соответствует фактическому выбранному значению. Каждая отдельная конфигурация также запускается напрямую:

```bash
uv run python -m knitwork.exps.mikasa.run   knitwork/exps/mikasa/config/rl_shortlist_mid/06_RepeatPreviousHard_two_reads.yaml   --device=cuda:0
```

Последняя команда — способ запустить выбранного участника отдельно, **не одиннадцатый эксперимент**. Для MLspace [run_mlspace_mikasa_lru.yaml](run_mlspace_mikasa_lru.yaml) вызывает этот же десятиэлементный скрипт. Для2M бюджета добавить `KNITWORK_RL_STEPS: "2097152"` в `container.environment`. Запускать job из обновлённого корня, а не из `aicenter3`. Задание на инфраструктуру в ходе подготовки не отправлялось.

Оценка проводится каждые524,288 slots на32 свежих эпизодах, с отдельным seed50000 и изоляцией training RNG. На8M это16 событий, на2M —4. Для решения по результатам смотреть кривые и среднее последних четырёх `eval/EpRet` событий при одном бюджете, отдельно на каждой задаче; лучший peak не считать основным результатом. Проверять сохранение качества после первого высокого return и сопоставлять фактические transitions, не только slots. Evaluation overhead не включён в training budget.

При одном training seed это содержательный screening архитектур, а не статистическое подтверждение превосходства: `eval/EpRetStd` характеризует эпизоды, не variability между seeds. Дополнительные seeds и расширение benchmark сейчас **не входят** в утверждённую очередь. Runner не сохраняет checkpoint и не поддерживает восстановление interrupted expensive jobs; post-training held-out checkpoint evaluation здесь не обещается.

## Проверки и границы готовности

Предыдущая проверка RL-реализации: **73 теста общей серии прошли**, включая10 RL-тестов из [test_mikasa_ppo.py](tests/test_mikasa_ppo.py). Все12 ранее подготовленных model families прошли CPU smoke на256 slots с recurrent PPO backward; Grid-LRU также прошёл короткий CPU torch.compile/Inductor smoke. Эти проверки относятся к реализации, не являются результатами нового десятиэлементного RL-плана.

Для сокращённой очереди проверяются YAML, число и уникальность запусков, единый protocol, budget arithmetic, построение всех десяти полных actor-critic моделей на соответствующих средах, finite CPU forward, синтаксис скрипта и связь MLspace с этой очередью. Качество, GPU peak/скорость и устойчивость на миллионах interactions пока не измерены.
