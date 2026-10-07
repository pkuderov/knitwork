# AISTATS 2027: правки текста статьи и оформление — без новых экспериментов

**Дата:** 2026-10-01 · **Статус:** отчёт-план, изменения в статью не вносились.
**Пара к этому файлу:** `2026-10-01-aistats-required-experiments.md` — там всё,
что требует новых прогонов или пересчёта уже проведённых экспериментов. Этот
файл — только то, что можно сделать прямо сейчас, без GPU и без обращения к
Comet за новыми числами: формулировки, ссылки, оформление под шаблон AISTATS.

Источник исходного анализа: `2026-09-30-aistats-resubmission-plan.md` (полный
комбинированный план, из которого разделён этот файл и его пара) —
`2026-08-10-mosaic-full-audit.md`, `docs/reviews/grnn_core_analysis.md`, три
Official Review + AI Review AAAI-27, AISTATS 2027 CFP/Reviewer Guidelines.

**⚠️ См. `required-experiments.md` §9** — переписка команды (01.10.2026)
ставит под вопрос, нужен ли ablation регуляторов (claim 2 ниже) в текущем
виде, и добавляет новый эксперимент (E5, state-matched control) и попытку
RIM/BRIM-аналога (E3-ext). Пока решение по §9 не закрыто, формулировки для
claim 2 в этом файле — рабочий вариант, может потребовать правки.

---

## 0. Как читать этот файл

Всё здесь не зависит от того, что происходит в `required-experiments.md`:
можно начинать прямо сейчас, параллельно с постановкой любых прогонов, и
закончить независимо от их результата. Там, где правка текста логически
связана с экспериментом из второго файла (например, M-1 ablation или C-1
пересчёт бюджета токенов), явно указано, какую *временную* формулировку
использовать сейчас и что в ней придётся поправить, когда числа появятся.

**Оценка времени на всё в этом файле: 8–14 часов**, без какой-либо
зависимости от доступности GPU.

---

## 1. Соответствие претензий AAAI-27 ↔ zero-compute ответ

Из 10 claims четырёх рецензентов (u43j, S6rY, Dpwf, AI-Reviewer) шесть
закрываются полностью текстом; четыре (claims 2, 3, 8, 9 — ablation
регуляторов, top-k routing, Acc++ held-out метрика, token budget) требуют
данных и разобраны в `required-experiments.md`. Для них ниже дан только
**временный текст-заполнитель**, который можно использовать, если
эксперимент не успеет выполниться к дедлайну.

| # | Претензия | Тип ответа | Где именно |
|---|---|---|---|
| 1 | Persistent-state size confounds parameter matching | **Zero-compute** (Table 3 reframe, §3 ниже) **+ экспериментальный** (state-matched GRU control, обновление 01.10 — см. §2 врезку и `required-experiments.md` эксперимент E5) | §2, §3 ниже |
| 2 | No component ablations (noise/comm/entropy) | Экспериментальный (M-1) | → `required-experiments.md`, эксперимент E1. Fallback-текст, если не успеет: см. §3 ниже |
| 3 | No comparison with RIMs/BRIMs/RMC | Экспериментальный (E3 routing sparsity + опционально E3-ext module-update sparsity, обновление 01.10 — команда намерена попытаться) **+ zero-compute** (BRIM structural-similarity note) | → `required-experiments.md` §4, §5.2. BRIM-заметка и fallback-текст готовы уже сейчас — см. §3 ниже |
| 4 | Modern baselines at non-comparable budgets | **Zero-compute**: убрать риторику сравнения из Related Work | §3 ниже |
| 5 | text8 val split used for tuning | **Zero-compute**: одна явная фраза | §3 ниже |
| 6 | Narrow empirical coverage (1 synth task + text8, 1 scale) | **Zero-compute**: сузить формулировку claim | §3, §6 ниже |
| 7 | No throughput/latency/memory numbers | **Zero-compute**: числа уже логируются, просто вынести в текст | §3 ниже |
| 8 | Acc++ floating-threshold metric | Экспериментальный (C-2) | → `required-experiments.md`, эксперимент E2. Fallback-текст готов уже сейчас — см. §3 ниже |
| 9 | Token budget mismatch (MoSAIC vs GRU) | Требует пересчёта уже готовых прогонов, не новый эксперимент, но и не чистый текст | → `required-experiments.md`, задача D1 (не прогон, а скрипт). Временная формулировка — см. §3 ниже |
| 10 | Missing novelty citations (Goyal 2021a, Rahaman 2021) | **Zero-compute**: добавить 2 ссылки + разграничение | §3 ниже |

---

## 2. Разбор претензий 1, 4, 5, 6, 7, 10 — почему они обоснованы

(Для претензий 2, 3, 8, 9 объяснение механизма см. в
`required-experiments.md` рядом с соответствующим экспериментом — там же
разбор, зачем нужны именно такие гиперпараметры.)

### (1) Persistent-state size confounds parameter matching

**Кто сказал.** Dpwf прямее всех: «MoSAIC stores substantially more
persistent-state values… For example, L2C4 has 3,392 state scalars, compared
with 1,824 for GRU-L2. Without state-matched or compute-matched controls, the
gains cannot be clearly attributed to modular organization or learned
communication». S6rY и AI-Reviewer — мягче, u43j — как общий пункт
«parameter matching leaves persistent state and routing compute unmatched».

**Почему обоснованно.** Параметрический матчинг (~10M весов у обоих
семейств) — это матчинг *ёмкости для обучения*, не *ёмкости памяти во время
инференса*. GRU-L2 хранит `L·H = 2·912 = 1824` скаляров состояния; MoSAIC-L2C4
— `L·C·H = 2·4·424 = 3392`. Простая альтернативная гипотеза «выигрыш не от
маршрутизации, а просто от бо́льшего state» логически совместима с текущими
данными и не опровергнута явно ни одним экспериментом в статье.

**Zero-compute ответ.** Table 3 статьи уже содержит данные, которые
опровергают эту альтернативную гипотезу — просто не поданные как
контраргумент. L2C16 имеет **вдвое больше** state scalars, чем L2C4 (7168
против 3392), но *хуже* по BPC. Если «больше state → лучше», L2C16 обязан
быть лучшим конфигом MoSAIC, а он худший среди L2-вариантов. Это нужно
явно поднять из Table 3 в Discussion как прямой контраргумент, а не оставлять
читателю самому сопоставлять две таблицы. Готовая формулировка:

> If routing gains were merely a function of additional persistent state,
> widening the grid at fixed depth should monotonically help. L2C16 instead
> has the largest state (7,168 scalars, 2.1× L2C4) and the worst text8 BPC
> among the L2 shapes evaluated — state volume alone does not explain the
> ordering of results.

**Обновление (01.10.2026, по итогам переписки Petr/Владимира — см.
`required-experiments.md` §9).** Этот claim теперь имеет и экспериментальный
ответ: **E5** (`required-experiments.md` §5.1) — GRU-baseline с
`hidden_size`, подобранным так, чтобы `L·H` совпало с `L·C·H` у MoSAIC
(state-matched control, в отличие от текущего param-matched). Эту строку
добавить в Table 3 как явно помеченный дополнительный контроль
(«state-matched, ~X× параметров») рядом с уже готовым zero-compute
контраргументом выше — они дополняют друг друга, не заменяют.

**Вторая, независимая линия аргументации (опционально, в дополнение, не
вместо E5).** В той же переписке высказано резонное сомнение в самой
валидности state-matching как критерия (Petr: сравнение по размеру стейта
само по себе может быть не физически обоснованным — ограничения обычно на
память и скорость, а не на число скаляров состояния как таковое). Это можно
явно использовать в Discussion как дополнительный довод **после** приведения
числового контроля E5, не вместо него: «we additionally report training/
inference throughput and per-step memory (see §Throughput) as a more
direct resource-constraint metric than raw state-scalar count, since the
latter does not by itself determine either memory or compute cost at
deployment». Это связывает claim 1 с уже запланированным zero-compute
пунктом про throughput (claim 7, п.11 в §3 ниже) — один абзац, две
взаимоусиливающие линии защиты.

### (4) Modern recurrent baselines at non-comparable budgets

**Кто сказал.** Все четыре отзыва. u43j: «the reduced-budget runs cannot
support a quality comparison». AI-Reviewer — самая детальная формулировка.

**Почему обоснованно.** HGRN2/DeltaNet/mLSTM в статье — на 100-250M токенов
и 32-128 потоков против 1B/512 у MoSAIC/GRU/Transformer (4-10× меньше
обучающих токенов). На SDQ они не сдвинулись выше уровня случайного
угадывания — но статья не может отличить «архитектура не подходит» от
«не успела обучиться» при таком расхождении бюджетов. Проблема в том, что
совместная таблица с MoSAIC создаёт визуальное сравнение, даже когда текст
его отрицает.

**Zero-compute ответ.** Полная переоценка на сопоставимом бюджете — это
новый эксперимент (`required-experiments.md`, Приоритет C — там же explicit
обоснование, почему это НЕ делается за отведённое время). Здесь, в
zero-compute треке, — только текстовая часть: убрать риторические
противопоставления в Related Work там, где рядом нет чисел («not sparse
expert selection», сравнения с HGRN2/DeltaNet/mLSTM по духу архитектуры), и
усилить формулировку таблицы: явно пометить эти три строки как
«non-comparable reduced-token context (100-250M tokens vs 1B for all other
rows); not used for any comparative conclusion in this paper» — не просто
оговорка в прозе, а пометка прямо в подписи к таблице, которую рецензент не
пропустит.

### (5) text8 validation split used for tuning, not held out

**Кто сказал.** S6rY: «the text8 validation split was also used for some
manual configuration tuning further weakens the evaluation». Dpwf — то же.
AI-Reviewer подтверждает.

**Почему обоснованно.** Стандартный протокол text8 — 90M/5M/5M
(train/val/test), литературные числа — по untouched 5M test. Статья
использует 90M/10M (val используется и для отчёта, и для подбора
гиперпараметров) — классическая утечка: число, которое видел разработчик при
принятии решений, репортится как итоговый результат.

**Zero-compute ответ.** Одна явная фраза в «Data and training»:

> Because we use a 10M validation split rather than the conventional 5M/5M
> validation/test split, and because this split was also used for manual
> configuration decisions, our BPC values are not directly comparable to
> published text8 numbers; all comparisons in this paper are internal.

### (6) Narrow empirical coverage: one synthetic task + text8 at one scale

**Кто сказал.** Все четыре отзыва, стержневая претензия. u43j: «Evidence is
limited to one synthetic task and text8 at a single small scale, so
generality to realistic sequence tasks is unclear».

**Почему обоснованно.** Результат на двух задачах при ~10M параметров — это
данные о конкретной точке (task, scale); заявление в Introduction звучит как
обобщение на класс задач. Разрыв между объёмом evidence и объёмом claim —
то, что рецензенты почти всегда наказывают, независимо от venue.

**Zero-compute ответ.** Это единственная из 10 претензий, которую
**невозможно** закрыть новыми данными в разумный срок (нужен масштаб на
порядок больше и/или другие задачи). Единственный инструмент — явно сузить
формулировку claim до объёма реально предъявленных данных: заменить широкие
обобщающие фразы («useful inductive bias for recurrent computation») на
явно ограниченные («at this parameter scale (~10M), on these two tasks»).
См. также §6 (позиционирование) — это переформулирование тесно связано с
общей стратегией подачи под AISTATS.

### (7) No throughput/latency/memory trade-off numbers

**Кто сказал.** u43j прямо просит в Specific Points of Feedback for
Rebuttal: «Please clarify any available training/inference FLOP, throughput,
latency, and peak-memory comparisons for the matched models». Dpwf и
AI-Reviewer — то же через «no throughput, memory, or inference-latency
measurements».

**Почему обоснованно.** Статья утверждает архитектурное свойство
(«fixed-size recurrent state… supports incremental inference») как
потенциальное преимущество, но не подтверждает его числом.

**Zero-compute ответ — самый дёшевый во всём списке.** `perf/fps` **уже
логируется в Comet каждым запуском** (`knitwork/exps/sdq/run.py:295`,
`scalars["perf/fps"]`) для всех уже завершённых прогонов — значит это не
новый эксперимент, а просто выгрузка уже существующих чисел. Формулы для
памяти на шаг уже выведены в статье («Fixed-State and Resource Accounting»:
`6LCH²`/`4LH²` параметров, `LCH` скаляров состояния). Нужно: взять `fps` из
уже существующих Comet-логов (той же выгрузкой, что и для задачи D1 в
`required-experiments.md` — можно сделать одним скриптом) и собрать
компактную таблицу fps/память по семействам. Это ответ на прямой вопрос
рецензента — не ответить означает почти наверняка получить тот же вопрос
повторно.

### (10) Missing novelty citations (Goyal 2021a, Rahaman 2021)

**Кто сказал.** Только AI-Reviewer, с точными библиографическими данными:
Goyal et al. 2021a (ICLR, «Factorizing declarative and procedural
knowledge...») и Rahaman et al. 2021 (ICLR, «Spatially structured recurrent
modules»). Человеческие рецензенты не называют эти конкретные работы, но
Dpwf и S6rY оба высказывают общую претензию о недостаточной проработке
Related Work.

**Почему обоснованно.** Обе работы структурно близки (модульные состояния с
обучаемой коммуникацией / топологически структурированные рекуррентные
модули) и отсутствуют в списке литературы — риск для claim о новизне:
рецензент, который их знает (вероятно на AISTATS), встретит непроцитированный
близкий прайм-арт как сигнал недостаточно тщательного review.

**Zero-compute ответ.** Добавить обе ссылки в Related Work с 1-2
предложениями разграничения (dense synchronous updates у MoSAIC vs
конкретный механизм в каждой из двух работ) — **перед формулировкой
прочитать оба абстракта**, не полагаться только на пересказ AI-Reviewer.

---

## 3. Блок правок текста — полный чеклист (≈8–14 часов)

Ничего не пересчитывается, только переписывается текст `paper_en.typ` (или
его портированная версия в AISTATS-шаблон, см. §4). Порядок — по убыванию
влияния на первое впечатление рецензента.

1. **Abstract, тройное сравнение SDQ.** Заменить текущее сравнение на:
   `0.843 ± 0.006` (L2C4) vs `0.5465 ± 0.0768` (depth-matched GRU-L2,
   **каноническое число из `docs/experiments/results_aaai.md`, run-ID
   `442850b9`/`69d27694`/`6dfc8619` — см. исправленную врезку ниже**) vs
   `0.618 ± 0.005` (лучший GRU, `rnn_L1`). Не теряет силы аргумента, снимает
   claim «подбора удобного числа». **Числа финализировать только после**
   задачи D1 из `required-experiments.md` (пересчёт на общем token-budget) —
   возможно текущие mean/std чуть сдвинутся.
2. **Logging-loss convention + Δtokens.** Определить термин «logging-loss
   convention» прямо в тексте (сейчас введён и не раскрыт). Добавить явную
   колонку «Δ tokens vs baseline» в Table 1/2 с указанием знака расхождения.
   Точные числа колонки — выход задачи D1 из `required-experiments.md`,
   здесь — только разметка таблицы и формулировка.
3. **Figure 2 — добавить ссылку из текста.** Ни один абзац сейчас не
   ссылается на Figure 2. В конце «Results» для text8:
   > Figure 2(a) shows that the separation between MoSAIC and the GRU
   > baselines appears within the first 200M tokens and persists to the
   > horizon, so the final-point gap is not an artifact of where the runs
   > were stopped.
   Аналогичное предложение — для SDQ после Table 1.
4. **7 отсутствующих ссылок.** Shazeer et al. 2017, Fedus et al. 2021 (MoE —
   упомянуто дважды без ссылок), Arora et al. 2023, Olsson et al. 2022,
   Graves et al. 2014 (NTM), Su et al. 2021 (RoPE), Zhang & Sennrich 2019
   (RMSNorm), Elfwing et al. 2018 (SiLU), Tieleman & Hinton 2012 (RMSprop).
   Плюс Goyal et al. 2021a и Rahaman et al. 2021 (см. §2, claim 10).
5. **text8 split — одна фраза.** См. §2, claim 5, формулировка готова выше.
6. **«Self-Attentive» — убрать неоднозначность.** В Introduction явно
   оговорить, что attention идёт по банку из C+I сообщений, а не по token
   history — название иначе наводит на ложную аналогию с
   Transformer-attention.
7. **Дисциплина защитного тона.** Свести ~10 отказных формулировок к 3–4 в
   едином абзаце Limitations; убрать полемические фразы вроде «rather than
   using rhetoric to designate…».
8. **Availability statement.** Явное заявление: код будет опубликован (или
   уже опубликован — решить сейчас, формулировка зависит от факта),
   оборудование — **H100** (`aicenter3`, подтверждено текущей
   инфраструктурой проекта, см. ниже примечание об обновлении документации),
   сиды прогонов. Коллизия: во всех текущих прогонах `seed: null`, сиды не
   логировались (`docs/reviews/grnn_core_analysis.md` §4.5) — писать «seeds:
   not fixed, replicates differ only by launch-time randomness», честно
   ограничивая claim воспроизводимости до «same config, independent
   launches».
9. **Все минорные правки аудита (m-1..m-14).** Method-нотация (размерности
   проекций, индексация β по слою и колонке), унификация формулировок таблиц,
   «consistently improves» → «improves across all evaluated shapes (n = 3
   replicates, no significance testing)», перенос AI-disclosure в отдельный
   раздел, вклад №2 переформулировать как результат, а не свойство метода.
10. **Привилегия колонки 0 — явное объяснение.** Это не minor-правка, а
    готовое объяснение неожиданного результата (L2C16 хуже L2C4).
    Сформулировать явно: маршрут для входа стоит `2c`, при `c=0` бесплатно,
    при `c=15` дороже любого рекуррентного маршрута — и связать с
    наблюдением из `docs/reviews/grnn_core_analysis.md` (на C=16 колонки
    схлопываются, `col_sim/avg` растёт до 0.20, ранговая корреляция с BPC
    ρ≈0.9). Это единственное место, где аудит и `grnn_core_analysis.md`
    объединяются в тексте статьи — раньше не были связаны.
11. **Throughput/память — таблица.** См. §2, claim 7. Числа `fps` — из уже
    существующих Comet-логов (не новый эксперимент).
12. **Positioning (claim 1 и claim 6).** Внести готовые формулировки из §2
    (claim 1 — Table 3 anti-scaling контраргумент; claim 6 — сузить scope
    claim) непосредственно в Discussion/Introduction.
13. **Fallback-формулировки на случай, если эксперименты из
    `required-experiments.md` не успеют завершиться** (готовить параллельно,
    использовать только если к моменту финальной сборки PDF числа ещё не
    готовы):
    - **Claim 2 (M-1, нет ablation регуляторов).** Если E1 не завершится:
      «Component ablations for the routing noise, communication cost, and
      entropy terms were not completed within this submission's compute
      budget; we flag the relative contribution of these terms as an open
      question rather than making an unsupported claim either way.»
    - **Claim 3 (M-3, нет сравнения с RIMs/BRIMs).** Если E3 не завершится:
      «A controlled dense-vs-sparse routing ablation requires a routing-path
      change not exercised in the current codebase; we report this as a
      specific, actionable limitation rather than omitting the comparison.»
    - **Claim 8 (C-2, Acc++ метрика).** Если E2 не завершится: оставить
      текущее определение метрики, но явно описать её ограничение в тексте
      (не прятать): «Acc++ uses a batch-relative gap threshold that shifts
      with the training curriculum; we report it as a training-time
      diagnostic rather than a fixed-protocol evaluation, and leave a
      fixed-gap held-out evaluation for future work.»
    - **Claim 9 (D1, token budget).** Если D1 (скрипт пересчёта) не
      завершится: минимальный вариант — явно определить «logging-loss
      convention» в тексте (п.2 выше) без пересчёта таблиц. Хуже, чем
      пересчёт, но закрывает самую острую часть претензии («термин введён и
      не объяснён»).

### Врезка: рассинхрон чисел между источниками — что использовать (исправлено 01.10.2026)

**Предыдущая версия этой врезки была неверной** — она ошибочно отдавала
приоритет `grnn_core_analysis.md` (`0.5322 ± 0.1027`) и тексту `paper_en.typ`
(`0.532 ± 0.103`) над аудитом (`0.547 ± 0.077`). Проверено напрямую:
`docs/experiments/results_aaai.md` — замороженный снапшот от 2026-07-29,
используемый как единственный источник чисел в текущей AISTATS-ревизии
(`article/latex/paper.tex`, ветка `aistats-2027-text-format`) — даёт
`rnn/rnn_L2 = 0.5465 ± 0.0768` по тем же run-ID (`442850b9`, `69d27694`,
`6dfc8619`), которые фигурируют и в новом отчёте о ресурсах
(`docs/experiments/aistats2027_resources.md`). Это **ближе к числу аудита**
(0.547), не к `grnn_core_analysis.md`/старому `paper_en.typ` (0.532) —
значит именно последние два источника цитировали нерепрезентативную
выборку (вероятно, только часть из трёх реплик на момент своей записи).

**Источник истины — `docs/experiments/results_aaai.md`** (зафиксированный
когорт 2026-07-29, уже проверенный на согласованность с
`inference/comet_aaai_snapshot.py`), а не более ранние документы
(`grnn_core_analysis.md`, старый `paper_en.typ`) и не аудит напрямую (аудит
совпал с правильным числом случайно/приблизительно, не как источник).
`required-experiments.md` задача D1 (пересчёт на едином token-budget) — это
следующий шаг **после** `results_aaai.md`, не альтернатива ему.

---

## 4. Формат: AISTATS-специфичные требования

Это не про правильность результатов — про то, что может привести к desk
reject **до** того, как содержание кто-то прочитает.

1. **Шаблон.** Текущий PDF — в AAAI two-column camera-ready style. AISTATS
   использует свой LaTeX-шаблон, лимит **8 страниц основного текста**
   (references/AI Use Statement/reproducibility checklist/appendix не
   считаются — просторнее, чем 7 у AAAI). Скачать AISTATS 2027 Author
   Kit/style file и **перекомпилировать статью в него** — margins, column
   width, caption style у PMLR/AISTATS шаблона другие, портировать
   `article/latex/`/`article/typst/paper_en.typ`, не просто сменить
   titlepage.
2. **AI Use Statement — обязателен**, отдельно от Reproducibility Checklist;
   отсутствие = desk reject. У AAAI-версии декларация об ИИ вклинена в конец
   Conclusion (m-12 аудита) — вынести в отдельный раздел под требуемым
   AISTATS названием.
3. **Reproducibility Checklist** — заново заполнить под форму AISTATS (секции
   пересекаются с AAAI-27, но не идентичны — не переиспользовать бездумно).
4. **Анонимность.** PDF уже проверен аудитом (🟢 Pass — нет имён, email,
   путей, ссылок на репозитории). Условие переносится без изменений на
   AISTATS, но **перепроверить заново после всех текстовых правок** — правки
   могут случайно внести идентифицирующие детали (например, «as in our
   earlier technical report» вместо anonymized third-person).
5. **Multiple/simultaneous submission policy.** Работа отклонена на AAAI-27
   (Phase-1 rejection, нотификация 24.09.2026, уже прошла). Формально не
   «simultaneous submission». **Перечитать буквально** формулировку AISTATS
   Submission FAQ/Author Kit о ранее отклонённых версиях перед аплоадом, не
   полагаться на аналогию с AAAI.

**Оценка времени:** 0.5–1 день, не требует переосмысления содержания —
делать **параллельно** с §3 (правки текста сразу в целевом шаблоне, не
переносить дважды).

---

## 5. Позиционирование под AISTATS

AISTATS, по данным reviewer guidelines (2025/2026), ценит «novel problems»,
«out-of-the-box ideas», «long-term impact over incremental advances»,
принимает эмпирический вклад без теории как полноценный (в отличие от
восприятия «просто another architecture beats GRU baseline», которое звучит
инкрементальным). Рекомендация по переписыванию (не меняя ни одной цифры):

- **Introduction**: сместить акцент с «MoSAIC улучшает GRU» на явную
  постановку вопроса — «can a fixed recurrent parameter budget be organized
  into persistent modules with learned communication, and does this topology
  matter independently of state volume?» (формулировка уже почти буквально
  есть в тексте — усилить как topic sentence раздела).
- **Contribution #2** («controlled topology») переформулировать как
  найденный результат, а не свойство метода: «we show that the allocation of
  state across depth and column count affects outcomes more than its raw
  volume» — прямой удар в epicentre претензии о state-size confound,
  превращает слабость в заявленный, подтверждённый Table 3 результат.
- **Значимость через статистику, не через голые числа.** Использовать
  освободившийся объём страниц (8, не 7) на: (a) везде, где есть 3 реплики —
  явные 95%-доверительные интервалы, не только ±std; (b) там, где n=1 — не
  подавать как равноценную строку таблицы, визуально/текстуально отделить
  (отдельная пометка † и сноска); (c) явно объяснить дизайн эксперимента
  (matched compute, seeds not fixed) — раз нет теоретического компонента,
  весь вклад оценивается через качество экспериментального дизайна, и
  AISTATS-рецензент читает это строже среднего AAAI-рецензента, не мягче.

---

## 6. Открытые вопросы этого файла (не экспериментальные)

1. **Формулировка политики AISTATS про ранее отклонённые версии работы** —
   перечитать буквально в Submission FAQ/Author Kit перед аплоадом, не
   полагаться на аналогию с AAAI-политикой.
2. **Compute statement (availability statement, §3 п.8) — финализировать
   формулировку «H100 (aicenter3)»** после того, как подтверждено, что именно
   на этой инфраструктуре был выполнен весь пересчёт/ablation из
   `required-experiments.md` (если часть исторических чисел статьи считалась
   на другом железе до миграции документации на H100-только политику — это
   нужно отразить честно, а не задним числом приписать все числа H100).
3. **Решение о публикации кода** — зависит от решения автора, не технический
   вопрос; формулировка availability statement (§3 п.8) подстраивается под
   это решение, а не наоборот.
