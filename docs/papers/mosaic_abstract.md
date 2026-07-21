# MoSAIC — Abstract (AAAI, RU)

## Основной вариант (качественный, без цифр)

Секвенциальные модели вынуждены одновременно решать две конкурирующие
задачи: гибко связывать и удерживать ассоциации «ключ–значение» для
последующего точного извлечения и моделировать плавную статистику
естественного языка. В рекуррентных сетях с единым скрытым состоянием эти
функции перемешиваются в общей памяти, из-за чего новая информация
перезаписывает ранее связанные ассоциации, а специализация подзадач между
вычислительными единицами не возникает. При этом известно, что модульная
организация и разреженная коммуникация через внимание повышают обобщение и
устойчивость к возмущениям (Vaswani et al. 2017; Goyal et al. 2019). Мы
предлагаем MoSAIC (Modular Specialised Attentive Interacting Columns) —
рекуррентную архитектуру из параллельных колонок, каждая из которых обладает
защищённой собственной памятью и специализируется благодаря индивидуальным
входным проекциям, а также собственным идентичностям и резкости внимания.
Колонки обмениваются информацией скупо, через ассоциативное внимание
хопфилдовского типа (Ramsauer et al. 2020): сообщения подмешиваются через
обучаемые вентили и никогда не перезаписывают память соседей, что защищает
уже связанные ассоциации от интерференции. Иерархия закладывается в сам
процесс обучения: априор разных временных масштабов (быстрые и медленные
колонки), родственный иерархическому гейтингу (Qin et al. 2023) и
многомасштабным рекуррентным сетям (Koutník et al. 2014; Chung et al. 2017),
и растущая с глубиной цель декорреляции (Zbontar et al. 2021), которая
препятствует коллапсу колонок к общему представлению и индуцирует разделение
труда между ними. В результате обучения возникает выраженная функциональная
специализация колонок; на задачах ассоциативного извлечения (Arora et al.
2023) и языкового моделирования MoSAIC даёт устойчивые улучшения над сильными
рекуррентными и линейными attention-моделями и заметно повышает устойчивость
к дистракторам и зашумлённым входам.

## Русская версия (сжатая, ~180 слов)

Секвенциальные модели должны одновременно связывать и удерживать ассоциации
«ключ–значение» для точного извлечения и моделировать плавную статистику языка.
В рекуррентных сетях с единым скрытым состоянием эти функции перемешаны в общей
памяти: новые входы перезаписывают ранее связанные ассоциации, а специализация
между единицами не возникает. При этом модульность и разреженная коммуникация
через внимание, как известно, повышают обобщение и устойчивость (Vaswani et al.
2017; Goyal et al. 2019). Мы предлагаем MoSAIC (Modular Specialised Attentive
Interacting Columns) — рекуррентную архитектуру из параллельных колонок, каждая
из которых обладает защищённой собственной памятью и специализируется благодаря
индивидуальной входной проекции и идентичностям внимания. Колонки обмениваются
информацией скупо, через ассоциативное внимание хопфилдовского типа (Ramsauer et
al. 2020): сообщения подмешиваются через вентили и никогда не перезаписывают
память соседей. Иерархия заложена в сам процесс обучения — априор разных
временных масштабов по колонкам, родственный иерархическому гейтингу (Qin et al.
2023), и растущая с глубиной цель декорреляции (Zbontar et al. 2021), которая
предотвращает коллапс колонок и индуцирует разделение труда. Обучение порождает
выраженную функциональную специализацию; на задачах ассоциативного извлечения и
языкового моделирования MoSAIC устойчиво превосходит сильные рекуррентные и
линейные attention-модели и заметно устойчивее к дистракторам.

## English version (AAAI, condensed ~180 words)

Sequence models must both bind and retain key–value associations for exact
retrieval and model the smooth statistics of language. In recurrent networks
with a single hidden state these functions are entangled in one memory, so new
inputs overwrite previously bound associations and no specialisation emerges
across the units. Yet modularity and sparse attention-based communication are
known to improve generalisation and robustness (Vaswani et al. 2017; Goyal et
al. 2019). We introduce MoSAIC (Modular Specialised Attentive Interacting
Columns), a recurrent architecture of parallel columns, each with a protected
private memory and specialised through its own input projection and attention
identities. Columns communicate sparingly via Hopfield-style associative
attention (Ramsauer et al. 2020): messages are gated in and never overwrite a
neighbour's memory. Hierarchy is built into learning itself—a multi-timescale
prior over columns, akin to hierarchical gating (Qin et al. 2023), and a
depth-growing decorrelation objective (Zbontar et al. 2021) that prevents
column collapse and induces a division of labour. Training yields pronounced
functional specialisation; on associative-recall and language-modelling
benchmarks, MoSAIC reliably improves over strong recurrent and linear-attention
baselines and is markedly more robust to distractors.

## Предлагаемые названия (8–10), не ассоциирующиеся с Grid-RNN

1. **MoSAIC** — *Modular Specialised Attentive Interacting Columns* (основной; акроним прямо кодирует модульность + специализацию + внимание).
2. **CorticoNet** — *Cortical Column Recurrent Network* (нейробиологическая рамка кортикальных колонок и их специализации).
3. **MoRE** — *Modular Recurrent Experts* (рамка «разделения труда» / mixture-of-experts).
4. **CAN** — *Columnar Attention Network* (нейтрально-описательное, без метафор).
5. **NeCA** — *Neural Column Assembly* (ансамбль специализированных единиц).
6. **SReM** — *Specialised Recurrent Modules* (акцент на специализацию модулей).
7. **STRATA** — *STacked Recurrent Attentive Timescale Assemblies* (иерархия «страт» временных масштабов).
8. **PSRE** — *Parallel Specialised Recurrent Ensemble* (параллельный специализированный ансамбль).
9. **ChoRNN** / **Chorus** — метафора «хора» согласованных, но различных голосов-колонок.
10. **CONCERT** — *COlumnar Networks with Communicating Experts and Recurrent Timescales* (колонки + коммуникация + временные масштабы).

## Ключевые ссылки

- Vaswani et al. 2017 — attention (NeurIPS).
- Goyal et al. 2019 — Recurrent Independent Mechanisms (arXiv:1909.10893): модульность, разреженная коммуникация, специализация.
- Ramsauer et al. 2020 — Hopfield Networks is All You Need: ассоциативное внимание.
- Qin et al. 2023 — HGRN, иерархический гейтинг (NeurIPS spotlight; arXiv:2311.04823).
- Koutník et al. 2014 — Clockwork RNN; Chung et al. 2017 — Hierarchical Multiscale RNN: многомасштабность.
- Zbontar et al. 2021 — Barlow Twins: декорреляционная цель.
- Arora et al. 2023 — Zoology / MQAR (associative recall).
