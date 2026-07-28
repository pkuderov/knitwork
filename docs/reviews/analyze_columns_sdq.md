# Разбор: inference/analyze_columns_sdq.py

## Назначение

Скрипт **инференса и анализа** (не обучения). Берёт готовый чекпоинт Grid-RNN
(`grnn_fix_v4`), обученный на задаче **SDQ** (Store–Distract–Query), и количественно
измеряет **вклад отдельных колонок** сети в решение задачи. Гипотеза, которую он
проверяет: память локализуется в узкой подгруппе колонок вокруг колонки 0.

Вход — путь к `.pt`-чекпоинту (`--checkpoint`) и параметры анализа (`--n_collect`,
`--subgroup_k`, `--epochs`). Выход — метрики, бар-чарты и attention-flow, залогированные
в Comet (`--log.logger=comet`); в stdout печатаются ключевые числа. Модель и данные
скрипт **не изменяет** (веса заморожены, всё под `torch.no_grad`).

## Основные классы и функции

| Символ | Роль | Ключевые сигнатуры / shape'ы |
|---|---|---|
| `collect(rnn, gen, device, n_collect)` | Прогон SDQ в инференсе; собирает фичи верхнего слоя, таргеты и усреднённый по времени attention | → `feats [N,C,H]`, `tgts [N]`, `attn [L,C,C]` |
| `train_readout(feats, tgts, V, epochs, split)` | Линейный зонд на замороженных фичах → held-out SDQ accuracy | вход `[N,D]` → `{'Acc': float}` |
| `plot_bars(values, title, ylabel)` | Бар-чарт `dict[str,float]` для Comet | → `matplotlib.Figure` |
| `main()` | Загрузка чекпоинта + три анализа A/B/C + логирование | — |
| `StoreDistractQueryGenerator` (из `gens/sdq.py`) | Генератор потока SDQ-токенов; таргеты неотвечаемых шагов = `CE_ignore_index` | `.V` — размер словаря значений |
| `build_model` (из `exps/sdq/run_sdq.py`) | Пересоздаёт модель по `model_type` и config | — |
| `AttnFlowVisualizer` (из `visualization/attn_flow.py`) | Визуализация межколоночного потока внимания | буферы по слоям |

Форма скрытого состояния Grid-RNN: `[L, C, B, H]` — `L` слоёв × `C` колонок × батч ×
скрытая размерность. Колонки на каждом слое обмениваются информацией через attention.

## Ключевые идеи и математика

### 1. Attention как маршрутизация информации между колонками

`collect` запрашивает `return_attn=True` и получает от модели `extras['attn_weights']` —
матрицы внимания между колонками. Они усредняются по времени (и по головам, если
`ndim > 2`):

```python
aw = extras.get('attn_weights')
if aw:
    for li, a in enumerate(aw):
        if a is not None:
            m = a.detach().float().cpu().numpy()
            while m.ndim > 2:          # collapse heads/batch dims -> [C, C]
                m = m.mean(0)
            attn_acc[li] += m
    attn_cnt += 1
# ...
attn = np.stack([a / max(attn_cnt, 1) for a in attn_acc], axis=0)  # [L, C, C]
```

Результат `attn[l, i, j]` — усреднённая доля внимания, которое колонка *i* уделяет
колонке *j* на слое *l*:

$$\bar{A}^{(l)}_{ij} = \frac{1}{T}\sum_{t} A^{(l)}_{ij}(t), \qquad \bar A^{(l)}\in\mathbb{R}^{C\times C}$$

Взаимодействие колонки *c* с колонкой 0 — сумма входящего и исходящего внимания
(усреднённая ещё и по слоям):

$$\text{interaction}(c) = \underbrace{\overline{A}_{c,0}}_{c\text{ смотрит на }0} + \underbrace{\overline{A}_{0,c}}_{0\text{ смотрит на }c}$$

```python
to_c0   = {f'C{c}': float(attn[:, c, 0].mean()) for c in range(1, C)}   # c attends to 0
from_c0 = {f'C{c}': float(attn[:, 0, c].mean()) for c in range(1, C)}   # 0 attends to c
interaction = {f'C{c}': to_c0[f'C{c}'] + from_c0[f'C{c}'] for c in range(1, C)}
```

По этому рангу выбирается подгруппа `S = {0} ∪ top-(k-1)` колонок (`subgroup_k`).

### 2. Линейный зонд (readout probe)

`train_readout` замораживает backbone и учит **только** линейную голову предсказывать
SDQ-таргет из фич колонки. Скорятся лишь активные (не `ignore_index`) таргеты — те же,
что и в оригинальной SDQ-метрике:

```python
valid = tgts != CE_ignore_index          # score only answerable query steps
feats, tgts = feats[valid], tgts[valid]
# ... 70/30 split ...
head = nn.Linear(feats.shape[1], V).to(device)   # frozen backbone, train head only
opt = torch.optim.Adam(head.parameters(), lr=5e-3, weight_decay=1e-4)
for _ in range(epochs):                          # 300 epochs
    opt.zero_grad(); lossf(head(Xtr), Ytr).backward(); opt.step()
acc = (head(Xev).argmax(-1) == Yev).float().mean()   # held-out SDQ accuracy
```

Held-out accuracy — нижняя оценка **линейно-декодируемой** информации об ответе,
содержащейся в этих активациях. Высокая accuracy на колонке означает, что ответ в ней
лежит в линейно-читаемой форме. При `N < 32` зонд не обучается и возвращает `NaN`.

### 3. Ablation через маскирование attention (причинное вмешательство)

Строится булева маска, где взаимодействуют только колонки подгруппы `S`, плюс диагональ
(каждая колонка всегда видит себя):

```python
mask = torch.eye(C, dtype=torch.bool)
for i in subgroup:
    for j in subgroup:
        mask[i, j] = True
rnn.attn_col_mask = mask.to(device)   # honored only in grnn_fix_v4.py:181
feats_m, tgts_m, _ = collect(rnn, gen, device, args.n_collect)
rnn.attn_col_mask = None               # restore
```

Это **причинное** вмешательство: физически обрезаются пути внимания вне `S`, сеть
пересобирает состояния с нуля, после чего readout по подгруппе показывает, достаточно ли
ей внутренних связей без остальных колонок. Важно: `attn_col_mask` учитывается только в
`grnn_fix_v4` (`grnn_fix_v4.py:181`), поэтому скрипт заточен под эту модель.

## Пошаговый разбор пайплайна

1. **Загрузка окружения и чекпоинта.** `_load_dotenv()` подтягивает `COMET_API_KEY`
   (скрипт обходит стандартный `run_experiment`). `torch.load(..., weights_only=False)`
   даёт `config`, `model_type`, `model_state`, `step`.
2. **Пересборка модели и генератора.** `StoreDistractQueryGenerator(**gen_cfg, ...)` и
   `build_model(rnn_type, config['models'][rnn_type], gen)`, затем `load_state_dict` и
   перенос на устройство. `V, C = gen.V, rnn.n_columns`.
3. **Логгер.** Конфиг лога берётся из чекпоинта, поверх накладываются `--log.*`
   overrides; по умолчанию Comet, проект `grid-rnn-sdq`.
4. **Сбор (unmasked).** `collect` → `feats [N,C,H]` (верхний слой), таргеты,
   `attn [L,C,C]`.
5. **Анализ A — ранжирование.** Считает `interaction(c)` и норму активаций каждой
   колонки; логирует attention-flow (`AttnFlowVisualizer`), бар-чарты и скаляры.
   Выбирает подгруппу `S = {0} ∪ top-(k-1)` по interaction.
6. **Анализ C — per-column readout.** `train_readout` на каждой одиночной колонке, на
   всех колонках (`ALL`) и на подгруппе (`SUB`); бар-чарт по колонкам.
7. **Анализ B — masked ablation.** Ставит `attn_col_mask` подгруппы, пересобирает фичи,
   меряет readout изолированной подгруппы (`subgroup_masked`), снимает маску.
8. **Финализация.** `logger.end()`.

## Ожидаемые результаты

Артефакты в Comet: attention-flow по слоям; бар-чарты `ablation/interaction_c0`,
`ablation/activation_norm`, `ablation/readout_acc`; скаляры `ablation/readout_acc/*`,
`ablation/interaction_c0/*`, `ablation/act_norm/*`, `ablation/subgroup_size`.

Как читать (подтверждение гипотезы о локализации памяти):

- **Interaction / norm:** колонка 0 и 1–2 соседние доминируют по вниманию и норме
  активаций.
- **Readout:** `SUB ≈ ALL` и заметно выше средней одиночной колонки → почти вся
  SDQ-точность сосредоточена в узкой подгруппе.
- **Masked readout:** `subgroup_masked` близок к unmasked `SUB` → подгруппе достаточно
  внутренних связей, внешние колонки не нужны.

Сигнал о проблеме (гипотеза не подтверждается): `SUB` заметно ниже `ALL`, либо
`subgroup_masked` проваливается относительно `SUB` — значит, ответ распределён по многим
колонкам, а не локализован. `NaN` в accuracy означает, что после фильтра `ignore_index`
осталось < 32 примеров — стоит увеличить `--n_collect`.

## Как запускать

```sh
uv run inference/analyze_columns_sdq.py \
    --checkpoint runs/checkpoints/<run>/step_40000000.pt \
    --device cuda --n_collect 400 --subgroup_k 3 \
    --log.logger=comet --name "v4 col-ablation 40M"
```
