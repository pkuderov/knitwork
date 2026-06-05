# grnn_delta — Grid RNN с двухмасштабной Delta-памятью

Grid RNN, в котором каждый слой дополнен явной ассоциативной памятью на основе правила дельта (Widrow-Hoff). Цель — разделить временну́ю динамику (LRU-состояние) и ключ-значение ассоциации (матрица быстрых весов): дистракторы не перезаписывают сохранённые пары, поскольку те хранятся в отдельной структуре.

## Ключевой механизм

**TwoScaleMemLayer** — векторизованная двухмасштабная delta-память для всех колонок слоя сразу:

```python
# для каждого слоя l, все C колонки батчатся в [C*B, H]
k = normalize(Wk(h))            # ключ записи
v = Wv(h)                        # значение
q = normalize(Wq(h) + col_bias) # ключ чтения (per-col bias)
g = sigmoid(Wg(h))               # вентиль записи

# дельта-правило: убираем старую ассоциацию, пишем новую
v_old = W @ k                    # что хранится для ключа k
W ← decay * W + g * k ⊗ (v − v_old)

# чтение
m = W^T @ q
```

Два масштаба (fast: `dk_f=H//8`, slow: `dk_s=H//4`) с разными decay per-layer. Слой 0: быстрый (decay_fast≈0.3, decay_slow≈0.95), слой L-1: медленный (decay_fast≈0.7, decay_slow≈0.999).

Почему дельта-правило лучше Hebbian (`grnn_engram`):
- Hebbian: `W ← W + η·v⊗k` — накапливает интерференцию между разными ключами
- Delta: явно удаляет старую запись для ключа k перед новой — точное переписывание без накопления ошибок

**LRU с двойной иерархией r_max** — быстрые нижние слои (r_max≈0.05) и медленные верхние (r_max≈0.999), плюс per-col вариация внутри слоя:

```python
r_max[l] = lerp(r_min_layers, r_max_layers, l / (L-1))
r_max[l, c] = r_max[l] * lerp(0.85, 1.0, c / (C-1))
```

**Cross-layer skip** (опционально, `use_cross_layer_skip=True`): верхний слой читает напрямую из медленной памяти нижнего слоя, создавая «коридор памяти»:

```python
q_skip = normalize(Wq_skip(h_top))
m_skip = W_slow[layer=0].T @ q_skip
h_top ← h_top + Wo_skip(m_skip)
```

## Гиперпараметры

| Параметр | SDQ default | Text default | Описание |
|----------|-------------|--------------|----------|
| `dk_fast` / `dv_fast` | 16 | 16 | Размерность быстрой памяти (H//8) |
| `dk_slow` / `dv_slow` | 32 | 32 | Размерность медленной памяти (H//4) |
| `mem_decay_fast` | [0.3, 0.5, 0.7] | [0.5, 0.7, 0.85, 0.9] | Decay быстрой памяти per-layer |
| `mem_decay_slow` | [0.95, 0.98, 0.999] | [0.97, 0.985, 0.995, 0.999] | Decay медленной памяти per-layer |
| `r_min_layers` | 0.05 | 0.2 | r_max нижнего слоя LRU |
| `r_max_layers` | 0.999 | 0.9995 | r_max верхнего слоя LRU |
| `use_cross_layer_skip` | false | false | Коридор памяти top→bottom |

## Состояние

```python
class DeltaGridState(NamedTuple):
    h:      Tensor  # [L, C, B, 2H]          — LRU скрытые состояния
    W_fast: Tensor  # [L, C, B, dk_f * dv_f] — быстрые delta-матрицы
    W_slow: Tensor  # [L, C, B, dk_s * dv_s] — медленные delta-матрицы
```
