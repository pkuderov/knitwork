# GridRnn

GridRnn решает задачу ассоциативной памяти и языкового моделирования, организуя рекуррентные ячейки в двумерную сетку «слои × колонки». Вместо одного скрытого вектора модель поддерживает матрицу состояний `[layers, cols, batch, hidden]`, где колонки на каждом слое обмениваются информацией через механизм внимания (post- или pre-messaging). Входной токен поступает только в нулевую колонку первого слоя; остальные колонки в первом слое получают нулевой фиктивный вход, но обогащаются через attention. Предсказание строится из состояния верхнего слоя, нулевой колонки.

## Ключевой механизм

Post-messaging: после независимого шага GRU по всем колонкам запускается attention, результат которого примешивается через обучаемые ворота.

```python
# hl_n: [cols, batch, hidden] — states after GRU step
hl_n = torch.stack([cell_forward(cells, x, hl, ix_col=c)
                    for c in range(self.n_columns)], dim=0)

msg, attn_w = attn(hl_n, return_weights=return_attn)
g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))
hl_n = (1 - g) * hl_n + g * msg          # gated mixing
```

Ворота `g` позволяют каждой колонке самостоятельно решать, насколько учитывать агрегированное сообщение от соседей.

## Важные детали реализации

`MessagePassingLayer` добавляет обучаемые идентификаторы колонок к запросам и ключам, чтобы attention мог различать участников:

```python
# ids: [n_cols, 1, dim] — learnable per-column bias
if self.ids is not None:
    qh = kh = qh + self.ids
h_mixed, attn_w = self.mha(qh, kh, vh, average_attn_weights=True)
return self.norm(h_mixed), attn_w
```

Инициализация `out_proj` близкой к нулю делает начальные сообщения незначительными, что стабилизирует старт обучения.

---

Вход первого слоя хранится как список (разные размерности на разных колонках), а со второго слоя — как плотный тензор `[cols, batch, hidden]`:

```python
def _prepare_grid_input(self, x):
    xl = [x]   # col 0: embedding, shape [batch, embed_dim]
    dummy = torch.zeros(bsz, 1, device=x.device, dtype=x.dtype)
    for _ in range(1, self.n_columns):
        xl.append(dummy)   # cols 1..C-1: dummy 1-dim input
    return xl
```

Это позволяет иметь разные размеры входа для первой колонки (embedding) и всех остальных.

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `base_hidden_size` | Опорный размер скрытого состояния однослойного GRU; `hidden_size` выбирается автоматически так, чтобы число параметров совпадало |
| `n_columns` | Число колонок в сетке; должно быть > 1 |
| `messaging` | `"post"` — attention после GRU-шага (с воротами); `"pre"` — attention до GRU-шага (конкатенация) |
| `col_identities` | Добавлять ли обучаемые идентификаторы колонок в attention |
| `n_attn_heads` | `hidden_size` округляется вниз до кратного этому числу |
