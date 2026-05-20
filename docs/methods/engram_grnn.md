# EngramGridRnn

Grid RNN, где каждая ячейка дополнена ассоциативной памятью (`EngramMemory`). На каждом шаге ячейка сначала извлекает из памяти релевантный вектор через разреженное косинусное внимание, конкатенирует его с входом и передаёт в GRU. После обновления скрытого состояния память перезаписывается по правилу Хебба. Сообщения между столбцами передаются через многоголовое внимание (режимы `post` и `pre`).

## Ключевой механизм

Чтение из памяти и запись по Хеббу реализованы в `EngramMemory.forward`:

```python
# read → hebbian write  [B, H], [B, S, H]
def forward(self, h, M):
    r, attn = self.read(h, M)
    return r, self.write(h, M, attn), attn
```

`read` вычисляет разреженное косинусное внимание по слотам, агрегирует значения взвешенной суммой и применяет read-gate. `write` сдвигает слоты к текущему скрытому состоянию пропорционально весам внимания.

## Важные детали реализации

**Разреженное косинусное внимание** — top-K маскирование нулит слабые слоты перед softmax:

```python
# sparse cosine attention over memory slots  [B, S]
q_norm = F.normalize(query.unsqueeze(1), dim=-1)   # [B, 1, H]
scores = (q_norm * F.normalize(M, dim=-1)).sum(-1)
threshold = scores.topk(self.top_k, dim=-1).values[:, -1:]
scores = scores.masked_fill(scores < threshold, float('-inf'))
attn = torch.softmax(scores, dim=-1)
```

**Hebbian write** с гейтом и нормализацией слотов:

```python
# Hebbian delta rule; w in [0,1] from write_gate  [B, S, H]
delta = h.unsqueeze(1) - M
lr = self.hebb_lr * w.unsqueeze(-1)
M_new = M + lr * attn.unsqueeze(-1) * delta
return M_new / M_new.norm(dim=-1, keepdim=True).clamp(min=1.0)
```

Нормализация `clamp(min=1.0)` не сжимает короткие векторы, но предотвращает взрывной рост длинных.

**Вход GRU** — конкатенация входного токена/сообщения и вектора извлечения `r`:

```python
# augment cell input with engram retrieval  [B, input_dim + H]
r, M_new, eng_attn = engram_row[col_i](h_prev, M_prev)
x_aug = torch.cat([x_input[col_i], r], dim=-1)
h_new = cells[col_i](x_aug, h_prev)
```

**Post-message gate** смешивает исходное скрытое состояние со столбцовыми сообщениями:

```python
# gated message mixing across columns  [cols, B, H]
msg, attn_w = attn_msg(hl_n, return_weights=return_attn)
g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))
hl_n = (1.0 - g) * hl_n + g * msg
```

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `n_engram_slots` | Количество слотов памяти на ячейку; больше слотов — выше ёмкость, но дороже внимание |
| `engram_top_k` | Сколько слотов участвует в softmax; управляет разреженностью извлечения |
| `engram_hebb_lr` | Скорость обновления памяти по Хеббу; при высоких значениях память быстро перезаписывается |
| `engram_gate_write` | Включает write-gate: ячейка может подавить запись, если текущий вход нерелевантен |
| `messaging` | `post` — сообщения после GRU-шага (с gate); `pre` — перед GRU-шагом (без gate) |
| `col_identities` | Добавляет позиционные эмбеддинги столбцов в механизм внимания |
