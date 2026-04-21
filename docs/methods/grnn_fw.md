# GridRnnFW

GridRNN с быстрыми весами (Fast Weights, Ba et al. 2016). Решает проблему кратковременной ассоциативной памяти: стандартный GRU помнит прошлое только через скрытое состояние, тогда как быстрые веса хранят явные попарные ассоциации между состояниями колонок в виде матрицы `A`. Матрица `A` обновляется по правилу Хебба на каждом шаге и читается content-based retrieval по запросу каждой колонки, создавая «межшаговую» память быстрее параметров, но медленнее активаций.

## Ключевой механизм

Хеббово обновление матрицы `A` и retrieval по запросу:

```python
# Hebbian write: outer product of value and key for each column  [batch, hidden, hidden]
delta_A = zeros_like(A)
for col_j in range(n_cols):
    k_j = F.normalize(k[col_j], dim=-1)   # (batch, hidden)
    v_j = F.normalize(v[col_j], dim=-1)
    delta_A += bmm(v_j.unsqueeze(2), k_j.unsqueeze(1))

A_new = decay * A + (fw_lr / n_cols) * delta_A   # exponential decay

# Content-based retrieval: each column reads from A via its query
msgs = []
for col_i in range(n_cols):
    q_i = F.normalize(q[col_i], dim=-1)
    msgs.append(bmm(A_new, q_i.unsqueeze(2)).squeeze(2))  # (batch, hidden)
h_msg = stack(msgs, dim=0)   # (cols, batch, hidden)
```

Каждая колонка записывает ассоциацию `v ⊗ k` в матрицу `A`, а затем все колонки читают из неё через свои запросы `q`. Параметры `k`, `q`, `v` — обучаемые линейные проекции (аналог head в attention, но без softmax).

## Важные детали реализации

**Состояние модели** — расширено матрицей `A` на каждый слой:

```python
# state = (h, A)
# h : (n_layers, n_cols, batch, hidden)
# A : (n_layers, batch, hidden, hidden)
```

**Псевдо-attention weights** для совместимости с визуализатором вычисляются из усреднённых по батчу ключей и запросов:

```python
# attn_w[i,j] ≈ similarity between col_i query and col_j key  [n_cols, n_cols]
q_mat = F.normalize(q.mean(dim=1), dim=-1)
k_mat = F.normalize(k.mean(dim=1), dim=-1)
scores = matmul(q_mat, k_mat.T)
attn_w = softmax(scores / scale, dim=-1)
```

**Gated merge** — как в базовом grnn.py, применяется поверх fast-weight retrieval:

```python
g = sigmoid(gate_lin(cat([hl_n, msg], dim=-1)))   # (cols, batch, 1)
hl_n = (1 - g) * hl_n + g * msg                   # (cols, batch, hidden)
```

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `fw_decay` | λ — экспоненциальное затухание матрицы `A`; 0.9 = медленное забывание, <0.5 = краткосрочная память |
| `fw_lr` | η — скорость записи в `A`; масштабирует силу каждого Хеббова обновления |
| `col_identities` | Если True — добавляет обучаемые векторы-идентификаторы к ключам и запросам колонок |
