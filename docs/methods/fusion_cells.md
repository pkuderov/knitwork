# BatchedHGRUColumns

`BatchedHGRUColumns` решает задачу эффективного параллельного вычисления нескольких HGRU-столбцов Grid RNN без Python-цикла по колонкам. Вместо `n_cols` отдельных `nn.Linear` все весовые матрицы упакованы в трёхмерные параметры формы `[n_cols, out, in]`, что позволяет обработать все столбцы одним вызовом `torch.bmm` с ускорением ~3–5× относительно поэлементного цикла. Каждый столбец использует собственный обучаемый параметр `beta` — нижнюю границу forget-gate, задающую временной масштаб.

## Ключевой механизм

```python
# x: (B, in)  h: (B, n_cols, hid)
x_t = x.T.unsqueeze(0).expand(self.n_cols, -1, -1)   # (n_cols, in, B)
h_t = h.permute(1, 2, 0)                               # (n_cols, hid, B)

# batched matmul for all columns simultaneously
def gate_x(W, b):
    out = torch.bmm(W, x_t).permute(0, 2, 1)           # (n_cols, B, hid)
    return out + b.unsqueeze(1) if b is not None else out

lam_t = torch.sigmoid(gate_x(W_f, b_f) + gate_h(U_f)) * (1.0 - betas) + betas
return (lam_t * h_perm + (1.0 - lam_t) * c_t).permute(1, 0, 2)   # (B, n_cols, hid)
```

`beta` параметризован через logit (`beta_raw`), что гарантирует значение в `(0, 1)` и позволяет каждому столбцу обучить свою временну́ю нишу.

## Важные детали реализации

**Иерархическая структура forget gate.** Формула `λ = sigmoid(...) * (1 - β) + β` обеспечивает нижнюю границу λ на уровне β: даже при максимальном "забывании" столбец удерживает долю β предыдущего состояния, что создаёт иерархию временных масштабов:

```python
betas = torch.sigmoid(self.beta_raw).view(n_cols, 1, 1)
lam_t = torch.sigmoid(gate_x(W_f, b_f) + gate_h(U_f)) * (1.0 - betas) + betas
```

**Инициализация весов.** Каждая матрица каждого столбца инициализируется ортогонально независимо, что снижает начальную корреляцию между столбцами:

```python
for name in ['W_f', 'W_o', 'W_c', 'U_f', 'U_o', 'U_c']:
    p = getattr(self, name)
    for i in range(self.n_cols):
        nn.init.orthogonal_(p[i])
```

**`BatchedReservoirColumns`.** Второй класс файла реализует замороженные GRU-столбцы с нормировкой спектрального радиуса рекуррентных матриц. Параметры `requires_grad=False` — сеть работает как эхо-состояние (ESN), добавляя нелинейное разнообразие без обучения.

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `beta_inits` | Начальные значения β для каждого столбца; разные значения задают разные временные масштабы с первого шага |
| `learnable_beta` | Если `False` — β фиксированы и иерархия задаётся вручную через `beta_inits` |
| `use_layer_norm` | LayerNorm применяется к candidate state `c_raw` до `tanh`, стабилизируя динамику при больших скрытых размерах |
| `spectral_radii` | (для `BatchedReservoirColumns`) Спектральный радиус рекуррентной матрицы каждого резервуарного столбца; значения < 1 обеспечивают эхо-состояние |

## Результаты

`BatchedHGRUColumns` и `BatchedReservoirColumns` — компоненты, используемые внутри `grnn_fusion`. Самостоятельных экспериментов не проводилось. Результаты интегрированной модели приведены в [grnn\_fusion.md](grnn_fusion.md): SDQ Acc=0.831, Acc++=0.708.
