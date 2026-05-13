# GridLRU

GridLRU — это вариант Grid RNN, в котором стандартные GRU-ячейки заменены блоками LRU (Linear Recurrent Unit). LRU работает в комплексном пространстве состояний, что позволяет явно контролировать диапазон запоминаемых временных масштабов через параметр спектрального радиуса `r_max`. Модель решает проблему ограниченной памяти GRU: за счёт диагональной рекуррентной матрицы в комплексных числах LRU стабильно удерживает долгосрочные зависимости, а сетка столбцов позволяет каждому столбцу специализироваться на своём временном масштабе.

## Ключевой механизм

```python
# each column gets its own r_max, linearly spaced across [r_min, r_max]
col_r_max = r_min + (r_max - r_min) * (icol + 1) / n_columns  # if lru_r_per_col

# LRUBlock returns two tensors: output activations and complex state
y_col, h_col_n = cells[icol](x_list[icol], hl[icol])
# y_col:   [batch, H]    — real-valued output
# h_col_n: [batch, 2*H]  — complex state (real + imag packed)
```

Состояние сети имеет форму `[layers, cols, batch, 2*H]`, где последнее измерение хранит вещественную и мнимую части комплексного состояния LRU. Выходные активации `[batch, H]` формируются отдельно и не смешиваются с состоянием.

## Важные детали реализации

**Per-column r_max.** При `lru_r_per_col=True` каждый столбец получает свой максимальный радиус памяти, линейно увеличивающийся от `r_min` до `r_max`. Столбец 0 — короткая память, последний столбец — длинная:

```python
col_r_max = r_min + (r_max - r_min) * (icol + 1) / n_columns
```

**Gated merge после message passing.** После агрегации сообщений гейт с сигмоидой управляет смешением исходных и полученных активаций:

```python
g      = torch.sigmoid(attn_gate(torch.cat([out_t, msg], dim=-1)))  # [cols, batch, 1]
merged = (1 - g) * out_t + g * msg                                  # [cols, batch, H]
```

**Инициализация состояния.** `init_state` возвращает нулевой тензор с двойным по размеру последним измерением для хранения комплексного состояния:

```python
# state: [layers, cols, batch, 2*hidden_size]
return torch.zeros(self.n_layers, self.n_columns, bsz, 2 * self.hidden_size, ...)
```

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `r_min`, `r_max` | Диапазон спектрального радиуса LRU. `r_max=0.999` приближается к нейтральной памяти (собственные значения на единичной окружности) |
| `lru_r_per_col` | Если `True` — каждый столбец получает свой `r_max`, линейно возрастающий. Позволяет сетке охватить разные временные масштабы |
| `ff_mult` | Множитель размера промежуточного слоя feed-forward внутри LRUBlock |
