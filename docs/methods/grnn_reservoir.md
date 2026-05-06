# GridRnnReservoir

GridRnnReservoir — это Grid RNN с Echo State Network (ESN)-подобными резервуарными столбцами. Идея: часть столбцов в каждом слое замораживается после инициализации (веса не обновляются), а их рекуррентная матрица масштабируется к заданному спектральному радиусу. Это создаёт богатое случайное проекционное пространство с контролируемой динамикой памяти, которую обучаемые столбцы могут читать через механизм межколоночного message passing.

## Ключевой механизм

```python
# scale recurrent weight_hh of each GRU gate to target spectral radius
for gate_idx in range(3):
    block = cell.weight_hh.data[gate_idx * hid:(gate_idx + 1) * hid]  # [H, H]
    _scale_to_spectral_radius(block, spectral_radius)

# freeze reservoir columns — no gradients
for param in cell.parameters():
    param.requires_grad = False
```

Функция `_scale_to_spectral_radius` использует `torch.linalg.eigvals` для матриц до 512×512 и степенную итерацию для больших, после чего умножает веса на `target / current_radius`.

## Важные детали реализации

**Разделение на обучаемые и резервуарные столбцы.** Первые `n_trainable_cols` столбцов обучаются обычно, последние `n_reservoir_cols` — заморожены:

```python
first_reservoir = self.n_trainable_cols   # = n_columns - n_reservoir_cols
for icol in range(first_reservoir, self.n_columns):
    self._init_reservoir_cell(cell, spectral_radius, reservoir_scale)
    for param in cell.parameters():
        param.requires_grad = False
```

**Инициализация резервуарной ячейки.** Входные веса масштабируются равномерно к `reservoir_scale`, смещения обнуляются для стабильности:

```python
nn.init.uniform_(cell.weight_ih, -scale, scale)
nn.init.zeros_(cell.bias_ih)
nn.init.zeros_(cell.bias_hh)
```

**Мониторинг спектральных радиусов.** Метод `reservoir_info()` возвращает фактические спектральные радиусы всех гейтов резервуарных ячеек — полезно для проверки корректности инициализации.

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `n_reservoir_cols` | Число замороженных столбцов (< n_columns). Резервуарные столбцы — последние по индексу |
| `spectral_radius` | Спектральный радиус рекуррентной матрицы. < 1 — затухающая память, ≈ 1 — критический режим (рекомендуется 0.9) |
| `reservoir_scale` | Масштаб инициализации входных весов резервуара. Малые значения (0.1) обеспечивают слабое влияние входа на резервуар |
