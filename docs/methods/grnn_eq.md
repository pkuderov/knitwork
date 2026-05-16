# EquilibriumGridRnnCoT (grnn_eq)

Модель решает проблему ограниченной «глубины мышления» за один шаг времени: стандартный Grid RNN делает ровно по одному проходу через каждый слой, тогда как некоторые входы требуют большего числа вычислительных шагов. Основная идея объединяет три механизма: `EquilibriumCell` ищет приближённую неподвижную точку GRU итеративным применением `h* = GRU(x, h*)` до сходимости; `ACT` (Adaptive Computation Time) динамически решает, когда остановить итерации для каждого примера в батче; `ChainOfThoughtGRU` накапливает отдельное «мысленное» состояние `thought` поверх верхнего слоя сетки, давая модели явный рабочий буфер для многошаговых рассуждений.

## Ключевой механизм

```python
# iterative equilibrium search with ACT halting
for act_step in range(self.max_eq_iters):
    h_layer_new = torch.stack([cell.cell(x_in[ic], h_layer[ic])
                               for ic, cell in enumerate(cells_row)], dim=0)
    h_pool  = h_layer_new.mean(dim=0)     # [batch, hidden]
    p_halt  = halter(h_pool)              # [batch,] halting probability

    is_last = (act_step == self.max_eq_iters - 1)
    p_use   = torch.where(is_last | (halt_acc + p_halt >= 1.0 - act_eps),
                          1.0 - halt_acc, p_halt)

    w      = p_use.unsqueeze(0).unsqueeze(-1)   # [1, batch, 1]
    h_acc += w * (h_layer_new if is_last else h_layer_new.detach())
    halt_mask = halt_mask | (halt_acc >= 1.0 - act_eps)
    if halt_mask.all():
        break
```

На каждом шаге ACT взвешивает выходы ячеек по вероятности остановки `p_use`; промежуточные шаги детачируются — только финальный шаг несёт градиент, что экономит память.

## Важные детали реализации

**Chain-of-Thought буфер:**

```python
# thought state accumulates across time steps
h_top   = h_new[-1, 0]                          # [batch, hidden]
thought = self.cot(h_top, thought)               # [batch, thought_size]
y       = self.head(torch.cat([h_top, thought], dim=-1))
```

`ChainOfThoughtGRU` — отдельная GRUCell с LayerNorm, которая обновляет вектор `thought` от шага к шагу. Голова читает конкатенацию `[h_top; thought]`, что позволяет модели использовать информацию из предыдущих «мыслей».

**HaltingUnit:**

```python
# initialized with near-zero weights and bias=-2 => initially almost never halts
self.proj = nn.Linear(hidden_size, 1)
nn.init.zeros_(self.proj.weight)
nn.init.constant_(self.proj.bias, -2.0)
```

Инициализация смещения `-2` соответствует стартовой вероятности остановки `sigmoid(-2) ≈ 0.12`, побуждая модель делать несколько итераций до схождения.

**ACT лосс:**

```python
# ponder penalty: minimizes unnecessary computation steps
def act_loss(self, act_iters_list) -> torch.Tensor:
    total = sum(it.float().mean() for it in act_iters_list)
    return self.act_loss_weight * total
```

Штраф пропорционален среднему числу итераций по всем слоям и примерам — компромисс между точностью и вычислительными затратами.

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `max_eq_iters` | Максимальное число итераций поиска неподвижной точки; ограничивает глубину вычислений |
| `eq_tol` | Порог сходимости по норме разности `‖h_new − h‖`; используется только в `EquilibriumCell.forward`, не в ACT-цикле основного grid step |
| `act_eps` | Допуск ACT: остановка при `halt_acc ≥ 1 − eps`; меньшее значение → строже |
| `act_loss_weight` | Вес штрафа за количество итераций; типично `1e-3` |
| `thought_size` | Размер вектора `thought`; по умолчанию равен `hidden_size`; можно уменьшить для экономии параметров |
