# EquilibriumGridRnnCoT

GridRNN с Adaptive Computation Time (ACT) и Chain-of-Thought (CoT). Решает проблему фиксированной глубины вычислений: вместо одного прохода через сетку каждый слой выполняет переменное число итераций до достижения неподвижной точки (equilibrium). ACT-механизм динамически определяет, когда остановиться, а CoT-GRU накапливает «мысли» поверх верхнего слоя как медленная долгосрочная память. Три вспомогательных лосса принуждают сеть к равновесию (MSE-невязка), экономии вычислений (ponder cost) и равномерному участию всех колонок в attention (энтропийный регуляризатор).

## Ключевой механизм

ACT-цикл с взвешенной аккумуляцией состояний и адаптивной остановкой:

```python
# ACT loop per layer  [cols, batch, hidden]
for act_step in range(self.max_eq_iters):
    h_layer_new = stack([cell(x_in[ic], h_layer[ic]) for ic, cell in enumerate(cells_row)])

    # Anderson mixing for faster convergence
    if self.anderson_beta > 0.0 and act_step >= 1:
        h_layer_new = h_layer_new + self.anderson_beta * (h_layer_new - h_layer_prev)

    # cross-column attention every attn_every steps
    if is_attn_step:
        msg, attn_w = attn_mod(h_layer_new, return_weights=True)
        g = sigmoid(gate_mod(cat([h_layer_new, msg], dim=-1)))
        h_layer_new = (1.0 - g) * h_layer_new + g * msg

    # halting probability
    p_halt = halter(h_layer_new.mean(dim=0))      # (batch,)
    p_use  = where(is_last | halt_acc + p_halt >= 1 - eps, 1 - halt_acc, p_halt)

    # weighted accumulation  [1, batch, 1]
    h_acc = h_acc + p_use.unsqueeze(0).unsqueeze(-1) * h_layer_new
    halt_mask = halt_mask | (halt_acc >= 1 - eps)
    if halt_mask.all():
        break
```

На каждом шаге ACT вычисляется вероятность остановки `p_halt`, и состояния взвешенно суммируются в `h_acc`. Цикл завершается, когда все примеры в батче остановились или исчерпан `max_eq_iters`.

## Важные детали реализации

**Equilibrium residual loss** — прямой сигнал для достижения неподвижной точки:

```python
# penalty: one more GRU step from h_acc should not change it  [cols, batch, hidden]
h_acc_detached = h_acc.detach()
h_check = stack([cell(x_in[ic], h_acc_detached[ic]) for ic, cell in enumerate(cells_row)])
residual_loss = F.mse_loss(h_check, h_acc_detached)
```

**Ponder cost (FIX-1)** — нормированное число итераций, а не накопленная вероятность (которая всегда = 1.0):

```python
# n_iters counts real steps per example  [batch,]
n_iters += (~halt_mask).float()
ponder_cost = n_iters.mean() / self.max_eq_iters   # range [1/max .. 1]
```

**Chain-of-Thought** — отдельный GRUCell поверх верхнего слоя, выход конкатенируется с `h_top` перед головой:

```python
h_top   = h_new[-1, 0]                            # (batch, hidden)
thought = self.cot(h_top, thought)                 # GRUCell + LayerNorm
y       = self.head(cat([h_top, thought], dim=-1))
```

**Column participation loss (FIX-5)** — штраф за неравномерное внимание между колонками:

```python
# maximize entropy of mean attention weights → uniform column participation
attn_w_mean = stack(attn_w_accum).mean(0)         # (cols, cols)
col_entropy  = -(attn_w_mean * (attn_w_mean + 1e-8).log()).sum(-1).mean()
participation_loss = -col_entropy                  # minimize negative entropy
```

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `max_eq_iters` | Максимальное число equilibrium-итераций на слой; по умолчанию 12 |
| `eq_tol` | Порог нормы невязки для логирования сходимости (не влияет на остановку — ею управляет ACT) |
| `act_eps` | Эпсилон ACT: остановка, когда `halt_acc >= 1 - eps`; по умолчанию 0.01 |
| `act_loss_weight` | Вес ponder cost в суммарном лоссе |
| `eq_residual_weight` | Вес MSE-невязки equilibrium; по умолчанию 1e-2 |
| `col_participation_weight` | Вес энтропийного регуляризатора колонок; по умолчанию 1e-3 |
| `anderson_beta` | Коэффициент Anderson mixing для ускорения сходимости; 0.0 = отключено |
| `attn_every` | Attention применяется каждые N итераций внутри ACT-цикла |
| `thought_size` | Размер CoT-состояния; по умолчанию равен `hidden_size` |
