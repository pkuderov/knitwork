# GridRnnFusion

GridRNN Fusion v3 — гибридная архитектура, объединяющая обучаемые HGRN-ячейки с замороженными резервуарными колонками. Решает проблему однородности скрытых представлений: резервуарные колонки с разными спектральными радиусами создают богатую многомасштабную динамику без дополнительных параметров, а cross-attention позволяет обучаемым колонкам явно читать из резервуара. Diversity loss принуждает обучаемые колонки к различию представлений. Батчевые операции над колонками устраняют Python-цикл и дают 3–5× ускорение.

## Ключевой механизм

На каждом слое: HGRN-колонки → cross-attention с резервуаром → совместный message passing → gated merge:

```python
# x_cols: (batch, n_cols, emb/hidden) — each column sees its own projection
h_t_new = self._batched_trainable_forward(li, x_t_in, h_t_in)  # (batch, n_t, hidden)
h_r_new = self._batched_reservoir_forward(li, x_r_in, h_r_in)  # (batch, n_r, hidden)

# trainable columns read from reservoir via cross-attention
if self.cross_attns is not None:
    h_t_new = self.cross_attns[li](h_t_new, h_r_new)

# joint message passing over all columns  [n_cols, batch, hidden]
h_all_seq = cat([h_t_new, h_r_new], dim=1).permute(1, 0, 2)
msg, attn_w = self.attn[li](h_all_seq, return_weights=return_attn)

# gate only for trainable columns
gate_logit = self.attn_gates[li](cat([h_t_seq, msg_t], dim=-1))
g = sigmoid(gate_logit)
h_t_merged = (1.0 - g) * h_t_seq + g * msg_t
```

## Важные детали реализации

**Батчевые HGRN-ячейки** — формула HGRN без Python-цикла по колонкам:

```python
# batched HGRN update  [n_t, batch, hidden]
o_t  = sigmoid(gx(W_o, b_o) + gh(U_o))          # output gate
c_t  = tanh(layer_norm(gx(W_c, b_c) + gh(U_c, o_t * h_p)))  # candidate
lam  = sigmoid(gx(W_f, b_f) + gh(U_f)) * (1 - betas) + betas  # forget (λ)
h_new = lam * h_p + (1.0 - lam) * c_t           # HGRN recurrence
```

Бета (`β`) — нижняя граница ворот забывания, растёт от нижних слоёв к верхним, давая нижним слоям более быструю, а верхним — более медленную динамику.

**Резервуарные колонки** с разными спектральными радиусами для многомасштабной памяти:

```python
# spectral_radii assigned per reservoir column, e.g. [0.7, 0.95]
# W_hh frozen after init, scaled to target spectral radius
# GRU-like update without backprop through reservoir weights
```

**Diversity loss** — три компоненты для разнообразия колонок:

```python
# cosine: penalizes cosine similarity > margin between column pairs
# covariance: penalizes off-diagonal covariance (VICReg-style)
# variance: penalizes low within-column variance
# gate_entropy: maximizes entropy of gate values
total = (cos_t + cov_t + var_t + gate_t) * cfg.total_weight
```

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `n_reservoir_cols` | Число замороженных резервуарных колонок; по умолчанию 2 из `n_columns` |
| `spectral_radii` | Список спектральных радиусов для каждой резервуарной колонки; если None — назначаются автоматически (например, `[0.7, 0.95]`) |
| `reservoir_scale` | Масштаб входных весов резервуара |
| `beta_min / beta_max` | Диапазон нижней границы λ в HGRN; `beta_min` у нижних слоёв, `beta_max` у верхних |
| `learnable_beta` | Если True — `β` обучается, иначе фиксировано |
| `use_cross_attention` | Включает cross-attention обучаемых колонок к резервуарным |
| `all_cols_get_input` | Все колонки получают вход через разные ортогональные проекции (не только первая) |
| `diversity_loss.total_weight` | Общий масштаб diversity loss; по умолчанию 0.05 |
| `diversity_loss.compute_every_n` | Вычислять diversity loss каждые N шагов для ускорения |
| `use_final_output_gate` | Sigmoid-гейт поверх выхода верхнего слоя перед головой |
