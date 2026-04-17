# HopfieldGridRnn

Модель исследует, насколько Modern Hopfield Networks (MHN) могут улучшить обмен информацией между колонками Grid RNN по сравнению со стандартным multi-head attention. В отличие от базовой `GridRnn`, здесь GRUCell заменён на LSTM (как в оригинальной статье об ассоциативной памяти) и используется `HopfieldMessageLayer` — attention с обучаемым масштабирующим параметром β на уровне каждой головы. При больших β слой стремится к winner-take-all режиму, соответствующему динамике Хопфилда, а при малых β ведёт себя как обычный softmax-attention.

## Ключевой механизм

`HopfieldMessageLayer` заменяет фиксированный scaling `1/√d_k` на обучаемый `β = exp(log_β)`:

```python
# beta per head: (num_heads,) -> (num_heads, 1, 1, 1) for broadcast
beta = self.log_beta.exp().view(self.num_heads, 1, 1, 1)
# Hopfield energy score instead of standard 1/sqrt(d) scaling
scores = beta * torch.matmul(q, k.transpose(-2, -1))   # [heads, B, C, C]
attn = torch.softmax(scores, dim=-1)
out = torch.matmul(attn, v)                             # [heads, B, C, d_k]
```

Большое β усиливает «остроту» внимания, и сеть работает как ассоциативная память с одним ключевым паттерном. Инициализируется как `log(1/√d_k)` — эквивалентно стандартному масштабированию, и дальше обучается.

## Важные детали реализации

**LSTM вместо GRU.** Состояние хранится как пара `(h, c)`:

```python
h_ic, c_ic = cells[ic](x_list[ic], (hl[ic], cl[ic]))
```

Из-за этого `state` — кортеж `(h, c)` формы `(layers, cols, batch, hidden)` каждый, а не один тензор как в `GridRnn`. Методы `reset_state`, `detach_state`, `init_state` работают с обоими тензорами.

**Малая инициализация out_proj.** Чтобы в начале обучения message почти не влияло на состояние, веса выходной проекции инициализируются очень малыми значениями:

```python
nn.init.normal_(self.out_proj.weight, 0.0, 0.001)
nn.init.zeros_(self.out_proj.bias)
```

**LayerNorm на выходе Hopfield-слоя.** `HopfieldMessageLayer.forward` возвращает `self.norm(out)` без residual-связи — нормализация стабилизирует активации при больших β.

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `log_beta` | Логарифм масштабирующего коэффициента β (по одному на каждую голову); чем больше, тем «острее» attention и тем ближе к режиму Хопфилда |
| `n_attn_heads` | Число голов; `hidden_size` обрезается до ближайшего кратного |
| `messaging` | `"post"` — атention после LSTM-шага; `"pre"` — до шага, меняет входную размерность ячеек |
