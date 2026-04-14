# HGRN_GridRnn

Модель решает проблему однородной «забывчивости» стандартного GRU в контексте Grid RNN: все слои сети сбрасывают состояние с одинаковой скоростью, что мешает одновременно хранить локальные и долгосрочные паттерны. Ключевая идея — заменить GRUCell на HGRUCell (Hierarchically Gated Recurrent Unit), где forget-гейт λ имеет обучаемую нижнюю границу β, специфичную для каждого слоя. Нижние слои (β ≈ 0) работают как обычный GRU и быстро переписывают состояние, тогда как верхние слои (β → 1) практически никогда не забывают — в духе иерархических рекуррентных сетей HGRN.

## Ключевой механизм

HGRUCell вводит три гейта вместо двух в GRU: output gate `o_t`, content candidate `c_t` и forget gate `λ_t` с нижней границей β.

```python
# output gate controls how much of h_{t-1} enters content computation  [B, H]
o_t = torch.sigmoid(self.W_o(x) + self.U_o(h))
# candidate content uses gated previous state                           [B, H]
c_t = torch.tanh(self.W_c(x) + self.U_c(o_t * h))
# forget gate bounded below by beta: lambda in [beta, 1]               [B, H]
raw_f = torch.sigmoid(self.W_f(x) + self.U_f(h))
lam_t = raw_f * (1.0 - self.beta) + self.beta
# state update
h_new = lam_t * h + (1.0 - lam_t) * c_t
```

β хранится как `beta_raw` в пространстве до сигмоиды (`β = sigmoid(beta_raw)`), что гарантирует β ∈ (0, 1) при любых значениях параметра.

## Важные детали реализации

**Иерархическое назначение β по слоям.** При инициализации β линейно распределяются от `beta_min` (нижний слой) до `beta_max` (верхний слой):

```python
# layer 0 -> beta_min, layer L-1 -> beta_max
betas = [
    beta_min + (beta_max - beta_min) * i / (n_layers - 1)
    for i in range(n_layers)
]
```

Это жёстко задаёт иерархию временны́х горизонтов: нижние слои следят за текущим токеном, верхние — за долгосрочным контекстом.

**Output gate на выходе всей сетки.** Поверх финального состояния добавлен `final_output_gate` — дополнительный sigmoid-блок, масштабирующий представление перед головой:

```python
gate = self.final_output_gate(z)   # [B, H]
z = gate * z
y = self.head(z)
```

**Post-messaging с гейтом слияния.** После атention-обмена сообщениями между колонками исходное состояние и сообщение смешиваются через обучаемый гейт:

```python
g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))   # [cols, B, 1]
hl_n = (1.0 - g) * hl_n + g * msg
```

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `beta_min` | Нижняя граница λ для самого нижнего слоя (≈ 0 — высокая забывчивость) |
| `beta_max` | Нижняя граница λ для самого верхнего слоя (≈ 0.99 — долгосрочная память) |
| `messaging` | `"post"` — атention после шага ячейки; `"pre"` — до шага (меняет размерность входа) |
| `col_identities` | Добавлять ли обучаемые позиционные bias в атention для различения колонок |
