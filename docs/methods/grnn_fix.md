# GridRnnFix

Исправленный вариант обмена вниманием между столбцами Grid RNN. Базовый `grnn` страдает от четырёх дефектов (см. `architecture_analysis.md` §2): выпуклое усреднение стягивает столбцы (col_sim ≈ 0.93), LayerNorm после крошечно инициализированной проекции превращает «пренебрежимое» сообщение в шум масштаба 1, столбцы без входа рождаются схлопнутыми, а сообщение перезаписывает единственное рекуррентное состояние GRU. GridRnnFix применяет все исправления §7.1 одновременно: аддитивное сообщение, закрытый на старте гейт, отсутствие пост-нормы, обучаемая температура β, вход во все столбцы через ортогональные проекции, защита рекуррентного состояния и concat-readout по столбцам.

## Ключевой механизм

Сообщение добавляется к выходу слоя, но не к рекуррентному состоянию — память столбца никогда не смешивается:

```python
# pure recurrence, then additive gated message  [C, B, H]
hl_n = torch.stack([cells[ic](x[ic], hl[ic]) for ic in range(self.n_columns)], dim=0)
msg, attn_w = attn(hl_n, return_weights=return_attn)
g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))
o = hl_n + g * msg      # goes up to the next layer / readout
h_n.append(hl_n)        # recurrent state stays unmixed
```

Рекуррентная динамика каждого столбца остаётся чистой (аналог защищённого `c` в LSTM), а обогащённое сообщением представление `o` питает следующий слой и readout.

## Важные детали реализации

Разрыв симметрии: каждый столбец получает эмбеддинг через собственную ортогональную проекцию (в `grnn` вход видит только col₀, остальные стартуют идентичными):

```python
self.col_input_projs = nn.ModuleList(
    nn.Linear(embedding_size, embedding_size, bias=False) for _ in range(n_columns)
)
x = torch.stack([proj(x) for proj in self.col_input_projs], dim=0)  # [C, B, E]
```

Гейт закрыт на старте — модель включает внимание по мере пользы, а не борется с шумом:

```python
nn.init.constant_(gate.bias, -3.0)   # sigmoid(-3) ~ 0.05 at init
```

`ColumnAttention` — Hopfield-стиль без пост-нормы: tiny-init `out_proj` действительно делает сообщение нулевым на старте (в `grnn` LayerNorm это аннулировал), β обучается на голову:

```python
beta = self.log_beta.exp().view(self.num_heads, 1, 1, 1)
attn = torch.softmax(beta * torch.matmul(q, k.transpose(-2, -1)), dim=-1)
nn.init.normal_(self.out_proj.weight, 0.0, 0.001)   # no norm after this
```

Readout — конкатенация столбцов верхнего слоя вместо чтения одного столбца `h[-1][0]`:

```python
self.head = nn.Linear(self.n_columns * self.hidden_size, output_size)
```

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `hidden_size` | 64 при 2L×3C даёт ~201K параметров (SDQ и text8) |
| `n_layers` | ≥2 осмысленно: при 1 слое сообщение влияет только на readout |
| `n_attn_heads` | H обрезается до кратного числу голов; β обучается на голову |
