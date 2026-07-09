# HopfieldGridRnnFixV4

Hopfield-версия `grnn_fix_v4`: та же архитектура персонального внимания между столбцами, но на LSTM-ячейках с **двойной памятью**. Мотивация — лучший результат семейства на SDQ исторически принадлежал hgrnn (Acc++ 0.870 при 215K), и его преимущество давала изоляция долговременной памяти `c` от сообщений. Модель проверяет, складываются ли выигрыши: механики v4 (per-column идентичности и β, гейт с обзором входа, таймскейлы, aux-лоссы) + защита памяти LSTM.

## Ключевой механизм

Двойная память: рабочая `h` несёт смешанное сообщение рекуррентно, ячейковая `c` не смешивается никогда:

```python
h_ic, c_ic = cells[ic](x[ic], (hl[ic], cl[ic]))   # LSTM per column
msg, _ = attn(hl_new)                              # PerColumnAttention from v4
g = torch.sigmoid(gates[ic](cat([hl_new, msg, x])))
hl_mix = hl_new + g * msg    # goes into recurrent h AND upward
c_n.append(cl_new)           # long-term memory: never mixed
```

Отличие от `grnn_fix_v4` (GRU): там рекуррентное состояние полностью защищено, а сообщение живёт только на восходящем пути; здесь сообщение **входит в рекуррентную** `h` (столбцы могут писать в рабочую память друг друга через гейт), но её ошибки не могут затереть `c`.

## Важные детали реализации

Мульти-таймскейл через forget-гейт LSTM (вместо update-гейта GRU):

```python
# f -> 1 remembers longer (slow column); f -> 0 forgets fast
shift = timescale_spread * (2 * ic / (n_columns - 1) - 1)
cell.bias_ih[H:2 * H] += shift    # LSTM bias layout: [i, f, g, o]
```

Остальное идентично v4: `PerColumnAttention` (идентичности Q/K + per-(столбец, голова) β), RMSNorm между слоями, concat-readout, четыре aux-лосса (Barlow с ростом веса по глубине, gate-std, activity-декорреляция, анти-сатурация верхних слоёв) раз в `aux_every` вызовов.

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `hidden_size` | 56 при 2L×3C даёт ~204K параметров (LSTM стоит 4/3 GRU) |
| `beta_scale` | 3.0 — как у v4; per-column разброс 0.5×–2× |
| `timescale_spread` | 1.0 — амплитуда сдвига forget-bias по столбцам |
