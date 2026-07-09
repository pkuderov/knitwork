# GridRnnFixV5

v5.1 — гибридная сетка: **быстрые GRU-столбцы + медленные LRU-столбцы хранения + замороженный reservoir-hub**. Ревизия после провала v5.0 (SDQ Acc++ 0.40@70M против 0.777@50M у v4), где все столбцы были линейными LRU. Диагноз v5.0: (а) линейной рекурренции нечем **связывать** — «запиши V, если пришёл K» требует мультипликативного гейтирования, которое у GRU есть каждый такт, а у LRU нет вовсе (Acc/distract 0.95, Acc/query 0.49); (б) двойная экспонента репараметризации λ = exp(−exp(ν)) взрывает градиенты (|Grad| = 8 при клипе 1.0 — каждый шаг урезался в 8 раз); (в) 4–5 отдельных матмулов на LRU-ячейку на такт — fps 4.5k против ~10k у v4.

## Ключевой механизм

Разделение труда по типам ячеек: связывание — нелинейным GRU, долгое хранение — линейным LRU с гарантированным полом удержания:

```python
if ic < self.n_gru:
    cell = nn.GRUCell(in_dim, H)          # fast/medium: multiplicative binding
    cell.bias_ih[H:2*H] += shift          # timescale stagger (v4)
else:
    cell = FastFloorLRUCell(in_dim, H, r_floor=floor)   # slow storage, |lambda| >= 0.9
```

Состояние упаковано в общий тензор [L, C, B, 2H]: GRU-столбцы используют первую половину, LRU — обе (re/im).

## FastFloorLRUCell — ускоренная и стабилизированная LRU

```python
self.B = nn.Linear(input_size, 2 * hidden_size, bias=False)  # merged B_re+B_im: 1 matmul
# no D feedthrough
y = self.C(h_n) * torch.sigmoid(self.G(h_n))                 # GLU: nonlinearity per step
self.nu.register_hook(lambda g: g * grad_scale)              # 0.1: damp double-exp gradients
```

Три исправления против v5.0: объединённая B-проекция и отказ от D (меньше кернел-лончей и параметров), GLU-нелинейность на выходе (у чистого LRU её не было), демпфер градиентов ν/θ ×0.1 (лечит |Grad|=8 без param-groups в раннере).

## Важные детали реализации

Остальная машинерия — из v4/v5.0: персональное внимание с hub-источником (`HubColumnAttention`, C приёмников × C+1 источников), скалярные гейты, аддитивное сообщение с защитой рекуррентного состояния, RMSNorm между слоями, concat-readout, aux-лоссы (Barlow с весом по глубине, gate-std, activity; сатурационный штраф — только по GRU-столбцам). Диагностика `lru/r_*` в раннерах показывает истинный |λ| LRU-столбцов с учётом пола; `attn_beta/L_C` — эволюцию температур.

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `n_lru_cols` | 1 — сколько столбцов с конца являются LRU-хранилищем |
| `hidden_size` | 60 при 2L×3C (2 GRU + 1 LRU) — ~196K активных параметров |
| `r_floor_min/max` | 0.9 / 0.95 — полы удержания LRU-столбцов |
| `lru_grad_scale` | 0.1 — демпфер градиентов ν/θ |
| `timescale_spread` | 1.0 — разброс bias update-гейта GRU-столбцов |
