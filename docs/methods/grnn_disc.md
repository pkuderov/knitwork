# GridRnnNoveltyGate (grnn_disc)

Модель решает проблему бесполезного обмена сообщениями между колонками Grid RNN: стандартный непрерывный gate не умеет явно игнорировать сообщения, которые не несут новой информации. Основная идея — заменить линейный attention gate на `NoveltyGate`, который вычисляет новизну сообщения относительно текущего скрытого состояния через косинусное расстояние, а затем квантует gate в дискретное множество `{0, 0.5, 1}` с помощью straight-through estimator: `0` — сообщение несёт старую информацию и игнорируется, `0.5` — частичное обновление, `1` — полная замена состояния новым сообщением.

## Ключевой механизм

```python
# novelty = cosine distance, mapped to [0, 1]
cos_sim = F.cosine_similarity(h_new, msg, dim=-1, eps=1e-8)  # [cols, batch]
novelty = (1.0 - cos_sim) / 2.0                              # [cols, batch]

# straight-through discretization: forward=discrete, backward=continuous
discrete = torch.where(score < lo, GATE_LOW,
           torch.where(score > hi, GATE_HIGH, GATE_MID))
return score + (discrete - score).detach()                   # [cols, batch, 1]
```

Операция `score + (discrete - score).detach()` в forward-проходе возвращает дискретное значение, но при backprop градиент идёт через непрерывный `score`, обходя недифференцируемое ветвление.

## Важные детали реализации

**Смешение косинусной и обученной новизны:**

```python
# blend raw cosine novelty with learned correction
raw     = self._raw_novelty(h_new, msg)                             # [cols, batch, 1]
learned = self.novelty_proj(torch.cat([h_new, msg], dim=-1))        # [cols, batch, 1]
blend   = torch.sigmoid(self.blend)                                 # scalar in (0,1)
score   = (1.0 - blend) * raw + blend * learned
```

`self.blend` — обучаемый скаляр, инициализированный значением `0.1`, чтобы на старте обучения доминировало простое косинусное расстояние.

**Применение gate в grid step:**

```python
# discrete novelty gate replaces standard sigmoid gate
g    = nov_gate(hl_n, msg)          # [cols, batch, 1] in {0.0, 0.5, 1.0}
hl_n = (1.0 - g) * hl_n + g * msg  # selective state update
```

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `novelty_low` | Нижний порог новизны; сообщения с оценкой ниже → gate=0 (игнор) |
| `novelty_high` | Верхний порог; сообщения выше → gate=1 (полная замена) |
| `GATE_LOW / GATE_MID / GATE_HIGH` | Фиксированные дискретные значения gate: `0.1`, `0.4`, `0.6`; не `0/0.5/1` буквально, что смягчает экстремальные обновления |
