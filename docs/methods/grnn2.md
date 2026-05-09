# GridRnn2

GridRnn2 расширяет базовую GridRnn тремя независимо включаемыми механизмами для улучшения обобщения и стабильности обучения. VAE-bottleneck на входе заменяет детерминированный эмбеддинг вероятностным: вместо точного вектора сеть получает сэмпл из гауссовского распределения, параметры которого предсказываются из эмбеддинга, — это регуляризует представления и добавляет штраф KL к лоссу. Column Time-Gate позволяет каждой колонке примешивать состояние левой соседней колонки с предыдущего шага, создавая «волну» обработки слева направо. Column Dropout случайно обнуляет целые колонки во время обучения, вынуждая каждую колонку быть полезной независимо от соседей.

## Ключевой механизм

VAE-эмбеддинг через трюк репараметризации позволяет градиентам проходить через стохастическое сэмплирование:

```python
def reparameterize(self, mu, log_var):
    if self.training:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std   # [batch, latent_dim]
    return mu                   # deterministic at inference

# KL penalty added to main loss
kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
return x, kl * self.kl_weight
```

На инференсе возвращается детерминированное `mu` — без шума.

## Важные детали реализации

Column Time-Gate смешивает свежее состояние колонки `j` с состоянием колонки `j-1` с предыдущего шага через обучаемые элементарные ворота:

```python
# h_new, h_prev: [cols, batch, hidden]
combined = torch.cat([h_new[j], h_prev[j - 1]], dim=-1)
g = torch.sigmoid(gate_fn(combined))           # [batch, hidden]
mixed = (1.0 - g) * h_new[j] + g * h_prev[j - 1]
```

Bias ворот инициализируется отрицательным числом (`delay_scale=-2.0`), поэтому изначально ворота почти закрыты и не мешают обучению.

---

Column Dropout масштабирует оставшиеся колонки на `1/(1-p)`, чтобы ожидаемое значение не изменялось:

```python
scale = 1.0 / (1.0 - self.drop_prob + 1e-8)
for i, col_idx in enumerate(range(start, self.n_columns)):
    if not keep[i]:
        result[col_idx] = 0.0
    else:
        result[col_idx] = result[col_idx] * scale
```

Нулевая (внешняя) колонка всегда сохраняется (`keep_first=True`).

---

`forward` возвращает тройку `(logits, h, kl_loss)` вместо пары, поэтому вызывающий код должен суммировать `kl_loss` с основным cross-entropy:

```python
y, h, kl_loss = model(tokens, h)
loss = ce_loss + kl_loss   # kl_loss already scaled by kl_weight
```

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `vae_latent_dim` | Размерность латентного пространства VAE; `None` отключает VAE и использует стандартный `nn.Embedding` |
| `vae_kl_weight` | Вес KL-штрафа; малое значение (1e-4..1e-2) не даёт регуляризации подавлять основной лосс |
| `use_time_gate` | Включает Column Time-Gate; `False` — поведение идентично базовой GridRnn |
| `time_gate_delay_scale` | Начальный bias ворот задержки; отрицательное значение = ворота почти закрыты на старте |
| `col_drop_prob` | Вероятность обнуления одной колонки за шаг; `0.0` отключает Column Dropout |
