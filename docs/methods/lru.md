# LRUCell

`LRUCell` реализует одношаговую рекуррентность в комплексном пространстве состояний на основе Linear Recurrent Unit. Ключевая идея: диагональная рекуррентная матрица Λ в комплексных числах позволяет явно управлять диапазоном временной памяти через спектральный радиус |λ| ∈ [r_min, r_max], при этом параметры ν и θ обучаются в log-пространстве, что гарантирует устойчивость градиентов. Ячейка решает проблему затухающих/взрывных градиентов стандартных RNN, сохраняя долгосрочные зависимости без механизмов типа LSTM/GRU.

## Ключевой механизм

```python
# u: [B, input_size]  h: [B, 2*hidden_size] — real + imag packed
def forward(self, u, h):
    h_re, h_im = h[:, :H], h[:, H:]
    lam_re, lam_im, gamma = self._lambda_gamma()   # derived from nu, theta

    # complex multiplication: Λ * h + γ * B * u
    new_re = lam_re * h_re - lam_im * h_im + gamma * self.B_re(u)
    new_im = lam_re * h_im + lam_im * h_re + gamma * self.B_im(u)
    h_n = torch.cat([new_re, new_im], dim=-1)      # [B, 2H]

    y = self.C(h_n)                                 # [B, H] — real output
    return y, h_n
```

`gamma = sqrt(1 - |λ|²)` нормирует вход так, чтобы энергия сигнала не зависела от выбранного радиуса λ.

## Важные детали реализации

**Параметризация λ через log-пространство.** Спектральный радиус |λ| = exp(-exp(ν)) всегда строго меньше 1 для любых вещественных ν, а фаза θ = exp(θ_param) ∈ (0, max_phase) задаёт вращение:

```python
log_r     = -torch.exp(self.nu)                    # log(|lambda|) <= 0
lambda_re = torch.exp(log_r) * torch.cos(torch.exp(self.theta))
lambda_im = torch.exp(log_r) * torch.sin(torch.exp(self.theta))
gamma     = torch.sqrt((1.0 - torch.exp(2.0 * log_r)).clamp(min=1e-6))
```

**Векторизованный `forward_sequence`.** Проекции B и D вычисляются одним вызовом на всей последовательности `[T*B, ...]`, а рекуррентный цикл проходит только по временному измерению:

```python
Bu_re = self.B_re(u_flat).view(T, B, -1)   # one matmul for entire sequence
for t in range(T):
    new_re = lam_re * h_re - lam_im * h_im + gamma * Bu_re[t]
```

**`LRUBlock`.** Обёртка над `LRUCell` с нормировками и feed-forward: `RMSNorm → LRUCell → GLU → residual → RMSNorm → PFFN → residual`. Инициализация последнего слоя FF с малым std (`0.01/sqrt(H)`) подавляет начальный вклад FF.

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `r_min`, `r_max` | Диапазон спектрального радиуса при инициализации; ν инициализируется равномерно в [r_min, r_max] |
| `max_phase` | Максимальная начальная фаза θ (по умолчанию 2π); ограничивает начальную скорость вращения состояния |
| `use_d_feedthrough` | Если `True` — добавляет прямой путь D·u к выходу y, улучшая аппроксимацию сигналов с высокой частотой |
| `ff_mult` | (в `LRUBlock`) Множитель размера feed-forward слоя относительно hidden_size |

## Результаты

`LRUCell` является компонентом и не тестируется напрямую. Результаты моделей, использующих LRU как рекуррентную ячейку, приведены в:
- [grnn\_lru.md](grnn_lru.md) — Grid RNN на основе LRU (SDQ: Acc=0.849, shakespeare Loss=1.268)
- [hgrnn\_lru.md](hgrnn_lru.md) — LRU + Hopfield attention (SDQ: Acc=0.967 — лучший результат по Acc++)
