# HopfieldGridLRU

Модель объединяет три идеи для улучшения Grid RNN на задачах ассоциативной памяти и языкового моделирования: (1) Linear Recurrent Unit (LRU) с диагональной параметризацией вместо LSTM — меньше параметров, стабильный градиент на длинных последовательностях; (2) Modern Hopfield Network для обмена сообщениями между колонками; (3) вспомогательный контрастный ассоциативный лосс, явно обучающий сеть разделять паттерны «запись» и «чтение». LRU работает в комплексном пространстве: состояние хранится как `[Re | Im]` — компактная форма, из которой Hopfield-слой видит только Re-часть.

## Ключевой механизм

LRUCell параметризует диагональную матрицу перехода через `nu_log` и `theta_log`, обеспечивая |λ| ∈ (0,1) по построению:

```python
# stable reparametrization of lambda: |lambda| = exp(-exp(nu)), angle = exp(theta)
r      = torch.exp(-torch.exp(self.nu_log))     # |lambda| in (0,1)
theta  = torch.exp(self.theta_log)
lam_re = r * torch.cos(theta)
lam_im = r * torch.sin(theta)
# gamma normalizes input contribution inversely to memory strength
gamma  = torch.sqrt(torch.clamp(1.0 - r * r, min=1e-6))

# complex state update (expanded to 4 real ops, no overhead)
new_re = lam_re * h_re - lam_im * h_im + gamma * bx_re  # [B, H]
new_im = lam_re * h_im + lam_im * h_re + gamma * bx_im  # [B, H]
# output: Re(C * h_new) + D * x
y = self.C_re(new_re) - self.C_im(new_im) + self.D(x)
```

`gamma = sqrt(1 - |λ|²)` — ключевое отличие от S4: чем сильнее «память» (|λ| → 1), тем слабее влияние нового входа.

## Важные детали реализации

**Detach Im-части при сборке состояния.** Im-часть не участвует в Hopfield-обмене и gate, поэтому её градиент уже учтён внутри LRUCell. Накопление графа через Im на длинных роллаутах привело бы к OOM:

```python
hl_im_stop  = hl_full[:, :, self.hidden_size:].detach()   # Im-part detached
hl_full_new = torch.cat([hl_re_gated, hl_im_stop], dim=-1)
```

**Ассоциативный контрастный лосс.** Штрафует за близость представлений случайных пар «запись/чтение» и поощряет близость соответствующих пар:

```python
sim_matrix = torch.matmul(h_query, h_store.T)   # (n, n) cosine similarity
cos_pos    = sim_matrix.diagonal()               # positive pairs
cos_neg    = sim_matrix.masked_fill(eye_mask, -1.0).max(dim=-1).values
loss       = (-cos_pos + F.relu(cos_neg + margin)).mean()
```

**PositionwiseFFN после каждого LRU.** LRUCell — линейная рекуррентность. FFN с Pre-LN и GELU добавляет нелинейность блока, как в Orvieto et al. 2023:

```python
# Pre-LN + GELU FFN with residual connection
self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * expansion),
                         nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(dim * expansion, dim), nn.Dropout(dropout))
def forward(self, x): return x + self.net(x)
```

**reset_state без clone.** Сброс состояния реализован через умножение на keep-маску — дешевле, чем `clone()` + индексирование:

```python
keep = (~reset_mask).to(dtype=state.dtype, device=state.device)
return state * keep.view(1, 1, -1, 1)   # broadcast over (layers, cols, batch, 2*hid)
```

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `lru_r_min`, `lru_r_max` | Диапазон |λ| при инициализации; нижняя граница 0.4 обеспечивает минимальную «память» |
| `lru_max_phase` | Максимальный начальный угол θ; 2π/3 даёт разнообразие начальных частот |
| `ffn_expansion` | Множитель расширения в FFN (обычно 2–4) |
| `attn_dropout` | Dropout на attention-весах Hopfield-слоя |
| `log_beta` | Обучаемый масштаб attention (один на голову); инициализируется как `log(1/√d_k)` |

## Результаты

### SDQ (Store-Distract-Query, hard)

| Конфигурация | H | Столб. / Слоёв | Acc | Acc++ | Loss | Шагов |
|---|---|---|---|---|---|---|
| grnn\_hopfield H=128 (`sdq-hgru-hopfield`) | 128 | 4 / 3 | **0.967** | **0.932** | **0.087** | ~40м |
| grnn\_hopfield 5col H=116 (`sdq-hgru-hopfield`) | 116 | 5 / 3 | 0.628 | 0.284 | 0.974 | ~12м |

Конфигурация 4 кол. / 3 сл. — лучший результат среди всех протестированных моделей на SDQ: Acc=0.967, Acc++=0.932 всего за 40м шагов, опережая grnn\_hgru (0.965 за 45м) и базовый grnn (0.960 за 57м). Комбинация LRU (диагональная рекуррентность) + Hopfield-attention (обучаемый β) + контрастный лосс обеспечивает максимальный Acc++. 5-колоночный запуск остановлен досрочно (12м).

### Текстовые эксперименты (shakespeare)

| Конфигурация | H | Столб. / Слоёв | Acc | BPC | PPL | Шагов |
|---|---|---|---|---|---|---|
| grnn\_hopfield H=128 (`text-hgru-hopfield`) | 128 | 4 / 3 | **0.636** | **1.686** | **3.22** | ~70м |
| grnn\_hopfield 5col H=116 (`text-hgru-hopfield`) | 116 | 5 / 3 | 0.635 | 1.690 | 3.23 | ~70м |
| grnn\_lru\_hop H=104 (`text-lru`) | 104 | 3 / 3 | 0.330 | 3.844 | 14.35 | ~17м |

На shakespeare результаты 4/3 конфигурации (BPC=1.686, PPL=3.22) практически идентичны grnn\_hgru (BPC=1.686) — Hopfield vs стандартный attention не даёт разницы в текстовых экспериментах. Вариант grnn\_lru\_hop в `text-lru` завершён очень рано и расходился (BPC=3.844, PPL=14.35).
