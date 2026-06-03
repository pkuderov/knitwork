# GridRnn (grnn_adv_loss)

Модель решает проблему коллапса столбцов в Grid RNN — ситуацию, когда внутренние колонки сетки обучаются представлениям, похожим друг на друга, и перестают нести различную информацию. Основная идея: ввести вспомогательный лосс `ColumnSpecializationLoss`, который явно штрафует за декорреляцию, низкую дисперсию, косинусное сходство и «неотбеленность» скрытых состояний колонок, вынуждая их специализироваться на разных аспектах входа. Параллельно `MessagePassingLayer` дополняется усиленными identity-якорями, поколоночными проекциями Q/K и post-attention нелинейностью, физически разводящими пространства колонок.

## Ключевой механизм

```python
# compute specialization loss over hidden states [L, C, B, D]
if compute_spec_loss:
    spec_loss, spec_details = self.spec_loss(h_new)
    extras["spec_loss"]    = spec_loss * self.spec_loss_weight
    extras["spec_details"] = spec_details
```

`ColumnSpecializationLoss` принимает полный тензор скрытых состояний `h_new` формы `[layers, cols, batch, D]` и возвращает скалярный лосс, взвешенный коэффициентом `spec_loss_weight`. Лосс складывается из четырёх компонент (декорреляция, дисперсия, косинус, отбеливание), каждая со своим `lambda`.

## Важные детали реализации

**Усиленные identity-якоря в MessagePassingLayer:**

```python
# larger std => columns start further apart in embedding space
nn.init.normal_(self.ids, 0.0, 0.1 * xavier_alpha)  # vs 0.01 in base grnn
```

Увеличенное стандартное отклонение при инициализации `ids` не позволяет колонкам схлопнуться к одному представлению на ранних шагах обучения.

**Поколоночные проекции Q/K:**

```python
# column-specific Q/K fingerprint via low-rank projection
proj = torch.einsum('cbd,cdp->cbp', qh, self.col_proj)        # [C, B, proj_dim]
proj = torch.einsum('cbp,cpd->cbd', proj, self.col_proj_out)  # [C, B, D]
qh = kh = qh + 0.1 * proj  # residual, keeps main signal intact
```

Каждый столбец имеет свою пару матриц проекций для формирования уникального «отпечатка» в пространстве запросов и ключей.

**Post-attention нелинейность:**

```python
# per-column nonlinear transform after MHA
for c in range(C):
    out_list.append(h_mixed[c] + self.post_proj[c](h_mixed[c]))
```

Каждый столбец имеет свою двухслойную MLP с SiLU-активацией, применяемую поверх выхода attention как residual.

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `spec_lambda_decorr` | Вес компоненты декорреляции в `ColumnSpecializationLoss`; управляет тем, насколько сильно штрафуется линейная зависимость между колонками |
| `spec_lambda_var` | Вес компоненты дисперсии; штрафует за малую внутриколоночную вариативность |
| `spec_lambda_cosine` | Вес косинусного штрафа; прямой штраф за угловое сходство представлений |
| `spec_lambda_whiten` | Вес штрафа за отбеливание; требует изотропного распределения активаций |
| `spec_target_layers` | Список слоёв, к которым применяется лосс специализации; `None` = все слои |
| `spec_loss_weight` | Итоговый скалярный множитель перед добавлением в общий лосс; позволяет анилировать по расписанию |

## Результаты

Конфигурация: H=128, 4 колонки, 4 слоя.

### SDQ (Store-Distract-Query, hard)

| Эксперимент | Acc | Acc++ | Loss | Шагов |
|---|---|---|---|---|
| grnn loss adv sdq (`grid-rnn-sdq`) | **0.807** | **0.629** | **0.505** | ~92м |

Результат сопоставим с grnn\_fusion v1 (Acc=0.831) и выше grnn\_disc (Acc≤0.655), но ниже grnn\_loss (Acc=0.862). Многокомпонентный `ColumnSpecializationLoss` (декорреляция + дисперсия + косинус + отбеливание) вместе с усиленными identity-якорями и поколоночными Q/K проекциями даёт ощутимый прирост по сравнению с базовым grnn 2/1 (Acc=0.734), однако более простой diversity loss в grnn\_loss оказывается эффективнее.
