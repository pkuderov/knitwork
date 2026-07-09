---
name: status
description: Use this skill when the user types "/status" or asks about the current state of experiments on the remote server — running jobs, queue, metrics, logs. Triggers on "статус экспериментов", "что запущено", "покажи метрики", "посмотри очередь", "результаты экспериментов", "check server status", "experiment status".
version: 1.0.0
---

# Status Skill

Подключается к серверу и выводит полный статус: очередь, метрики запущенных экспериментов, последние завершённые.

---

## Порядок действий

### Шаг 1 — Поднять WireGuard (если нужно)

Сначала проверь, активен ли интерфейс:

```bash
wg show 2>/dev/null | grep -q wg-knitwork && echo "up" || echo "down"
```

Если `down`:

```bash
echo 'q123e' | sudo -S wg-quick up wg-knitwork 2>&1
```

Проверь связь:

```bash
ping -c 1 -W 3 10.8.0.1 2>&1 | grep -q "1 received" && echo "ok" || echo "no route"
```

Если `no route` — сервер выключен, сообщи пользователю.

### Шаг 2 — Установить MTU (критично)

При первом подключении после поднятия WireGuard туннель зависает при передаче данных (MTU-проблема через playit.gg). Всегда устанавливать MTU 1280 перед любой работой:

```bash
echo 'q123e' | sudo -S ip link set wg-knitwork mtu 1280 2>&1
```

Никогда не использовать ssh с большим выводом напрямую — зависает. Правильный паттерн:

```bash
# 1. Записать на сервере в файл
ssh knitwork-server-global "команда > /tmp/out.txt 2>&1 && echo done"
# 2. Скачать через scp
scp knitwork-server-global:/tmp/out.txt /tmp/out_local.txt
# 3. Прочитать локально через Read tool
```

Исключение: команды с выводом < 200 байт можно запускать напрямую.

### Шаг 3 — Получить статус очереди

```bash
ssh knitwork-server-global \
  "cd ~/knitwork && /opt/uv-envs/knitwork/.venv/bin/python server/queue_status.py > /tmp/qstatus.txt 2>&1 && echo done"
scp knitwork-server-global:/tmp/qstatus.txt /tmp/qstatus_local.txt
cat /tmp/qstatus_local.txt
```

Вывод содержит три секции: `RUNNING`, `PENDING`, `COMPLETED (last 5)`.

Формат строки running-эксперимента:
```
#076 'grnn_harmonic MIKASA Easy v6'  pid=356161  started=2026-06-17T19:50:18+00:00
    [20.00M / 200M] 216 fps | LR:79% | PL:-0.000 VL:0.007 H:1.32 R:-0.009 EpRet:-0.496
```

Расшифровка метрик:
- `LR` — % выполнения LR-расписания (warmup + cosine decay)
- `PL` — Policy Loss (PPO surrogate)
- `VL` — Value Loss (MSE critic)
- `H` — энтропия политики (> 1.0 в начале, > 0.5 в конце)
- `R` — средний reward за шаг
- `EpRet` — суммарный return за эпизод (EMA, **главная метрика**)

### Шаг 4 — Подробные метрики из логов

```bash
ssh knitwork-server-global \
  "for f in ~/knitwork_logs/007*.log; do echo \"=== $f ===\"; grep -E '^\[' $f | tail -5; done > /tmp/metrics.txt 2>&1 && echo done"
scp knitwork-server-global:/tmp/metrics.txt /tmp/metrics_local.txt
cat /tmp/metrics_local.txt
```

Список логов (короткий вывод — можно напрямую):
```bash
ssh knitwork-server-global "ls -lht ~/knitwork_logs/ | head -20"
```

### Шаг 5 — Comet.ml метрики (для запущенных экспериментов)

Comet.ml — предпочтительный способ смотреть метрики **во время выполнения** (не блокируется в отличие от AIM).

- **URL:** https://www.comet.com/adran-dasadfan/knitwork
- **API key:** `P14Y6mW8ymn1Z441hlnhx4lYw` (хранится в `agent.md`)

Эксперименты логируют в Comet только если запущены с `--log.logger=comet`. Treasure Hunt — только AIM.

Запуск с Comet:
```bash
bash server/knitwork_run.sh knitwork/exps/sdq/run_sdq.py \
    knitwork/exps/sdq/config/extend_config.yaml \
    --model=grnn --log.logger=comet -- --name "grnn sdq"
```

### Шаг 6 — AIM метрики (только после завершения экспериментов)

**Важно:** Пока эксперименты запущены, AIM база заблокирована (RocksDB LOCK). `query_results.py` выдаст пустые строки. Запускать только для завершённых экспериментов.

```bash
ssh knitwork-server-global \
  "cd ~ && /opt/uv-envs/knitwork/.venv/bin/python ~/knitwork/server/query_results.py --last 20 > /tmp/qresults.txt 2>&1 && echo done"
scp knitwork-server-global:/tmp/qresults.txt /tmp/qresults_local.txt
cat /tmp/qresults_local.txt
```

Опции: `--last 20`, `--model grnn`, `--sort val_acc`, `--metric reward`

**Известная проблема `query_results.py`:** AIM API изменился — `run.config` не существует, параметры хранятся в `run['hparams']`. Если падает с `AttributeError`, в [server/query_results.py](../../server/query_results.py) должна использоваться функция `get_hparams(run)`.

---

## Справочник окружения сервера

| Что | Значение |
|---|---|
| SSH (локальная сеть) | `knitwork-server` → 192.168.1.58:2222 |
| SSH (WireGuard/global) | `knitwork-server-global` → 10.8.0.1:2222 |
| WireGuard интерфейс | `wg-knitwork` |
| sudo пароль | `q123e` |
| Python venv | `/opt/uv-envs/knitwork/.venv/bin/python` |
| Проект | `~/knitwork/` |
| Логи экспериментов | `~/knitwork_logs/` |
| AIM база | `~/.aim/` (корень home, не в knitwork!) |
| Очередь (JSON) | `~/knitwork_queue.json` |
| Макс. параллельных | 3 |
| Comet.ml проект | https://www.comet.com/adran-dasadfan/knitwork |

---

## Частые проблемы

### SSH зависает при передаче данных

```bash
echo 'q123e' | sudo -S ip link set wg-knitwork mtu 1280 2>&1
```

Всегда использовать паттерн: записать в /tmp → scp → читать локально.

### `Connection closed by 10.8.0.1 port 2222`

```bash
echo 'q123e' | sudo -S wg-quick up wg-knitwork 2>&1
echo 'q123e' | sudo -S ip link set wg-knitwork mtu 1280 2>&1
```

### `No route to host` для knitwork-server (192.168.1.58)

Клиент не в локальной сети — использовать `knitwork-server-global`.

### Очередь не запускает эксперименты

```bash
ssh knitwork-server-global "systemctl --user restart knitwork-queue" && \
ssh knitwork-server-global "systemctl --user status knitwork-queue > /tmp/svc.txt 2>&1" && \
scp knitwork-server-global:/tmp/svc.txt /tmp/svc_local.txt && cat /tmp/svc_local.txt
```

### playit упал на сервере

```bash
ssh knitwork-server "sudo systemctl restart playit"
# работает только если локальная сеть доступна
```

---

## Быстрый однострочник — полный статус

```bash
echo 'q123e' | sudo -S ip link set wg-knitwork mtu 1280 2>&1
ssh knitwork-server-global "
  cd ~/knitwork
  /opt/uv-envs/knitwork/.venv/bin/python server/queue_status.py > /tmp/status_full.txt 2>&1
  echo '=== LAST LOG LINES ===' >> /tmp/status_full.txt
  for f in \$(ls -t ~/knitwork_logs/*.log | head -6); do
    echo \"--- \$f\" >> /tmp/status_full.txt
    grep -E '^\[' \$f | tail -3 >> /tmp/status_full.txt
  done
  echo done
" && scp knitwork-server-global:/tmp/status_full.txt /tmp/status_full_local.txt && cat /tmp/status_full_local.txt
```

---

## Интерпретация метрик

### MIKASA / POPGym (RL)

Главная метрика: **EpRet** (суммарный return за эпизод).

| EpRet | Интерпретация |
|---|---|
| < -0.8 | Случайная политика или хуже |
| -0.5 .. -0.2 | Обучение идёт, но медленно |
| -0.2 .. 0.0 | Хороший прогресс |
| > 0.0 | Модель освоила задачу |

Энтропия H: начало > 1.5 (исследование), середина ~1.0, конец < 0.3 — модель схлопнулась (плохо если EpRet низкий).

### SDQ (Store-Distract-Query)

Главная метрика: **val_acc** (точность ответов на запросы из памяти).

### Text (shakespeare / text8)

Главная метрика: **val_loss** (bits per character).

### TreasureHunt (RL)

Главная метрика: **ep_return** (суммарная награда за эпизод).
