# Подключение к aicenter3

Доступ к серверу идёт в два шага: сначала сеть netbird, затем SSH. Работать на сервере можно только отправляя команды по SSH (интерактивных сервисов, GUI и проброса портов не настроено).

## Данные

| Что | Значение |
|---|---|
| Сервер | aicenter3 (хост `node207-28`) |
| IP в netbird | `100.98.241.137` |
| Домен | `aicenter3.cogmodel.mipt` |
| Пользователь | `annenkov_vd` |
| Management URL netbird | `https://nettouse.ru` |
| Этот компьютер в netbird | hostname `MasterPC`, FQDN `masterpc.cogmodel.mipt`, IP `100.98.130.53` |
| SSH-ключ | `/mnt/c/Users/master/.ssh/id_ed25519` (взят из Windows) |

Пароль и setup-ключ в этот файл не записаны. Setup-ключ нужен только при первой регистрации машины.

## Шаг 1. Netbird (WSL)

Клиент установлен в WSL (`/usr/bin/netbird`, версия 0.79.0), машина уже зарегистрирована и авторизована администратором.

```bash
netbird status      # должно быть: Management: Connected, Signal: Connected
netbird up          # если статус Disconnected или NeedsLogin после перезагрузки
```

Если `NeedsLogin` (регистрация слетела), нужен новый setup-ключ от администратора (ФИО и hostname `MasterPC`):

```bash
netbird up --management-url https://nettouse.ru --setup-key <KEY>
```

После этого сообщите администратору hostname и дождитесь авторизации. Отключиться: `netbird down`.

Проверка связи: `ping -c2 100.98.241.137`. Если ping не идёт, а netbird подключён, проверьте, не перехватывает ли трафик прокси или VPN в режиме TUN (адаптер `198.18.0.1`). Тогда добавьте обход для подсети netbird:

```bash
sudo ip rule add to 100.98.0.0/16 table main priority 100
```

Правило не сохраняется после перезагрузки.

Домен `aicenter3.cogmodel.mipt` резолвится только при включённом DNS netbird. В WSL по умолчанию надёжнее ходить по IP.

## Шаг 2. SSH

OpenSSH отказывается использовать ключ с открытыми правами, а на `/mnt/c` они `777`. Поэтому ключ копируется в домашний каталог WSL:

```bash
mkdir -p ~/.ssh
cp /mnt/c/Users/master/.ssh/id_ed25519 ~/.ssh/aicenter3_key
chmod 600 ~/.ssh/aicenter3_key
```

Запись в `~/.ssh/config`:

```
Host aicenter3
    HostName 100.98.241.137
    User annenkov_vd
    IdentityFile ~/.ssh/aicenter3_key
    IdentitiesOnly yes
```

Проверка:

```bash
ssh aicenter3 hostname      # ожидаемый ответ: node207-28
```

Без записи в config:

```bash
ssh -i ~/.ssh/aicenter3_key -o IdentitiesOnly=yes annenkov_vd@100.98.241.137 hostname
```

Если ключ не принят, публичный ключ должен лежать на сервере в `~/.ssh/authorized_keys`. Публичная часть: `/mnt/c/Users/master/.ssh/id_ed25519.pub`.

## Как работать на сервере

Только отправкой команд по SSH, по одной за вызов:

```bash
ssh aicenter3 'nvidia-smi'
ssh aicenter3 'ls -la /storage /mnt/aicenter1-datasets'
ssh aicenter3 'cd /storage/annenkov_vd/proj && CUDA_VISIBLE_DEVICES=3 python train.py'
```

Для фоновых задач запускайте их через `nohup` или `tmux` в одной команде и забирайте результат отдельными вызовами:

```bash
ssh aicenter3 'cd /storage/annenkov_vd/proj && CUDA_VISIBLE_DEVICES=3 nohup python train.py > train.log 2>&1 &'
ssh aicenter3 'tail -50 /storage/annenkov_vd/proj/train.log'
```

Копирование файлов: `scp file aicenter3:/storage/annenkov_vd/proj/` или `rsync -av ./dir/ aicenter3:/storage/annenkov_vd/proj/dir/` (без `--delete`).

Интерактивный вход (`ssh aicenter3` без команды) нужен только для действий с TTY, например `passwd`.

## Обязательные ограничения

1. **Удалять можно только файлы, созданные самим агентом в `/storage/annenkov_vd/`.** Любые другие файлы и каталоги (чужие данные, `/mnt/aicenter1-datasets`, домашние каталоги других пользователей, системные файлы) не удалять, не перезаписывать и не перемещать. Сервер общий.
2. **Использовать можно только GPU 3.** Все запуски с `CUDA_VISIBLE_DEVICES=3`. Другие карты (0, 1, 2, 4, 5) не занимать, даже если они свободны.

Перед удалением проверяйте путь: он должен начинаться с `/storage/annenkov_vd/` и указывать на объект, созданный агентом. Не используйте `rm -rf` с переменными и масками без проверки, что они раскрываются внутрь этого каталога.

## Хранилища

- `/storage` — s3-хранилище для постоянных данных. Рекомендуется для хранения. Рабочий каталог: `/storage/annenkov_vd/` (создаётся при первом использовании: `ssh aicenter3 'mkdir -p /storage/annenkov_vd'`).
- `/mnt/aicenter1-datasets` — общие датасеты (Argoverse2, KITTI, nuScenes, Ego4D, веса моделей и др.). Только чтение.

## GPU

Шесть H100 80GB (индексы 0-5), но разрешена только **GPU 3**. Перед запуском проверьте её занятость:

```bash
ssh aicenter3 'nvidia-smi -i 3 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv'
ssh aicenter3 'nvidia-smi -i 3 --query-compute-apps=pid,process_name,used_memory --format=csv'
```

Если карта занята чужим процессом, не запускайте задачу и не завершайте чужие процессы: сообщите об этом.

Запуск строго на GPU 3:

```bash
ssh aicenter3 'cd /storage/annenkov_vd/proj && CUDA_VISIBLE_DEVICES=3 python train.py'
```

## Типичные проблемы

| Симптом | Причина и решение |
|---|---|
| `Connection timed out` на порту 22 | Netbird не подключён или трафик перехватывает прокси. Проверьте `netbird status`, `ping`, правило `ip rule` |
| `Connection closed by ... port 22` сразу после установления | Соединение принимает локальный TUN-прокси (`198.18.0.1`), а не сервер. Нужен обход для `100.98.0.0/16` |
| `Could not resolve hostname aicenter3.cogmodel.mipt` | Не работает DNS netbird. Используйте IP |
| `Your password has expired` / `Password change required but no TTY` | Смените пароль в интерактивном сеансе: `ssh aicenter3` и `passwd` |
| `UNPROTECTED PRIVATE KEY FILE` | Права на ключ: `chmod 600 ~/.ssh/aicenter3_key` |
| `netbird: command not found` | Установите: `curl -fsSL https://pkgs.netbird.io/install.sh \| sh`. Если apt сломан, сначала `sudo apt --fix-broken install` |
