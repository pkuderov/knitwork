# Shiroa — Typst-книга документации

Второй (параллельный) вид документации к [`docs/`](../docs): `docs/` рендерится
как обычный сайт [Docsify](https://docsify.js.org/) (просто `.md` + `index.html`),
а `shiroa/` пересобирает те же `.md`-файлы в книгу через
[Typst](https://typst.app/) + [Shiroa](https://github.com/Myriad-Dreamin/shiroa) —
даёт красивую типографику, PDF-экспорт и постраничную навигацию (mdBook-like).

Оба варианта деплоятся на один и тот же GitHub Pages одним workflow:
Docsify — в корень сайта, Shiroa-книга — в `/typst/`.

## Как это устроено (pipeline)

```
docs/methods/*.md ─┐
docs/experiments/*.md ─┼─ pandoc (gfm → typst) → fix_for_shiroa.py ─→ shiroa/methods/*.typ
docs/README.md ─┘                                                     shiroa/experiments/*.typ
                                                                       shiroa/README.typ

docs/_sidebar.md ──── gen_summary.py ──→ shiroa/book.typ  (из шаблона book.typ.tmpl)

shiroa/book.typ + shiroa/methods/*.typ + templates/ ──→ `shiroa build` ──→ shiroa/dist/ (HTML-книга)
```

Всё это выполняет один скрипт: `bash shiroa/build-content.sh --build`.

### Файлы

| Файл | Роль |
|---|---|
| `shiroa/build-content.sh` | Оркестратор: md→typst конвертация + генерация summary + `shiroa build` + инъекция CSS |
| `shiroa/fix_for_shiroa.py` | Постобработка pandoc-вывода (typst-синтаксис, который pandoc генерирует криво: `#align`, `//`, `/ ` и т.д.) |
| `shiroa/gen_summary.py` | Генерирует `book.typ` из `book.typ.tmpl` + `docs/_sidebar.md` |
| `shiroa/book.typ.tmpl` | Шаблон книги (метаданные, плейсхолдер `__SUMMARY__` для оглавления). **Правится вручную.** |
| `shiroa/book.typ` | Сгенерированный файл (см. ниже). **В git не хранится**, пересоздаётся при каждой сборке. |
| `shiroa/methods/*.typ`, `shiroa/experiments/*.typ`, `shiroa/README.typ` | Сгенерированные `.typ`-файлы из `docs/*.md`. В git не хранятся. |
| `shiroa/templates/page.typ` | HTML-шаблон страницы (шрифты, стили) |
| `shiroa/templates/extra.css` | Дополнительный CSS, инжектится в собранный HTML после `shiroa build` |
| `shiroa/dist/` | Собранная HTML-книга. В git не хранится. |

Всё, что генерируется (`book.typ`, `methods/`, `experiments/`, `README.typ`, `dist/`),
перечислено в `shiroa/.gitignore` — в репозитории лежат только исходники (`docs/*.md`,
`book.typ.tmpl`, скрипты, шаблоны).

## Настройка на GitHub (уже сделано, для справки)

1. **Workflow**: `.github/workflows/docs.yml`, триггер — `push` в `main` с
   изменениями в `docs/**`, `shiroa/book.typ.tmpl`, `shiroa/templates/**`,
   `shiroa/build-content.sh`, `shiroa/fix_for_shiroa.py`, `shiroa/gen_summary.py`
   (плюс `workflow_dispatch` для ручного запуска).
2. **Права**: `permissions: contents: write` — нужно, чтобы workflow мог
   запушить собранный сайт в ветку `gh-pages` (использует встроенный
   `secrets.GITHUB_TOKEN`, ничего дополнительно заводить не надо).
3. **Шаги workflow**:
   - checkout репозитория
   - `apt-get install pandoc`
   - скачать бинарник `shiroa` с GitHub Releases (`Myriad-Dreamin/shiroa`, сейчас `v0.3.0`)
   - `bash shiroa/build-content.sh --build` — весь pipeline выше
   - собрать `_site/`: `docs/` копируется в корень (Docsify как есть),
     `shiroa/dist/` — в `_site/typst/`
   - деплой в ветку `gh-pages` через `peaceiris/actions-gh-pages@v4`
     с `force_orphan: true` (ветка каждый раз перезаписывается, без истории —
     иначе бинарные ассеты книги раздувают репозиторий)
4. **GitHub Pages settings** (одноразово, руками в UI репозитория):
   Settings → Pages → Source → Deploy from branch → `gh-pages` / `/ (root)`.
   После первого успешного запуска workflow ветка `gh-pages` появится сама —
   тогда и включать источник.

## Как добавить новую модель в документацию

1. Создать `docs/methods/<name>.md` (см. правило в `CLAUDE.md` — по файлу на
   каждый файл модели в `knitwork/models/`).
2. Добавить строку в `docs/_sidebar.md` в нужную категорию:
   `- [Название](methods/<name>.md)`.
3. Закоммитить и запушить в `main` — Docsify обновится сразу (статика),
   Shiroa-книга пересоберётся автоматически через workflow (обычно 1–2 минуты).

Ничего в `shiroa/` руками менять не нужно.

## Локальная проверка перед пушем

```sh
# конвертация .md → .typ + генерация book.typ + сборка + инъекция CSS
bash shiroa/build-content.sh --build

# открыть локально
python3 -m http.server -d shiroa/dist 8000
```

Требуются в `PATH`: `pandoc`, `shiroa` (бинарник с GitHub Releases), `typst`
(ставится автоматически широа-пакетом при первой сборке, локальный `typst`
не обязателен, но полезен для отладки `.typ`-файлов напрямую).

Предупреждения `unknown font family: ...` в логе сборки — ожидаемы (шрифты,
прописанные в `templates/page.typ`, не установлены в CI/локально), на итоговый
HTML не влияют.
