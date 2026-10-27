"""Report completed and paused local communication runs without starting training."""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml
from markdown_it import MarkdownIt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.communication_report import digest, read_runs, metric, save_figure, axes_style, ORDER, LABELS, appearance, figures


def training_processes():
    modules = {
        'knitwork.exps.communication.queue', 'knitwork.exps.communication.run',
        'knitwork.exps.communication.run_single_column', 'knitwork.exps.communication.run_concat',
    }
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode().split('\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(arg in modules for arg in args):
            found.append({'pid': int(path.parent.name), 'args': args})
    return found


def paused_records(queue):
    records = []
    for job in queue['jobs']:
        if job['status'] != 'paused':
            continue
        directory = Path(job['config']['output_dir']) / job['run_id']
        cfg = yaml.safe_load((directory / 'config.yaml').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        metadata = json.loads((directory / 'metadata.json').read_text())
        snapshot = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=False)
        assert snapshot['step'] == summary['tokens']
        assert snapshot['config_fingerprint'] == metadata['config_fingerprint']
        assert snapshot['source_hashes'] == metadata['source_hashes']
        assert all(key in snapshot for key in ['optimizer', 'recurrent_state', 'generator', 'rng', 'lr', 'eval_schedule'])
        logs = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines() if line]
        evaluation = [row for row in logs if 'val/Loss' in row['metrics']]
        batch = cfg['n_envs'] * cfg['rollout_len']
        schedule = int(cfg['eval']['schedule'])
        steps = [((token + batch - 1) // batch) * batch for token in range(schedule, snapshot['last_eval_step'] + 1, schedule)]
        assert len(evaluation) == len(steps)
        assert steps[-1] == snapshot['last_eval_step']
        key = 'BPC' if cfg['exp'] == 'text8' else 'Acc'
        assert np.isclose(evaluation[-1]['metrics']['val/' + key], summary['final_validation'][key])
        records.append({
            'task': cfg['exp'], 'condition': cfg['condition'], 'config': cfg, 'summary': summary,
            'metadata': metadata, 'logs': logs, 'evaluation': evaluation, 'evaluation_steps': steps,
            'last_evaluation_step': snapshot['last_eval_step'],
            'artifacts': {name: {'path': str((directory / name).resolve()), 'sha256': digest(directory / name)}
                for name in ['config.yaml', 'summary.json', 'metadata.json', 'metrics.jsonl', 'latest.pt']},
        })
    return records


def label(record):
    if record['config']['topology'] == 'L2C1':
        return 'none / L2C1'
    return LABELS[record['condition']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='docs/experiments/communication_2026-10-13')
    args = parser.parse_args()
    root = Path('runs/communication_1m')
    queue_path = root / '_queue/status.json'
    queue = json.loads(queue_path.read_text())
    processes = training_processes()
    if queue['status'] != 'paused' or processes:
        raise RuntimeError(f'Training has not stopped: {processes}')
    complete, _, contrasts = read_runs(root)
    paused = paused_records(queue)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    now = datetime.now(ZoneInfo('Europe/Moscow'))
    by_task = {task: sorted([r for r in complete if r['task'] == task], key=lambda r: ORDER.index(r['condition'])) for task in ['text8', 'sdq']}
    lookup = {(r['task'], r['condition']): r for r in complete}
    matched = []
    for task, target in [('text8', 320000000), ('sdq', 460000000)]:
        rows = by_task[task] + [r for r in paused if r['task'] == task]
        for record in rows:
            batch = record['config']['n_envs'] * record['config']['rollout_len']
            step = ((target + batch - 1) // batch) * batch
            index = record['evaluation_steps'].index(step)
            key = 'val/BPC' if task == 'text8' else 'val/Acc'
            matched.append({'task': task, 'label': label(record), 'step': step, 'value': record['evaluation'][index]['metrics'][key]})
    gallery = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.axisbelow': True})
    with PdfPages(output / 'all_figures.pdf') as pdf:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for i, record in enumerate(by_task['text8']):
            s = record['summary']
            color = appearance(record)['color']
            values = [s['final_validation']['BPC'], s['final_test']['BPC']]
            ax.plot(values, [i, i], color=color, alpha=0.3)
            for value, marker in zip(values, ['o', 'D']):
                ax.scatter(value, i, color=color, marker=marker)
                ax.annotate(f'{value:.4f}', (value, i), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8)
        ax.set_yticks(range(6), [label(r) for r in by_task['text8']])
        ax.invert_yaxis()
        ax.set_xlim(1.58, 1.81)
        axes_style(ax, xlabel='BPC')
        for marker, text in [('o', 'Validation'), ('D', 'Test')]:
            ax.scatter([], [], marker=marker, color='#333333', label=text)
        ax.legend()
        save_figure(fig, output, '01_text8_final', 'Text8', 'Все шесть условий завершены при одинаковом бюджете около 1B токенов. Ниже BPC — лучше.', pdf, gallery)

        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for i, record in enumerate(by_task['sdq']):
            s = record['summary']
            for value, marker in [(s['final_validation']['Acc'] * 100, 'o'), (s['fixed_gap_evaluation']['macro_Acc'] * 100, 's')]:
                ax.scatter(value, i, marker=marker, color=appearance(record)['color'])
            ax.annotate(f"{s['final_validation']['Acc']*100:.2f}", (s['final_validation']['Acc']*100, i), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8)
        ax.set_yticks(range(5), [label(r) for r in by_task['sdq']])
        ax.invert_yaxis()
        axes_style(ax, xlabel='Acc (%)')
        ax.set_xlim(15, 95)
        for marker, text in [('o', 'Held-out'), ('s', 'Fixed-gap macro')]:
            ax.scatter([], [], marker=marker, color='#333333', label=text)
        ax.legend()
        save_figure(fig, output, '02_sdq_final', 'SDQ', 'Пять завершённых условий, одинаковый полный бюджет. Частичный none_mean в итоговый рейтинг не включён.', pdf, gallery)

        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for record in by_task['sdq']:
            values = [100 * record['summary']['fixed_gap_evaluation'][f'gap{gap}/Acc'] for gap in [8, 32, 128, 256, 512]]
            ax.plot(range(5), values, marker='o', **appearance(record))
        ax.set_xticks(range(5), ['8', '32', '128', '256', '512'])
        ax.set_ylim(0, 105)
        axes_style(ax, xlabel='Gap', ylabel='Acc (%)')
        ax.legend()
        save_figure(fig, output, '03_sdq_fixed_gaps', 'Память SDQ', '1024 примера на каждое расстояние. Выбранная запись не перезаписывается; это отдельное распределение, не обычные held-out queries.', pdf, gallery)

        fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for ax, task in zip(axs, ['text8', 'sdq']):
            for record in by_task[task] + [r for r in paused if r['task'] == task]:
                x, y = metric(record, 'val/BPC' if task == 'text8' else 'val/Acc', evaluation=True)
                if record['summary']['completed']:
                    style = appearance(record)
                else:
                    style = {'color': '#202020', 'linestyle': ':', 'label': label(record) + ' / paused'}
                ax.plot(x, y * (1 if task == 'text8' else 100), **style)
                if not record['summary']['completed']:
                    ax.scatter(x[-1], y[-1] * (1 if task == 'text8' else 100), color='#202020', marker='x')
            ax.set_title('Text8' if task == 'text8' else 'SDQ')
            axes_style(ax, xlabel='Tokens (M)', ylabel='BPC' if task == 'text8' else 'Acc (%)')
            if task == 'sdq':
                ax.set_ylim(0, 100)
            ax.legend()
        save_figure(fig, output, '04_validation_curves', 'Траектории validation', 'Пунктирные чёрные кривые — остановленные обучения. Оценки привязаны к фактическому расписанию evaluation, а не к моменту отложенной записи в лог.', pdf, gallery)

        fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for ax, task in zip(axs, ['text8', 'sdq']):
            rows = [r for r in matched if r['task'] == task]
            for i, row in enumerate(rows):
                value = row['value'] * (1 if task == 'text8' else 100)
                ax.scatter(value, i, color='#202020' if 'L2C1' in row['label'] or row['label'] == 'none / mean' and task == 'sdq' else '#355c7d')
                ax.annotate(f'{value:.4f}' if task == 'text8' else f'{value:.2f}', (value, i), xytext=(7, 0), textcoords='offset points', va='center', fontsize=8)
            ax.set_yticks(range(len(rows)), [r['label'] for r in rows])
            ax.invert_yaxis()
            ax.margins(x=0.3)
            ax.set_title('Text8 / 320M' if task == 'text8' else 'SDQ / 460M')
            axes_style(ax, xlabel='BPC' if task == 'text8' else 'Acc (%)')
        save_figure(fig, output, '05_matched_budget', 'Сопоставимый промежуточный бюджет', 'Text8 сравнивается при 320,012,288 токенах, SDQ при 460,013,568. L2C1 имеет другое число state и другую ширину; сравнение его архитектуры не является state-matched.', pdf, gallery)

    main_gallery = figures(complete, output / 'main_plots', limit=3, partial_records=paused)
    for item in main_gallery:
        item['name'] = 'main_plots/' + item['name']
    import pymupdf
    merged = pymupdf.open()
    for path in [output / 'main_plots/all_figures.pdf', output / 'all_figures.pdf']:
        with pymupdf.open(path) as document:
            merged.insert_pdf(document)
    merged.save(output / 'all_figures_with_main.pdf')
    assert len(merged) == 8
    merged.close()

    t = lambda condition, field='final_validation': lookup['text8', condition]['summary'][field]['BPC']
    a = lambda condition: lookup['sdq', condition]['summary']['final_validation']['Acc'] * 100
    counts = dict(Counter(j['status'] for j in queue['jobs']))
    lines = [
        '# Новые локальные результаты коммуникации MoSAIC', '',
        f'Срез: {now:%d.%m.%Y %H:%M} по Москве. Очередь остановлена по запросу пользователя: {counts.get("complete", 0)} завершённых запусков, {counts.get("paused", 0)} приостановленных, {counts.get("pending", 0)} ещё не начатых. Активных supervisor/worker процессов нет. Возобновление только вручную.', '',
        'Новые завершённые результаты относительно отчёта за 12 октября: Text8 `none_mean` и SDQ `static_mean`. В основном сравнении L2C4/H136, clean, seed модели и данных 42; около 1M параметров, 1088 чисел recurrent state. Все завершённые условия внутри задачи получили одинаковый бюджет около 1B токенов и совпадающий протокол. Source/config/data/initialization matching проверен штатным summarizer.', '',
        '**Все завершённые условия.** BPC ниже — лучше, accuracy выше — лучше. Выбраны последние checkpoints, а не лучшие точки траекторий.', '',
        '| Условие | Text8 val BPC | Text8 test BPC | SDQ held-out Acc | SDQ fixed-gap macro |',
        '|---|---:|---:|---:|---:|',
    ]
    for condition in ORDER:
        text, sdq = lookup.get(('text8', condition)), lookup.get(('sdq', condition))
        lines.append(f"| {condition} | {text['summary']['final_validation']['BPC']:.4f} | {text['summary']['final_test']['BPC']:.4f} | " + (f"{sdq['summary']['final_validation']['Acc']*100:.2f}% | {sdq['summary']['fixed_gap_evaluation']['macro_Acc']*100:.2f}% |" if sdq else 'не завершён | — |'))
    lines += ['', '**Что добавил Text8 none_mean.**', '',
        f"Без общения, но с чтением всех четырёх колонок, получено {t('none_mean'):.4f} val BPC и {t('none_mean', 'final_test'):.4f} test BPC. Это лучше none_column0 на {t('none_column0')-t('none_mean'):.4f} val BPC: доступность остальных колонок действительно имеет значение в этом запуске. При этом none_mean хуже dynamic_mean на {t('none_mean')-t('dynamic_mean'):.4f} и static_mean на {t('none_mean')-t('static_mean'):.4f} val BPC. Test сохраняет направление различий: соответственно +{t('none_mean', 'final_test')-t('dynamic_mean', 'final_test'):.4f} и +{t('none_mean', 'final_test')-t('static_mean', 'final_test'):.4f} BPC.", '',
        'Это более содержательный контроль коммуникации, чем none_column0: все 1088 state-чисел имеют путь к предсказанию, readout mean одинаков. Здесь преимущество communicating-моделей не объясняется только недоступностью трёх колонок в none_column0. Номинальное число параметров и dense matrix FLOPs совпадают; эффективные градиенты и общие вычисления не обязаны совпадать. Результат описательный, одного seed. Преимущество маршрутизации, зависящей от содержимого, над обучаемыми static-связями не установлено.', '',
        '**Что добавил SDQ static_mean.**', '',
        f"Финальная held-out accuracy {a('static_mean'):.2f}%: выше dynamic_mean на {a('static_mean')-a('dynamic_mean'):.2f} процентного пункта, но ниже static_column0 на {a('static_column0')-a('static_mean'):.2f} пункта. Открыть readout всю верхнюю память через mean не оказалось универсальным улучшением: у dynamic это помогло, у static в этом seed ухудшило итоговую общую accuracy.", '',
        'Средняя accuracy пяти fixed gaps у static_mean — 70.27%, у dynamic_mean — 70.88%, у static_column0 — 72.89%. На gap=512 static_mean даёт 16.50%, static_column0 4.98%, dynamic_mean 30.27%. Поэтому преимущество на обычной held-out задаче не означает лучшую память на каждом расстоянии. Mean может помогать сохранить доступ к длинным зависимостям, одновременно ухудшая короткие и средние; это интерпретация наблюдаемой картины, не установленный механизм.', '',
        '**Приостановленные обучения: сохранён прогресс, результаты промежуточные.**', '',
        '| Запуск | Checkpoint tokens | Последняя оценка при tokens | Метрика |', '|---|---:|---:|---:|',
    ]
    for record in paused:
        s, cfg = record['summary'], record['config']
        score = f"{s['final_validation']['BPC']:.4f} BPC" if record['task'] == 'text8' else f"{s['final_validation']['Acc']*100:.2f}% Acc"
        lines.append(f"| {record['task']} / {cfg['topology']} / {record['condition']} | {s['tokens']:,} | {record['last_evaluation_step']:,} | {score} |")
    lines += ['', 'Последняя validation у остановленного обучения относится к последнему выполненному evaluation, а не к весам в более позднем checkpoint остановки. Новая evaluation после остановки не выполнялась. Checkpoints прочитаны на CPU: проверены число токенов, fingerprint, source hashes и наличие состояния optimizer, recurrent memory, генератора, RNG и расписаний. Ничего не возобновлялось.', '',
        'Одноколоночный Text8 имеет H228, 1 057 493 номинальных параметра и 456 активных state-чисел; L2C4 имеет H136, 1088 state-чисел, а none_column0 L2C4 использует для ответа только 272. Поэтому этот контроль устраняет бесполезные колонки, но не уравнивает размер памяти с L2C4. Его текущее качество не является итогом полного бюджета.', '',
        '**Сравнение частичных запусков при одинаковом числе обучающих токенов.**', '',
        '| Задача | Условие | Tokens evaluation | Метрика |', '|---|---|---:|---:|',
    ]
    for row in matched:
        score = f"{row['value']:.4f} BPC" if row['task'] == 'text8' else f"{row['value']*100:.2f}% Acc"
        lines.append(f"| {row['task']} | {row['label']} | {row['step']:,} | {score} |")
    matched_values = {(row['task'], row['label']): row['value'] for row in matched}
    lines += ['',
        f"При 320M токенов широкая единственная колонка даёт {matched_values['text8', 'none / L2C1']:.4f} BPC, а none_mean L2C4 — {matched_values['text8', 'none / mean']:.4f}. Одноколоночный вариант близок к static_mean ({matched_values['text8', 'static / mean']:.4f}) и уступает dynamic_column0 ({matched_values['text8', 'dynamic / column0']:.4f}). Это показывает конкурентность широкой колонки на этой стадии, несмотря на меньший state. Число колонок само по себе не определяет качество; окончательного сравнения на полном бюджете ещё нет.", '',
        f"При 460M токенов SDQ none_mean даёт {matched_values['sdq', 'none / mean']*100:.2f}%, выше none_column0 ({matched_values['sdq', 'none / column0']*100:.2f}%), но ниже dynamic_mean ({matched_values['sdq', 'dynamic / mean']*100:.2f}%) и static_mean ({matched_values['sdq', 'static / mean']*100:.2f}%). Значит, промежуточное качество также нельзя объяснить только недоступностью трёх колонок в none_column0; отставание остаётся при чтении всей независимой памяти. Это наблюдение на промежуточном бюджете, не итоговый эффект.", '',
        'Эта таблица сравнивает промежуточное качество при одинаковом бюджете и помогает читать ранние траектории. Она не заменяет результаты после 1B токенов. В частности, SDQ none_mean ещё не закончил curriculum: в момент остановки training T около 306, а validation с самого начала использует T=500.', '',
        '**Что остаётся открытым для замечания ревью.**', '',
        'На Text8 появились описательные данные в пользу межколоночного обмена при одинаковом номинальном state и доступе readout ко всем колонкам. На SDQ главный контраст dynamic_mean против none_mean пока не завершён. Dynamic не показал общего преимущества над обучаемым static. Один seed не даёт оценки устойчивости между обучениями; тестовая Text8-оценка — дополнительный сегмент того же обучения, не независимый seed. Конкатенационный none-readout и одноколоночный полный бюджет остаются незавершёнными.', '',
        'Ещё не запускались: SDQ L2C1 none_column0, Text8 L2C4 none_concat и SDQ L2C4 none_concat. Они остаются в приостановленной очереди и автоматически не начнутся.', '',
        '**Три основных графика, обновлённые по новым данным.** На кривых обучения показаны 11 завершённых и два остановленных запуска. Финальное качество и память по gap сравнивают только завершённые обучения с полным бюджетом. Подписи осей краткие, без общего заголовка и текстовых сносок внутри изображений.', '',
    ]
    for figure in main_gallery:
        lines += [f"![{figure['title']}]({figure['name']}.png)", '', figure['note'], '']
    lines += ['**Дополнительные сравнения.**', '']
    for figure in gallery:
        lines += [f"![{figure['title']}]({figure['name']}.png)", '', figure['note'], '']
    lines += ['**CKA и чтение памяти.** Диагностики прежних девяти checkpoints сохранены в [отчёте с 13 графиками](../communication_diagnostics_2026-10-12_selected/report.md). Они не пересчитаны для двух новых завершённых моделей, поэтому их выводы не распространяются автоматически на новые checkpoints.', '',
        '**Артефакты и воспроизведение.** [Все восемь графиков PDF](all_figures_with_main.pdf), [три основных графика PDF](main_plots/all_figures.pdf), [исходные данные и SHA256](data.json), [проверка остановки и целостности](validation.json). Основные протоколы: [конфиги и запуск](../../../knitwork/exps/communication/config/README.md).', '',
        '```bash', '.venv/bin/python inference/communication_update_report.py', '```', '',
        'Команда только читает локальные артефакты и строит отчёт. Она требует остановленной очереди; обучение и внешние tracker-запросы не запускает.',
    ]
    report = '\n'.join(lines) + '\n'
    (output / 'report.md').write_text(report)
    html = '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Новые результаты коммуникации</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#222}img{display:block;max-width:100%;height:auto;margin:28px auto}table{display:block;overflow-x:auto;border-collapse:collapse;margin:24px 0}td,th{border-bottom:1px solid #ddd;padding:8px 12px;text-align:left}pre{overflow:auto;background:#f5f5f5;padding:16px}a{color:#355c7d}</style>' + MarkdownIt('commonmark').enable('table').render(report) + '</html>'
    for name in ['report.html', 'index.html']:
        (output / name).write_text(html)
    gallery = main_gallery + gallery
    export = {'created_at': now.isoformat(), 'report_source_sha256': digest(Path(__file__)), 'main_plot_source_sha256': digest(Path('inference/communication_report.py')), 'queue_counts': counts, 'queue_status_sha256': digest(queue_path), 'completed': complete, 'paused': paused, 'matched_budget': matched, 'contrasts': contrasts, 'figures': gallery}
    (output / 'data.json').write_text(json.dumps(export, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    processes = training_processes()
    assert not processes
    collisions = {g['name']: g['text_overlaps'] for g in gallery if g['text_overlaps']}
    (output / 'validation.json').write_text(json.dumps({'queue_status': queue['status'], 'live_training_processes': processes, 'completed_runs': len(complete), 'paused_runs': len(paused), 'resume_checkpoints_verified': True, 'matched_evaluation_budgets_verified': True, 'figures': len(gallery), 'text_overlaps': collisions, 'report_sha256': digest(output / 'report.md')}, ensure_ascii=False, indent=2) + '\n')
    print(f'{len(complete)} complete, {len(paused)} paused; {len(gallery)} figures; overlaps={collisions}; {output / "report.html"}')


if __name__ == '__main__':
    main()
