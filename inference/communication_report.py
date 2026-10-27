"""Frozen, reproducible report of completed local communication ablations."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.report_plot_style import prepare_figure, text_overlaps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knitwork.exps.communication.summarize import summarize

ORDER = ['dynamic_column0', 'static_column0', 'none_column0', 'dynamic_mean', 'static_mean', 'none_mean']
COLORS = {'dynamic': '#355c7d', 'static': '#c47a42', 'none': '#69767d'}
STYLE = {'column0': '-', 'mean': '--'}
LABELS = {name: name.replace('_', ' / ') for name in ORDER}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_runs(root):
    verified = summarize(root)
    records = []
    incomplete = []
    for path in sorted(root.glob('*/L2C4/clean/*/seed42-data42/summary.json')):
        summary = json.loads(path.read_text())
        cfg = yaml.safe_load((path.parent / 'config.yaml').read_text())
        if not summary['completed']:
            incomplete.append({'run_id': summary['run_id'], 'tokens': summary['tokens'], 'completed': False})
            continue
        if summary['tokens'] != cfg['effective_tokens']:
            raise ValueError(f'Incomplete budget: {path}')
        logs = [json.loads(line) for line in (path.parent / 'metrics.jsonl').read_text().splitlines() if line]
        if any(logs[i]['step'] <= logs[i-1]['step'] for i in range(1, len(logs))):
            raise ValueError(f'Duplicate or reversed logging steps: {path}')
        evaluation = [row for row in logs if 'val/Loss' in row['metrics']]
        batch = cfg['n_envs'] * cfg['rollout_len']
        schedule = int(cfg['eval']['schedule'])
        expected_steps = [((token + batch - 1) // batch) * batch
            for token in range(schedule, cfg['effective_tokens'], schedule)] + [cfg['effective_tokens']]
        if len(evaluation) != len(expected_steps):
            raise ValueError(f'Cannot align evaluations: {path}')
        for row, evaluated_step in zip(evaluation, expected_steps):
            if not evaluated_step <= row['step'] < evaluated_step + int(cfg['log']['schedule']) + batch:
                raise ValueError(f'Evaluation logging delay inconsistent with schedule: {path}')
        if not np.isclose(evaluation[-1]['metrics']['val/Loss'], summary['final_validation']['Loss']):
            raise ValueError(f'Final evaluation differs from summary: {path}')
        metadata = json.loads((path.parent / 'metadata.json').read_text())
        files = ['config.yaml', 'metadata.json', 'summary.json', 'metrics.jsonl']
        files += [p.name for p in path.parent.glob('*final*.npz')]
        artifacts = {}
        for filename in files:
            source = path.parent / filename
            artifacts[filename] = {'path': str(source.resolve()), 'sha256': digest(source)}
        records.append({
            'task': cfg['exp'], 'condition': cfg['condition'], 'config': cfg, 'summary': summary,
            'metadata': metadata, 'logs': logs, 'evaluation_steps': expected_steps,
            'evaluation': evaluation, 'artifacts': artifacts,
        })
    return records, incomplete, verified


def appearance(record):
    router, readout = record['condition'].split('_')
    return {'color': COLORS[router], 'linestyle': STYLE[readout], 'label': LABELS[record['condition']]}


def metric(record, name, *, evaluation=False):
    rows = record['evaluation'] if evaluation else record['logs']
    selected = [(row, index) for index, row in enumerate(rows) if name in row['metrics']]
    steps = [record['evaluation_steps'][index] if evaluation else row['step'] for row, index in selected]
    return np.asarray(steps) / 1e6, np.asarray([row['metrics'][name] for row, _ in selected])


def save_figure(fig, output, name, title, note, pdf, gallery):
    prepare_figure(fig)
    collisions = text_overlaps(fig)
    if collisions:
        print('LAYOUT', name, collisions)
    fig.savefig(output / f'{name}.png', dpi=170, bbox_inches='tight', facecolor='white')
    fig.savefig(output / f'{name}.svg', bbox_inches='tight', facecolor='white')
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    pdf.savefig(fig, bbox_inches='tight')
    gallery.append({'name': name, 'title': title, 'note': note, 'text_overlaps': collisions})
    plt.close(fig)


def axes_style(ax, *, xlabel=None, ylabel=None):
    ax.grid(True, alpha=0.18)
    ax.spines[['top', 'right']].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def figures(records, output, *, limit=None, partial_records=()):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titleweight': 'bold', 'axes.axisbelow': True})
    by_task = {task: sorted([r for r in records if r['task'] == task], key=lambda r: ORDER.index(r['condition'])) for task in ('text8', 'sdq')}
    gallery = []
    note = 'L2C4, около 1M параметров, clean, seed 42. Только завершённые запуски; полос неопределённости нет.'
    with PdfPages(output / 'all_figures.pdf') as pdf:
        fig, axs = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
        for col, task in enumerate(('text8', 'sdq')):
            key = 'BPC' if task == 'text8' else 'Acc'
            scale = 1 if task == 'text8' else 100
            for record in by_task[task] + [r for r in partial_records if r['task'] == task]:
                if record['summary']['completed']:
                    style = appearance(record)
                else:
                    name = 'none / L2C1' if record['config']['topology'] == 'L2C1' else LABELS[record['condition']]
                    style = {'color': '#202020', 'linestyle': ':', 'label': name + ' / paused'}
                x, y = metric(record, key)
                axs[0, col].plot(x, y * scale, **style)
                x, y = metric(record, 'val/' + key, evaluation=True)
                axs[1, col].plot(x, y * scale, **style)
                if not record['summary']['completed']:
                    axs[1, col].scatter(x[-1], y[-1] * scale, marker='x', color='#202020')
            axs[0, col].set_title(f'{task.upper()}: обучение (EMA)')
            axs[1, col].set_title(f'{task.upper()}: независимая validation')
            for row in range(2):
                axes_style(axs[row, col], xlabel='Tokens (M)', ylabel='BPC' if task == 'text8' else 'Acc (%)')
                if task == 'sdq':
                    axs[row, col].set_ylim(0, 100)
            axs[0, col].legend(fontsize=8)
        save_figure(fig, output, '01_learning', 'Как меняется качество по мере обучения',
            ('L2C4, около 1M параметров, clean, seed 42. Чёрным пунктиром отмечены остановленные SDQ none_mean и Text8 L2C1; их последние validation предшествуют checkpoint остановки. ' if partial_records else note + ' ')
            + 'SDQ train меняет сложность по curriculum; validation всегда использует T=500. X validation восстановлен по расписанию.', pdf, gallery)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
        for col, task in enumerate(('text8', 'sdq')):
            lookup = {r['condition']: r for r in by_task[task]}
            for i, condition in enumerate(ORDER):
                r = lookup.get(condition)
                if r is None:
                    axs[col].axhspan(i-0.35, i+0.35, color='#f1f3f5')
                    axs[col].text(1.565 if task == 'text8' else 3, i, 'Not completed', va='center', fontsize=8, color='#777777')
                    continue
                color = appearance(r)['color']
                s = r['summary']
                if task == 'text8':
                    values = [s['final_validation']['BPC'], s['final_test']['BPC']]
                    axs[col].plot(values, [i, i], color=color, alpha=0.35)
                    for v, marker in zip(values, ('o', 'D')):
                        axs[col].scatter(v, i, marker=marker, color=color, s=60)
                        axs[col].annotate(f'{v:.4f}', (v, i), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8)
                else:
                    values = [s['final_validation']['Acc'], s['final_validation']['Acc_nonmiss'], s['fixed_gap_evaluation']['macro_Acc']]
                    for v, marker in zip(values, ('o', '^', 's')):
                        axs[col].scatter(100*v, i, marker=marker, color=color, s=60)
                    axs[col].annotate(f"{values[0]*100:.2f}%", (100*values[0], i), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)
            axs[col].set_yticks(range(len(ORDER)), [LABELS[c] for c in ORDER])
            axs[col].invert_yaxis()
            axes_style(axs[col], xlabel='BPC' if task == 'text8' else 'Acc (%)')
            axs[col].set_title('Text8: финальный checkpoint' if task == 'text8' else 'SDQ: финальный checkpoint')
            marker_labels = [('o', 'validation'), ('D', 'test')] if task == 'text8' else [('o', 'все held-out queries'), ('^', 'queries с предшествующим store'), ('s', 'macro фиксированных gaps')]
            for marker, label in marker_labels:
                axs[col].scatter([], [], marker=marker, color='#222222', label=label)
            axs[col].legend(fontsize=8, loc='lower right')
            axs[col].set_xlim((1.55, 1.86) if task == 'text8' else (0, 105))
        save_figure(fig, output, '02_final_quality', 'Финальное качество: одинаковый бюджет около 1B токенов',
            note + ' Отсутствующие условия показаны серым; это не нулевой результат. Точки с разными маркерами измеряют разные критерии.', pdf, gallery)

        fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for r in by_task['sdq']:
            s = r['summary']
            val = s['final_validation']
            native = [100*val.get(f'gap_bin{i}/Acc', np.nan) for i in range(1, 7)]
            fixed = [100*s['fixed_gap_evaluation'][f'gap{gap}/Acc'] for gap in r['config']['eval']['fixed_gaps']]
            axs[0].plot(range(6), native, marker='o', **appearance(r))
            axs[1].plot(range(5), fixed, marker='o', **appearance(r))
        axs[0].set_xticks(range(6), ['[0,8)', '[8,32)', '[32,128)', '[128,256)', '[256,512)', '≥512\nN/A'])
        axs[1].set_xticks(range(5), ['8', '32', '128', '256', '512'])
        axs[0].set_title('Обычный SDQ: расстояние от сохранения до запроса')
        axs[1].set_title('Отдельная проверка: нужная запись не перезаписывается')
        for ax in axs:
            ax.set_ylim(0, 105)
            ax.axhline(10, color='#999999', linestyle=':', label='Chance: 10%')
            axes_style(ax, xlabel='Gap', ylabel='Acc (%)')
        axs[1].legend(fontsize=8)
        save_figure(fig, output, '03_memory_gaps', 'Как далеко модель сохраняет полезную память',
            note + ' Слева реальные queries с известным store; counts: 17130 / 41872 / 51513 / 6286 / 393 / 0. Справа 1024 примера на gap.', pdf, gallery)
        if limit == 3:
            return gallery

        fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for r in by_task['sdq']:
            x, y = metric(r, 'val/Loss', evaluation=True)
            axs[0].plot(x, y, **appearance(r))
            x, y = metric(r, '|Grad|')
            axs[1].plot(x, y, **appearance(r))
        axs[0].set_title('Ошибка на неизменной held-out задаче')
        axs[1].set_title('Норма градиента до clipping (EMA)')
        axs[1].set_yscale('log')
        axes_style(axs[0], xlabel='Tokens (M)', ylabel='CE')
        axes_style(axs[1], xlabel='Tokens (M)', ylabel='Grad norm')
        axs[0].legend(fontsize=8)
        save_figure(fig, output, '04_sdq_stability', 'SDQ: финальное качество и устойчивость траектории',
            note + ' Градиенты показаны до ограничения нормы до 1; большие нормы не означают пропуск optimizer update.', pdf, gallery)

        fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for col, task in enumerate(('text8', 'sdq')):
            key = 'val/BPC' if task == 'text8' else 'val/Acc'
            for r in by_task[task]:
                pairs = [(row['metrics']['resource/training_seconds']/3600, row['metrics'][key])
                    for row in r['evaluation'] if 'resource/training_seconds' in row['metrics']]
                x, y = np.asarray(pairs).T
                axs[col].plot(x, y*(1 if task == 'text8' else 100), **appearance(r))
            axes_style(axs[col], xlabel='Train time (h)', ylabel='BPC' if task == 'text8' else 'Acc (%)')
            axs[col].set_title(task.upper())
            if task == 'sdq':
                axs[col].set_ylim(0, 100)
            axs[col].legend(fontsize=8)
        save_figure(fig, output, '05_quality_time', 'Качество относительно измеренного времени обучения',
            note + ' Время без ночных пауз и evaluations, с Python/compile overhead. Два совместных GPU-процесса: не isolated latency benchmark.', pdf, gallery)

        fig, axs = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
        for col, task in enumerate(('text8', 'sdq')):
            rs = by_task[task]
            labels = [LABELS[r['condition']] for r in rs]
            hours = [r['summary']['training_seconds']/3600 for r in rs]
            peaks = [r['summary']['cuda_peak_allocated_bytes']/2**30 for r in rs]
            for row, values, title in ((0, hours, 'Training loop, часов'), (1, peaks, 'Reported CUDA peak, GiB')):
                axs[row, col].barh(range(len(rs)), values, color=[appearance(r)['color'] for r in rs], alpha=0.85)
                axs[row, col].set_yticks(range(len(rs)), labels)
                axs[row, col].invert_yaxis()
                for i, value in enumerate(values):
                    axs[row, col].text(value+max(values)*0.015, i, f'{value:.2f}', va='center', fontsize=9)
                axs[row, col].set_xlim(0, max(values)*1.18)
                axs[row, col].set_title(task.upper()+' · '+title)
                axes_style(axs[row, col])
        save_figure(fig, output, '06_resources', 'Фактически записанные время и память',
            note + ' CUDA peak относится к последнему процессу/сеансу: счётчик сбрасывается при resume. Ресурсы между разными задачами не уравнены.', pdf, gallery)

        fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        comparisons = [('dynamic_column0', 'static_column0'), ('dynamic_mean', 'static_mean'), ('dynamic_mean', 'dynamic_column0'), ('static_mean', 'static_column0')]
        descriptions = ['Dynamic − Static / column0', 'Dynamic − Static / mean', 'Mean − Column0 / dynamic', 'Mean − Column0 / static']
        for col, task in enumerate(('text8', 'sdq')):
            lookup = {r['condition']: r for r in by_task[task]}
            for i, (a, b) in enumerate(comparisons):
                if a not in lookup or b not in lookup:
                    axs[col].text(0.01 if task == 'text8' else 1, i, 'Not completed', va='center', color='#888888', fontsize=8)
                    continue
                va, vb = lookup[a]['summary']['final_validation'], lookup[b]['summary']['final_validation']
                delta = vb['BPC']-va['BPC'] if task == 'text8' else 100*(va['Acc']-vb['Acc'])
                axs[col].barh(i, delta, color='#23958b' if delta >= 0 else '#b95755', height=0.55)
                axs[col].text(delta, i, f' {delta:+.4f}' if task == 'text8' else f' {delta:+.2f} pp', va='center', ha='left' if delta >= 0 else 'right')
            axs[col].axvline(0, color='#333333', linewidth=1)
            axs[col].set_yticks(range(4), descriptions)
            axs[col].set_ylim(3.6, -0.6)
            axs[col].set_xlim((-0.04, 0.04) if task == 'text8' else (-55, 55))
            axs[col].set_title(task.upper()+' · положительное = первый вариант лучше')
            axes_style(axs[col], xlabel='BPC gain' if task == 'text8' else 'Acc gain (pp)')
        save_figure(fig, output, '07_contrasts', 'Доступные сравнения: маршруты и способ чтения памяти',
            note + ' Строки readout — отдельная диагностика. Главные контрасты с none_mean пока отсутствуют; сравнение с none_column0 их не заменяет.', pdf, gallery)
    return gallery


def report(records, incomplete, verified, output, gallery):
    lookup = {(r['task'], r['condition']): r for r in records}
    endpoints = []
    for task, condition in sorted(lookup):
        r = lookup[task, condition]
        s, v = r['summary'], r['summary']['final_validation']
        endpoints.append({
            'task': task, 'condition': condition, 'tokens': s['tokens'], 'updates': s['updates'],
            'validation_bpc': v.get('BPC'), 'test_bpc': (s['final_test'] or {}).get('BPC'),
            'heldout_acc': v.get('Acc') if task == 'sdq' else None,
            'heldout_nonmiss_acc': v.get('Acc_nonmiss'),
            'fixed_gap_macro_acc': (s['fixed_gap_evaluation'] or {}).get('macro_Acc'),
            'training_hours': s['training_seconds']/3600,
            'reported_last_session_cuda_peak_gib': s['cuda_peak_allocated_bytes']/2**30,
        })
    with (output/'endpoints.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoints[0]))
        writer.writeheader()
        writer.writerows(endpoints)
    export = {
        'created_at': datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
        'completed_runs': len(records), 'incomplete': incomplete, 'records': records,
        'prespecified_contrasts': verified, 'figures': gallery,
        'curve_method': '49 regular evaluation steps reconstructed from configured schedule and full optimizer batch, then final budget. Every association checked against logging delay and final summary. Training EMA uses logged token step. No interpolation or additional smoothing.',
    }
    (output/'data.json').write_text(json.dumps(export, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    lines = ['# Коммуникация MoSAIC: простая постановка и текущие результаты', '',
        'Срез на 12 октября 2026: **9 завершённых запусков из 12**. Все обучения остановлены по просьбе пользователя. В этот отчёт и графики входят только завершённые запуски; незавершённые результаты не смешиваются с итоговыми.', '',
        '## Что проверяем', '',
        'У модели есть четыре колонки памяти, каждая состоит из двух последовательно работающих GRU-ячеек. На каждом шаге она получает новый символ или событие, обновляет память и делает предсказание. Мы меняем только правила обмена между колонками и способ, которым предсказание читает их память. Вопрос: помогает ли само общение колонок, нужно ли выбирать связи по текущему содержимому памяти и достаточно ли просто открыть предсказанию всю память?', '',
        'На **Text8** нужно предсказать следующий символ текста. Ошибка измеряется в BPC: меньше лучше. Текст разделён на 90M символов обучения, 5M validation и 5M test. Validation и test проходят непрерывно, без случайных сбросов памяти; test читается после обучения.', '',
        'В **SDQ** модель запоминает пары «ключ — значение», встречает отвлекающие события и отвечает на запросы. Правильный ответ учитывает также счётчики store/query и накопленные отвлекающие значения, поэтому это задача вычисления с памятью, а не простого копирования. Основная оценка — доля правильных ответов на отдельном неизменном генераторе. Дополнительная проверка задаёт расстояния 8, 32, 128, 256 и 512 шагов; нужная запись там не перезаписывается.', '',
        '## Что одинаково', '',
        'Топология L2C4, ширина каждой ячейки 136, четыре attention-heads. Около 1M весов: Text8 **1 050 499**, SDQ **1 053 338**, включая вход и prediction head. На один поток выделено **1088 чисел recurrent state**. Все условия внутри задачи начинают с одинаковых весов; seed модели и данных равен **42**. Нет routing noise, штрафа коммуникации и entropy bonus: это рецепт `clean`.', '',
        'Каждому завершённому запуску дано около **1B training tokens**. Batch — 512 потоков; градиент проходит через 64 шага для Text8 и 32 для SDQ. Поэтому фактические бюджеты составляют 999 981 056 / 999 997 440 токенов и 30 517 / 61 035 updates. Optimizer RMSprop, clipping нормы до 1, одинаковое расписание LR внутри задачи. SDQ постепенно усложняется до T=500; оценка с самого начала использует фиксированный T=500. Проверка качества каждые 20M токенов, локальные метрики примерно каждые 5M.', '',
        '## Как реализованы dynamic, static и none', '',
        '**Dynamic — общение зависит от того, что сейчас хранится.** К текущим сообщениям добавляются обучаемые идентификаторы колонок. Из этих сумм линейные Q/K-проекции и SiLU строят запросы и ключи; V-проекция переносит само содержимое сообщения. Q и K определяют attention-веса. Чем изменилось содержимое памяти, тем могут измениться связи. Это исходный production `StochasticMessagePassingLayer`.', '',
        '**Static — связи обучаются, но не зависят от текущего содержимого.** Q и K строятся только из обучаемых идентификаторов колонок. V по-прежнему содержит текущую память или вход. Во время обучения веса связей могут меняться вместе с параметрами. При фиксированных параметрах один и тот же маршрут используется для любого текста или события. Это не замороженная случайная матрица и не простое равное усреднение. Полные проекции пересчитываются на каждом шаге, без кэширования.', '',
        '**None — колонки не читают состояние друг друга.** Сначала вычисляются те же dense-проекции. Затем перед softmax маска закрывает межколоночные связи. В нижнем слое каждой колонке доступны внешний ввод и собственное задержанное состояние верхнего слоя; выше — только сообщение собственной колонки снизу. Локальное смешивание ввода и собственной памяти остаётся обучаемым и зависит от содержимого. Общие веса проекций не означают обмен состояниями. Отключается именно межколоночный обмен, а не весь attention.', '',
        'Во всех трёх режимах после смешивания работают одинаковые выходная проекция, LayerNorm и GRU-ячейки. Везде четыре heads. В текущих `clean` запусках шум и вспомогательные штрафы равны нулю.', '',
        'Если нужны точные формулы, для каждой attention-head в текущем clean-режиме:', '',
        '```text',
        'dynamic: q = SiLU(Wq · (h_prev + id_q) + bq)',
        '         k = SiLU(Wk · (message + id_k) + bk)',
        'static:  q = SiLU(Wq · id_q + bq)',
        '         k = SiLU(Wk · id_k + bk)',
        'both:    v = SiLU(Wv · message + bv)',
        '         score = softplus(temp) · (q @ k.T) / sqrt(head_dim)',
        'none:    запрещённым межколоночным score присваивается −∞',
        '         остальные score строятся как в dynamic',
        'all:     weights = softmax(score)',
        '         msg = LayerNorm(Wo · concat_heads(weights @ v) + bo)',
        '         h_next = GRU(msg, h_prev)',
        '```', '',
        'В нижнем слое `message` содержит предыдущие верхние состояния всех колонок и текущий внешний ввод. В верхнем слое — свежие состояния нижнего слоя. При none маска оставляет только свою колонку; внешний ввод доступен каждой колонке нижнего слоя. Здесь head_dim = 136 / 4 = 34.', '',
        'Код: [production Grid и dynamic router](../../../knitwork/models/grnn_core.py), [static/none и readout](../../../knitwork/models/grnn_communication.py). Это matched-контроль с четырьмя heads, а не отдельный старый класс `StaticMessagePassingLayer`, у которого другая конструкция.', '',
        '**Чтение column0:** prediction head получает только верхнюю память колонки 0. **Чтение mean:** получает среднее верхних состояний всех четырёх колонок. Mean не добавляет весов и не возвращает смесь назад в recurrent state.', '',
        '## Каждая ablation', '',
        '| Условие | Что меняем | Какой вопрос задаём | Завершено |',
        '|---|---|---|---|',
        '| `dynamic_column0` | Исходные dynamic-связи; читаем колонку 0 | Опорный communicating grid | Text8, SDQ |',
        '| `static_column0` | Q/K только из идентификаторов; читаем колонку 0 | Нужно ли менять связи по содержимому? | Text8, SDQ |',
        '| `none_column0` | Запрещаем межколоночные связи; читаем колонку 0 | Что происходит при потере доступа к остальным колонкам? | Text8, SDQ |',
        '| `dynamic_mean` | Dynamic-связи; читаем среднее всех колонок | Меняет ли результат доступ head ко всей верхней памяти? | Text8, SDQ |',
        '| `static_mean` | Static-связи; читаем среднее всех колонок | Нужны ли dynamic-связи, когда вся память доступна head? | Только Text8 |',
        '| `none_mean` | Нет межколоночного обмена; читаем все колонки через mean | Достаточно ли независимой памяти без общения? | Ни одной задачи |', '',
        '**Особенность none_column0:** остальные три колонки не имеют пути к предсказанию. Выделено 1088 state-чисел, но к ответу могут повлиять только 272. Его проигрыш может объясняться потерей доступной памяти. Главный контроль, который снимает эту проблему, — `none_mean`; он ещё не завершён.', '',
        'Полные dense matrix products совпадают: примерно 3,06M forward FLOPs/token внутри каждой задачи; matching проверялся также для dense forward/backward. Softmax, elementwise операции, optimizer и эффективное число параметров с ненулевым task gradient этим не уравнены. Мы не делаем из номинального равенства весов вывод о равной полезной ёмкости.', '',
        '## Итоговые числа', '',
        '| Условие | Text8 val BPC ↓ | Text8 test BPC ↓ | SDQ held-out accuracy ↑ | SDQ fixed-gap macro ↑ |',
        '|---|---:|---:|---:|---:|']
    for condition in ORDER:
        text = lookup.get(('text8', condition))
        sdq = lookup.get(('sdq', condition))
        tval = f"{text['summary']['final_validation']['BPC']:.4f}" if text else '—'
        ttest = f"{text['summary']['final_test']['BPC']:.4f}" if text else '—'
        acc = f"{sdq['summary']['final_validation']['Acc']*100:.2f}%" if sdq else '—'
        macro = f"{sdq['summary']['fixed_gap_evaluation']['macro_Acc']*100:.2f}%" if sdq else '—'
        lines.append(f'| `{condition}` | {tval} | {ttest} | {acc} | {macro} |')
    lines += ['', '«—» означает отсутствие завершённого результата. Это не ноль. Итоговые числа относятся к последнему checkpoint после полного бюджета, а не к лучшей точке кривой.', '',
        '## Что уже видно', '',
        'На Text8 `dynamic_column0` и `static_column0` практически совпали: 1,6006 и 1,6004 BPC. В варианте mean static также немного ниже: 1,6144 против 1,6204. В этом seed преимущества изменения маршрутов по содержимому не наблюдаем.', '',
        'На SDQ `static_column0` дал 85,54% held-out accuracy, `dynamic_column0` — 45,71%: static выше на **39,83 процентного пункта**. Чтение всей памяти у dynamic подняло результат до 79,79%, то есть на **34,08 пункта** относительно column0. Это описательные разницы конкретных запусков, а не проверка статистической значимости.', '',
        'У SDQ dynamic/column0 качество заметно меняется ближе к концу: лучшая записанная held-out accuracy — 66,01% около 840M, финальная — 45,71%. Поэтому итог по последнему checkpoint и лучший результат за всё обучение — разные критерии. Основной итог здесь финальный; график не скрывает ухудшение.', '',
        'None/column0 хуже communicating-вариантов: Text8 1,7235 BPC, SDQ 23,32%. Из-за недоступности остальных колонок это ещё не доказательство необходимости общения. Чтобы отделить общение от простой доступности памяти, нужны завершённые `none_mean`; для SDQ отсутствует также `static_mean`.', '',
        'Проверки дальних gaps дают другую картину, чем общая accuracy. Например, на gap=512 dynamic/mean даёт 30,27%, static/column0 — 4,98%, dynamic/column0 — 3,32%. Static/column0 лучше по коротким и средним gaps и по общей accuracy, но это не универсальное преимущество на всех расстояниях.', '',
        '## Графики и как их читать', '']
    for item in gallery:
        lines += [f"### {item['title']}", '', f"![{item['title']}]({item['name']}.png)", '', item['note'], '']
    lines += ['## Ограничения простыми словами', '',
        'Это один seed и одна форма L2C4. У графиков нет p-values и полос «надёжности» между обучениями: отдельные символы и queries не заменяют повторные обучения. Результаты относятся к `clean` и текущей новой оценке, поэтому их нельзя напрямую подставлять вместо исторических таблиц статьи.', '',
        'Для SDQ online training accuracy сглажена EMA и измеряется на меняющейся по сложности задаче; held-out accuracy измеряется на фиксированной отдельной задаче. Их расхождение не равно обычному train/test gap. Fixed-gap probes — ещё одно распределение: в них нет перезаписи выбранного ключа.', '',
        'Время измерялось при двух параллельных процессах на RTX 5060 Laptop. Оно исключает evaluations и ночные паузы, но включает Python/compile overhead; это наблюдение о наших сеансах, а не чистый benchmark архитектур. CUDA peak из summary относится только к последнему процессу/сеансу. Счётчик после resume начинается заново.', '',
        'У validation-кривых восстановлены точные моменты оценки: расписание 20M, округление до полного optimizer update, затем финальный budget. Проверено соответствие каждой точки её записи в логе и совпадение последней оценки с summary. Дополнительное сглаживание и интерполяция не применялись.', '',
        '## Данные и воспроизведение', '',
        'Источники: локальные `runs/communication_1m/<task>/L2C4/clean/<condition>/seed42-data42/{config.yaml,metadata.json,metrics.jsonl,summary.json}` и финальные npz. Для каждой копии в [data.json](data.json) записаны SHA256; там же сырые числовые кривые и восстановленные evaluation steps. [Таблица CSV](endpoints.csv), [все графики PDF](all_figures.pdf), [галерея](index.html).', '',
        '```bash', '.venv/bin/python inference/communication_report.py', '```', '',
        'Скрипт читает только локальные файлы. Он не запускает обучение, не возобновляет очередь и не меняет исходные логи.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    html = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Коммуникация MoSAIC</title>', '<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;background:#f7f9fc;color:#233044}img{width:100%;background:white}section{margin:36px 0}a{color:#2563a6}</style>', '<h1>Коммуникация MoSAIC · 9 завершённых запусков · seed 42</h1><p><a href="report.md">Полный отчёт</a> · <a href="all_figures.pdf">PDF графиков</a> · <a href="endpoints.csv">CSV</a></p>']
    for item in gallery:
        html += [f"<section><h2>{item['title']}</h2><img src='{item['name']}.png'><p>{item['note']}</p><a href='{item['name']}.svg'>SVG</a> · <a href='{item['name']}.pdf'>PDF</a></section>"]
    html.append('</html>')
    (output/'index.html').write_text('\n'.join(html))


def integrate_diagnostics(output, diagnostics):
    """Embed saved checkpoint diagnostics without refreshing the frozen run snapshot."""
    from markdown_it import MarkdownIt
    import pymupdf

    output, diagnostics = Path(output), Path(diagnostics)
    source = diagnostics / 'report.md'
    diagnostic_text = source.read_text()
    prefix = Path(os.path.relpath(diagnostics, output)).as_posix()

    def rebase_link(match):
        target = match.group(2)
        if target.startswith(('/', '#')) or '://' in target:
            return match.group(0)
        return f'{match.group(1)}({prefix}/{target})'

    body = diagnostic_text.split('\n', 1)[1].strip()
    body = re.sub(r'^(#{2,6}) ', r'#\1 ', body, flags=re.MULTILINE)
    body = re.sub(r'(!?\[[^\]\n]*\])\(([^)\n]+)\)', rebase_link, body)
    images = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', body)
    if len(images) != 13:
        raise ValueError(f'Expected 13 saved diagnostic figures, found {len(images)}')
    for image in images:
        if not (output / image).is_file():
            raise FileNotFoundError(output / image)

    begin, end = '<!-- checkpoint-diagnostics:start -->', '<!-- checkpoint-diagnostics:end -->'
    block = '\n'.join([
        begin, '## Диагностика памяти и участия колонок', '',
        'В основной отчёт встроены 13 дополнительных графиков по тем же девяти завершённым локальным checkpoints. Это анализ замороженных моделей; основной срез качества обучения остаётся за 12 октября 2026. Новые головы и повреждения памяти оцениваются на отдельной диагностической выборке.', '',
        '[Все 20 графиков одним PDF](all_figures_with_diagnostics.pdf).', '',
        body, '', end, '',
    ])
    report_path = output / 'report.md'
    text = report_path.read_text()
    if begin in text:
        text = re.sub(re.escape(begin) + r'.*?' + re.escape(end) + r'\n*', lambda _: block + '\n', text, flags=re.DOTALL)
    else:
        anchor = '## Ограничения простыми словами'
        if anchor not in text:
            raise ValueError('Cannot locate the diagnostic insertion point')
        text = text.replace(anchor, block + '\n' + anchor, 1)

    merged = pymupdf.open()
    for path in [output / 'all_figures.pdf', diagnostics / 'all_figures.pdf']:
        with pymupdf.open(path) as document:
            merged.insert_pdf(document)
    pdf = output / 'all_figures_with_diagnostics.pdf'
    merged.save(pdf)
    pages = len(merged)
    merged.close()
    if pages != 20:
        raise ValueError(f'Expected 20 PDF pages, found {pages}')

    report_path.write_text(text)
    rendered = MarkdownIt('commonmark', {'html': True}).enable('table').render(text)
    html = '\n'.join([
        '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Коммуникация MoSAIC: результаты и диагностика памяти</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#222}img{display:block;max-width:100%;height:auto;margin:24px auto}table{display:block;overflow-x:auto;border-collapse:collapse;margin:24px 0}td,th{border-bottom:1px solid #ddd;padding:8px 12px;text-align:left}pre{overflow-x:auto;padding:16px;background:#f5f5f5}a{color:#355c7d}h2{margin-top:42px}</style>',
        '<p><a href="report.md">Markdown</a> · <a href="all_figures_with_diagnostics.pdf">PDF: 20 графиков</a> · <a href="endpoints.csv">CSV</a></p>',
        rendered, '</html>',
    ])
    for name in ['report.html', 'index.html']:
        (output / name).write_text(html)
    validation = {
        'diagnostic_report': str(source), 'diagnostic_report_sha256': digest(source),
        'diagnostic_figures': len(images), 'combined_figures': 20, 'combined_pdf_pages': pages,
        'report_sha256': digest(report_path),
        'diagnostic_images': {image: digest(output / image) for image in images},
        'snapshot_refreshed': False,
    }
    (output / 'diagnostics_integration.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    print(f'Integrated {len(images)} diagnostic figures; {output / "report.html"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs/communication_1m')
    parser.add_argument('--output', default='docs/experiments/communication_2026-10-12')
    parser.add_argument('--diagnostics-dir', default='docs/experiments/communication_diagnostics_2026-10-12_selected')
    parser.add_argument('--integrate-diagnostics-only', action='store_true')
    args = parser.parse_args()
    if args.integrate_diagnostics_only:
        integrate_diagnostics(args.output, args.diagnostics_dir)
        return
    records, incomplete, verified = read_runs(Path(args.root))
    output = Path(args.output)
    gallery = figures(records, output)
    report(records, incomplete, verified, output, gallery)
    if (Path(args.diagnostics_dir) / 'report.md').is_file():
        integrate_diagnostics(output, args.diagnostics_dir)
    print(f'{len(records)} completed runs; {len(gallery)} figures; {output / "report.md"}')


if __name__ == '__main__':
    main()
