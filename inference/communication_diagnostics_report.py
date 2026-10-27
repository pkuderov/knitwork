"""Restrained figures and plain-language report for frozen column diagnostics."""

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from inference.report_plot_style import prepare_figure, text_overlaps


ORDER = ['dynamic_column0', 'static_column0', 'none_column0', 'dynamic_mean', 'static_mean', 'none_mean']
HEADS = ['C0', 'C1', 'C2', 'C3', 'Mean', 'Concat']


def main(output):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    rows = [json.loads(path.read_text()) for path in output.glob('runs/*/*/result.json')]
    if len(rows) != len(manifest['runs']):
        raise ValueError('Some checkpoint diagnostics are incomplete')
    rows.sort(key=lambda row: (row['task'], ORDER.index(row['condition'])))
    tasks = {task: [row for row in rows if row['task'] == task] for task in ('text8', 'sdq')}
    figures = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'figure.constrained_layout.use': True})
    pdf = PdfPages(output / 'all_figures.pdf')

    def save(fig, name, caption):
        prepare_figure(fig)
        collisions = text_overlaps(fig)
        if collisions:
            raise ValueError(f'Figure has overlapping text: {name}: {collisions}')
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(output / f'{name}.{extension}', dpi=180, bbox_inches='tight', facecolor='white')
        pdf.savefig(fig, bbox_inches='tight')
        figures.append({'name': name, 'caption': caption, 'text_overlaps': collisions})
        plt.close(fig)

    def heatmap(values, labels, columns, name, metric, caption, *, decimals=1, diverging=False):
        values = np.asarray(values, dtype=float)
        fig, ax = plt.subplots(figsize=(9.8, max(3.2, len(labels) * 0.55 + 1.3)))
        if diverging:
            extent = max(float(np.nanmax(np.abs(values))), 1e-5)
            image = ax.imshow(values, aspect='auto', cmap='RdBu_r', vmin=-extent, vmax=extent)
        else:
            image = ax.imshow(values, aspect='auto', cmap='Blues')
        ax.set_xticks(range(len(columns)), columns)
        ax.set_yticks(range(len(labels)), labels)
        for i, j in np.ndindex(values.shape):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            limit = max(abs(image.norm.vmin), abs(image.norm.vmax))
            white = abs(value) > limit * 0.65 if diverging else image.norm(value) > 0.6
            ax.text(j, i, f'{value:.{decimals}f}', ha='center', va='center', color='white' if white else '#252525', fontsize=9)
        fig.colorbar(image, ax=ax, label=metric, fraction=0.035, pad=0.035)
        save(fig, name, caption)

    for task in ('text8', 'sdq'):
        for layer in (0, 1):
            selected = tasks[task]
            fig, axes = plt.subplots(2, 3 if task == 'text8' else 2, figsize=(11, 7))
            for ax, row in zip(axes.flat, selected):
                values = np.asarray(row['cka'][layer], dtype=float)
                image = ax.imshow(values, cmap='Blues', vmin=0, vmax=1)
                ax.set_title(row['condition'].replace('_', ' / '))
                ax.set_xticks(range(4), ['C0', 'C1', 'C2', 'C3'])
                ax.set_yticks(range(4), ['C0', 'C1', 'C2', 'C3'])
                for i, j in np.ndindex(values.shape):
                    if np.isfinite(values[i, j]):
                        ax.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center', fontsize=8, color='white' if values[i, j] > 0.6 else '#252525')
            for ax in list(axes.flat)[len(selected):]:
                ax.set_visible(False)
            fig.colorbar(image, ax=list(axes.flat), label='CKA', fraction=0.025, pad=0.035)
            save(fig, f'cka_{task}_L{layer+1}', f'{task.upper()}, слой {layer+1}: linear CKA состояний на одинаковых проверочных событиях. Высокое значение означает сходство представлений; низкое не доказывает статистическую независимость.')

    for field, filename, caption in (
        ('stored_values', 'sdq_stored_value', 'SDQ: accuracy линейного чтения последнего сохранённого значения запрошенного ключа. Включены только запросы с предшествующей записью.'),
        ('targets', 'sdq_query_answer', 'SDQ: accuracy линейного чтения полного ответа задачи, включая distractor accumulator и store/query counters. Native — исходная голова на той же выборке известных записей.'),
    ):
        selected = tasks['sdq']
        values = [[row['probes'][field][head]['validation']['Acc'] * 100 for head in HEADS] for row in selected]
        columns = HEADS
        if field == 'targets':
            values = [[row['native_sample']['Acc'] * 100, *scores] for row, scores in zip(selected, values)]
            columns = ['Native', *HEADS]
        heatmap(values, [row['condition'] for row in selected], columns, filename, 'Acc (%)', caption)

    selected = tasks['sdq']
    values = [[-lesion['delta_Acc'] * 100 for lesion in row['query_state_lesions']] for row in selected]
    heatmap(values, [row['condition'] for row in selected], HEADS[:4], 'sdq_query_state_lesions', 'Acc drop (pp)',
        'SDQ: перед текущим запросом обнулены оба слоя одной колонки; исходная голова сохранена. Положительное число — падение accuracy. Изменение не переносится на следующие события.', diverging=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
    for ax, row in zip(axes.flat, selected):
        values = np.asarray([[bin['Acc'] * 100 if bin['count'] >= 32 and bin['Acc'] is not None else np.nan
            for bin in row['probes']['stored_values'][head]['gaps']] for head in HEADS])
        image = ax.imshow(values, vmin=0, vmax=100, cmap='Blues', aspect='auto')
        ax.set_title(row['condition'].replace('_', ' / '))
        ax.set_xticks(range(6), ['0–7', '8–31', '32–127', '128–255', '256–511', '512+'], rotation=30, ha='right')
        ax.set_yticks(range(6), HEADS)
        ax.set_xlabel('Gap')
        for i, j in np.ndindex(values.shape):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f'{values[i,j]:.0f}', va='center', ha='center', fontsize=8, color='white' if values[i,j] > 60 else '#252525')
    fig.colorbar(image, ax=list(axes.flat), label='Acc (%)', fraction=0.025, pad=0.035)
    save(fig, 'sdq_stored_value_gaps', 'Чтение сохранённого значения на естественных SDQ gaps. В ячейках с менее чем 32 запросами число скрыто. Это интервалы естественных gaps, а не исходный fixed-gap benchmark.')

    selected = tasks['text8']
    values = [[row['native_sample']['BPC'], *[row['probes']['targets'][head]['validation']['BPC'] for head in HEADS]] for row in selected]
    heatmap(values, [row['condition'] for row in selected], ['Native', *HEADS], 'text8_readouts', 'BPC',
        'Text8: исходная голова и дополнительные линейные головы на замороженной памяти. Новые головы обучались на train-отрывках; таблица оценивает отдельные validation-отрывки.', decimals=3)
    heatmap([[share * 100 for share in row['concat_weight_share']] for row in selected],
        [row['condition'] for row in selected], HEADS[:4], 'text8_concat_weights', 'Share (%)',
        'Text8: доли норм четырёх блоков дополнительной concat-головы после учёта масштаба признаков и удаления общего для классов сдвига. Это описание декодера, а не уникальная причинная доля колонки.')
    heatmap([[entry['delta_BPC'] for entry in row['concat_column_replacement']] for row in selected],
        [row['condition'] for row in selected], HEADS[:4], 'text8_concat_column_replacement', 'BPC increase',
        'Text8: блок состояния заменён его средним на probe-train; concat-голова не переобучалась. Положительное число — ухудшение. Взаимозаменяемость коррелированных колонок остаётся ограничением.', decimals=3, diverging=True)
    selected = tasks['sdq']
    heatmap([[-entry['delta_Acc'] * 100 for entry in row['concat_column_replacement']] for row in selected],
        [row['condition'] for row in selected], HEADS[:4], 'sdq_concat_column_replacement', 'Acc drop (pp)',
        'SDQ: замена блока верхнего состояния средним для дополнительной concat-головы полного ответа. Это доступность информации для нового декодера; исходная модель остаётся неизменной.', diverging=True)
    native_text = None
    if (output / 'text8_lesions.json').exists():
        native_text = json.loads((output / 'text8_lesions.json').read_text())
        lesions = sorted(native_text['rows'], key=lambda row: ORDER.index(row['condition']))
        heatmap([[entry['delta_BPC'] for entry in row['column_lesions']] for row in lesions],
            [row['condition'] for row in lesions], HEADS[:4], 'text8_native_column_lesions', 'BPC increase',
            'Text8: перед токеном обнуляются оба слоя одной колонки; исходные router и prediction head сохранены. Каждая из 3584 позиций сравнивается со своим неизменным состоянием. Интервенция не переносится на последующие токены.', decimals=3, diverging=True)
    pdf.close()

    selected_probes = 'probe_selection' in manifest['settings']
    probe_description = (
        'Каждая новая голова — единственный линейный слой с bias. Проверены C0, C1, C2, C3, Mean и Concat. Последние четыре из шестнадцати probe-train потоков используются только для выбора: по их cross-entropy выбирается L2-штраф 0/0.01/0.1 на отклонение от начальных весов и checkpoint 0/50/100/…/400 updates. Остальные двенадцать потоков используются для градиентов. Mean/std признаков вычислены только по fitting-потокам. Основная validation не участвует в выборе. Optimizer AdamW, batch 512, LR=0.01, weight_decay=0.0001. Для полного ответа головы начинают с исходной trained head: concat сначала точно воспроизводит исходный column0 или mean readout; для декодирования сырого сохранённого значения используется нулевая инициализация. Сохранены эквивалентные веса в исходных координатах.'
        if selected_probes else
        'Каждая новая голова — единственный линейный слой с bias. Отдельно проверены C0, C1, C2, C3, Mean и Concat. Признаки стандартизованы по probe-train; эквивалентные веса сохранены в исходных координатах. Использованы фиксированные 400 updates AdamW, batch 512, LR=0.01, weight_decay=0.0001, без выбора настроек по validation.'
    )
    reproduction = (
        '`OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python inference/select_communication_readouts.py --output /tmp/communication-diagnostics-reproduced`'
        if selected_probes else
        '`OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python inference/communication_diagnostics.py`'
    )
    text = [
        '# Диагностика памяти завершённых локальных ablation',
        '',
        f'Проанализированы {len(rows)} финальных checkpoints на полном бюджете около 1B токенов: {len(tasks["text8"])} Text8 и {len(tasks["sdq"])} SDQ. Все модели — L2C4/H136, clean, seed 42. Незавершённые запуски не включены. Основные веса не менялись; дополнительные линейные головы обучены отдельно на CPU. Это диагностические результаты одного seed, а не новые полноценные обучения backbone.',
        '',
        '## Что измерено',
        '',
        'Linear CKA сравнивает геометрию состояний двух колонок на одних и тех же событиях: ||XᵀY||²_F / (||XᵀX||_F · ||YᵀY||_F), после центрирования признаков. Значения близки к 1 при сходных представлениях. CKA не доказывает независимость, специализацию или причинный обмен. Общий вход может сделать состояния похожими даже у независимых колонок. Для none_column0 колонки 1–3 не обучаются основной задачей; их CKA не следует трактовать как приобретённую специализацию.',
        '',
        probe_description + ' Недостаток информации и недостаточное обучение probe не всегда различимы; плохой линейный probe не доказывает отсутствия нелинейно закодированной памяти.',
        '',
        'Text8: 16 случайно выбранных непрерывных отрывков по 1024 шага отдельно из train и validation; первые 128 шагов каждого отрывка служат прогревом. Проверка содержит 14336 предсказаний, поэтому её BPC не заменяет основной full-validation BPC. Шесть новых голов обучаются предсказывать следующий символ из замороженных верхних состояний. Norm/share блоков concat-головы дополнены заменой каждого блока его train-средним при неизменных остальных блоках и весах головы.',
        '',
        'SDQ: 16 потоков × 2048 шагов, T=500, p_store=0.1, p_query=0.25. Probe-train и validation используют разные генераторы и seed; первые 128 шагов исключены. Все модели получают одинаковые последовательности. Анализируется только query с известной предшествующей записью. Одна группа probes декодирует последнее сохранённое значение запрошенного ключа; вторая — полный ответ с накопителем distractors и счётчиками. Это разные цели. Native accuracy здесь тоже измерена только на таких queries и не равна общей held-out accuracy отчёта.',
        '',
        'В отдельной SDQ-интервенции непосредственно перед query обнуляется состояние обоих слоёв одной колонки. Query-токен, правильный ответ, остальные состояния и исходная голова сохраняются; исходная траектория продолжается без этого изменения. Такая проверка учитывает доступ через коммуникацию, в отличие от простого удаления блока в новой выходной голове.',
        '',
        '## Числа на диагностической выборке',
        '',
        '| Задача | Условие | Queries/tokens | Native | Probe C0 | Probe Concat |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        metric = 'BPC' if row['task'] == 'text8' else 'Acc'
        factor = 1 if metric == 'BPC' else 100
        scores = [row['native_sample'][metric], row['probes']['targets']['C0']['validation'][metric], row['probes']['targets']['Concat']['validation'][metric]]
        numbers = [f'{value*factor:.3f}' + ('%' if factor == 100 else '') for value in scores]
        text.append(f'| {row["task"]} | {row["condition"]} | {row["validation_probe_samples"]} | ' + ' | '.join(numbers) + ' |')
    if selected_probes:
        dynamic = next(row for row in tasks['sdq'] if row['condition'] == 'dynamic_column0')
        top_cka = dynamic['cka'][1][2][3]
        probes = dynamic['probes']['targets']
        text += ['', '## Что уже видно в памяти', '',
            f'В SDQ dynamic_column0 верхние колонки 2 и 3 очень похожи: CKA={top_cka:.3f}. Из колонки 0 небольшой линейный probe декодирует полный ответ с accuracy {probes["C0"]["validation"]["Acc"]*100:.2f}%, из остальных — около 10%. Исходная голова на этой отдельной выборке известных записей даёт {dynamic["native_sample"]["Acc"]*100:.2f}%. Значит, часть ухудшения можно уменьшить заменой декодера, но этот probe не восстанавливает результат static_column0. Подобие двух колонок — признак избыточности представлений на выбранных событиях, не доказательство общей причины провала.', '',
            'При обнулении памяти каждой колонки SDQ dynamic_column0 ухудшается. Остальные колонки функционально используются, даже если полный ответ плохо читается из них отдельной линейной головой. Для dynamic_mean ответ доступен линейному чтению из каждой верхней колонки существенно лучше. Это разные организации информации; из этих наблюдений нельзя выбрать единственную причину различий без дополнительных controlled trainings.', '',
            'В Text8 train-калибровка выбрала исходные веса concat-кандидата, update=0, у всех пяти моделей. Маленькая выборка не поддержала улучшение исходного декодера; обученные кандидаты переобучались. Поэтому нормы concat-блоков в этой серии в основном отражают исходную схему column0/mean. Их нельзя выдавать за обнаруженные уникальные вклады колонок. Для оценки функционального участия дополнительно выполнено удаление recurrent memory перед токеном при неизменном исходном декодере.', '',
            'Probe сырого сохранённого SDQ-значения заметно слабее probe полного ответа. Это не доказывает отсутствия записи: SDQ требует ответа с дополнительной арифметикой, и память может кодировать преобразованное значение. Линейный decoder имеет ограниченную выразительность.', '']
    if native_text:
        text += ['### Участие колонок в исходном Text8-предсказании', '',
            '| Условие | ΔBPC C0 | ΔBPC C1 | ΔBPC C2 | ΔBPC C3 |', '|---|---:|---:|---:|---:|']
        for row in sorted(native_text['rows'], key=lambda row: ORDER.index(row['condition'])):
            text.append('| ' + row['condition'] + ' | ' + ' | '.join(f'{entry["delta_BPC"]:+.3f}' for entry in row['column_lesions']) + ' |')
        text += ['', 'Все числа — увеличение BPC при обнулении обоих слоёв указанной колонки перед текущим токеном. В communicating column0 моделях колонки 1–3 влияют на ответ косвенно через router, хотя выходная голова читает только колонку 0. У none_column0 удаление 1–3 не меняет ни одного logit: это проверено точным сравнением массивов. Падения при удалении разных колонок не складываются в проценты уникальных вкладов; повреждение памяти — сильная интервенция.', '']
    text += ['', '## Как это связано с низкой SDQ accuracy dynamic_column0', '',
        'Хороший серверный Grid-GRU из отчёта H100 оценивался на Text8: 1.5966 ± 0.0050 BPC у L2C4 против локальных 1.6006 BPC. Это не демонстрация успешного SDQ в той серверной серии. Серверная Text8-конфигурация также использовала routing noise 0.05, communication penalty 0.05 и entropy bonus 0.005; локальные clean-controls их отключают.', '',
        'На SDQ локальная training EMA dynamic_column0 доходит до 69.84%, а полная независимая accuracy последнего checkpoint — 45.71%. Лучшая записанная независимая accuracy — 66.01% около 840M; в поздних точках она резко колеблется. Dynamic_mean даёт 79.79%, static_column0 — 85.54%. Это указывает на чувствительность dynamic/single-column readout, но не устанавливает единственную причину. Новые probes и интервенции помогают отделить отсутствие доступной информации от ограничения исходного декодера. Они не доказывают, что выключение регуляризаторов вызвало падение: для этого нужен отдельный matched paper/clean contrast на SDQ и несколько seed.', '',
        '## Графики', '']
    for figure in figures:
        text += [f'![{figure["name"]}]({figure["name"]}.png)', '', figure['caption'], '']
    text += ['## Воспроизведение и артефакты', '',
        'Команда из корня репозитория: ' + reproduction + '. Для повторного построения графиков без вычислений: `python -m inference.communication_diagnostics_report <output_dir>`. Manifest фиксирует входные checkpoints и их SHA256, настройки, источники реализации. В `runs/<task>/<condition>/` находятся features.npz, result.json и веса всех обученных probes. Исходные checkpoints только читаются. Все графики доступны в PNG/SVG/PDF, общий файл — [all_figures.pdf](all_figures.pdf).', '',
        ('Первый пилот с фиксированными 400 updates и нулевой инициализацией показал переобучение concat-головы на коротких Text8-отрывках; он сохранён в соседней папке communication_diagnostics_2026-10-12. Этот отчёт использует выбор на training-потоках и сохранённые признаки того же пилота. Протокол уточнялся после просмотра пилота: основная validation не является полностью нетронутым test. Уточнение метода не меняет CKA и интервенции с обнулением памяти, которые рассчитаны по исходным frozen checkpoints.' if selected_probes else 'Фиксированные головы этого пилота могут переобучаться на коротких отрывках; их веса нельзя считать окончательной оценкой вкладов. Основная версия с выбором на training-потоках сохраняется отдельно в communication_diagnostics_2026-10-12_selected.'), '',
        'Нет междуобученческих confidence intervals: один seed. Блоки времени и отдельные запросы не заменяют независимые обучения. Линейная доступность информации, сходство состояний и вклад в исходное предсказание — разные свойства.', '']
    if native_text:
        text += ['Дополнительная команда для Text8 memory lesions при воспроизведении: `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python inference/text8_column_lesions.py --directory /tmp/communication-diagnostics-reproduced`, затем `python -m inference.communication_diagnostics_report /tmp/communication-diagnostics-reproduced`. Новый output directory необходим при изменении хэшей исходников; существующие результаты не перезаписываются другой реализацией probes. Данные и хэш реализации сохранены в text8_lesions.json, попарные logits и targets — в native_text_lesions.npz каждой модели.', '']
    (output / 'report.md').write_text('\n'.join(text))
    import hashlib
    write = {'checkpoint_count': len(rows), 'figures': figures, 'all_text_overlap_checks_passed': True,
        'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if native_text:
        write['native_text_lesions_sha256'] = hashlib.sha256((output / 'text8_lesions.json').read_bytes()).hexdigest()
    (output / 'validation.json').write_text(json.dumps(write, indent=2) + '\n')
    (output / 'analysis.json').write_text(json.dumps({'manifest': manifest, 'runs': rows}, indent=2, allow_nan=False) + '\n')
    gallery = ''.join(f'<figure><img src="{figure["name"]}.svg"><figcaption>{html.escape(figure["caption"])}</figcaption></figure>' for figure in figures)
    (output / 'index.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Column diagnostics</title><style>body{font-family:system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#222}img{width:100%}figure{margin:30px 0}figcaption{color:#555;line-height:1.5}</style><h1>Диагностика памяти локальных ablation</h1><p><a href="report.md">Полный отчёт</a> · <a href="all_figures.pdf">PDF графиков</a></p>' + gallery + '</html>')
    print(f'Report written: {output / "report.md"}; {len(figures)} figures', flush=True)


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
