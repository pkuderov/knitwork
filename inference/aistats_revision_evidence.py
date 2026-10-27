"""Regenerate AISTATS tables and vector figures from frozen logs, offline.

Historical cohorts are fixed by the July report. New cohorts are fixed by the
weekly summary. No test values are used for model selection, and no experiments
are launched. A retained CSV supplies separately reported test evaluations.
"""

from collections import defaultdict
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

from knitwork.common.state_size import measure
if __package__:
    from .aistats_reporting_checks import quality_cell, write_reporting_checks
else:
    from aistats_reporting_checks import quality_cell, write_reporting_checks


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/experiments'
PAPER = ROOT / 'article/latex'
PARAMS = {
    'rnn_L1': 10.16, 'rnn_L2': 10.04, 'rnn_L3': 10.23,
    'grnn_L1C8': 10.11, 'grnn_L2C4': 10.11, 'grnn_L2C8': 10.17,
    'grnn_L2C16': 10.09, 'grnn_L3C4': 9.99,
    'transformer': 10.07, 'transformer_64': 10.07,
}
LABELS = {
    'rnn_L1': 'GRU-L1', 'rnn_L2': 'GRU-L2', 'rnn_L3': 'GRU-L3',
    'grnn_L1C8': 'MoSAIC-L1C8', 'grnn_L2C4': 'MoSAIC-L2C4',
    'grnn_L2C8': 'MoSAIC-L2C8', 'grnn_L2C16': 'MoSAIC-L2C16',
    'grnn_L3C4': 'MoSAIC-L3C4', 'transformer': 'Transformer-256',
    'transformer_64': 'Transformer-64',
}
MID_KEYS = {
    'Grid-GRU L2C4': 'grnn_L2C4',
    'Grid-GRU L3C4': 'grnn_L3C4', 'Grid-GRU L2C8': 'grnn_L2C8',
    'GRU L2': 'rnn_L2', 'Grid-LRU L3C4': 'grnn_lru_L3C4',
    'HGRN2': 'hgrn2', 'mLSTM': 'mlstm', 'DeltaNet': 'delta_net',
    'RIMs': 'rims', 'BRIMs': 'brims',
    'GRU L1': 'rnn_L1', 'GRU L3': 'rnn_L3',
    'Grid-GRU L1C8': 'grnn_L1C8', 'Grid-LRU L1C8': 'grnn_lru_L1C8',
    'Grid-LRU L2C8': 'grnn_lru_L2C8',
    'Grid-LRU L2C4': 'grnn_lru_L2C4', 'Transformer-256': 'transformer',
}


def read(name):
    return json.loads((DATA / (name + '.json')).read_text())


def trace(run, metric):
    points = {}
    for p in sorted(run['metrics'][metric], key=lambda p: float(p.get('timestamp') or 0)):
        try:
            step, value = int(float(p['step'])), float(p['metricValue'])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and step <= 1_000_000_001:
            points[step] = value
    if not points:
        raise ValueError('No usable curve: ' + metric)
    return points


def common(traces):
    steps = sorted(set.intersection(*(set(t) for t in traces)))
    if not steps:
        raise ValueError('No common logged step')
    return steps


def stats(values):
    return {'n': len(values), 'mean': statistics.mean(values), 'sd': statistics.stdev(values)}


def budget_score(run, metric, scale=1.0):
    curve = trace(run, metric)
    step = max(curve)
    assert step >= 950_000_000, 'Full-budget result requires a late evaluation'
    return {'step': step, 'value': curve[step] / scale}


def pm(row, candidates, *, higher=False):
    return quality_cell(row['mean'], candidates, sd=row['sd'], higher=higher)


def dense_matrix_flops(row, run):
    """Forward matrix-product FLOPs, including output head; not total work."""
    name = row['model']
    if not name.startswith(('Grid-GRU', 'GRU ')):
        return None
    prefix = MID_KEYS[name] + '|'
    cfg = run['parameters']
    layers = int(cfg[prefix + 'n_layers'])
    width = row['hidden']
    if name.startswith('Grid-GRU'):
        columns = int(cfg[prefix + 'n_columns'])
        pairs = columns * (columns + 1) + (layers - 1) * columns**2
        macs = (10 * layers * columns + 2) * width**2 + 2 * pairs * width + 27 * width
    else:
        macs = 6 * layers * width**2 + 27 * width
    return 2 * macs


def accounting(run, key):
    parameters = run['parameters']
    prefix = key + '|'
    model_cfg = {name[len(prefix):]: yaml.safe_load(value) for name, value in parameters.items() if name.startswith(prefix)}
    if not model_cfg:
        raise ValueError('Missing logged model configuration: ' + key)
    cfg = {'wrapper_model': 'token', 'token_wrapper': {}, key: model_cfg}
    with redirect_stdout(io.StringIO()):
        return measure(cfg, key)


def check_pilots(analysis, runs):
    """Recheck published pilot scores against raw curves, not report prose."""
    checks = 0

    def check(value, expected):
        nonlocal checks
        assert math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-10), (value, expected)
        checks += 1

    for name in ['architecture', 'ablation']:
        group = analysis[name]
        step = group['common_step']
        ref = trace(runs[group['rows'][0]['id']], 'val/Loss')[step] / math.log(2)
        for row in group['rows']:
            value = trace(runs[row['id']], 'val/Loss')[step] / math.log(2)
            check(value, row['bpc'])
            check(value - ref, row['delta'])
            if name == 'architecture':
                check(trace(runs[row['id']], 'val_w1024/Loss')[step] / math.log(2), row['window1024_bpc'])
                late = [s for s in group['common_steps'] if s >= 800_000_000]
                ref_curve = trace(runs[group['rows'][0]['id']], 'val/Loss')
                curve = trace(runs[row['id']], 'val/Loss')
                check(statistics.mean((curve[s] - ref_curve[s]) / math.log(2) for s in late), row['late_800M_mean_delta'])
    control = next(r for key, r in runs.items() if key.startswith('6dbc5ce5'))
    for row in analysis['sparse_pairs']:
        step = row['step']
        value = trace(runs[row['id']], 'val/Loss')[step] / math.log(2)
        ref = trace(control, 'val/Loss')[step] / math.log(2)
        check(value, row['bpc'])
        check(value - ref, row['delta'])
    for row in analysis['rollout']:
        values = [trace(runs[key], 'val/Loss')[row['step']] / math.log(2) for key in row['ids']]
        check(values[0], row['r64_bpc'])
        check(values[1], row['r512_bpc'])
        check(values[1] - values[0], row['delta'])
    group = analysis['mqar']
    for row in group['rows']:
        run = runs[row['id']]
        step = group['common_step']
        check(trace(run, 'val/Acc')[step], row['val_acc'])
        check(trace(run, 'val/Loss')[step], row['val_loss'])
        check(trace(run, 'val/Exact_match')[step], 0)
        for bin_name, expected in zip(group['bins'], row['bin_acc']):
            check(trace(run, f'val/{bin_name}/Acc')[step], expected)
        check(sum(w*a for w, a in zip(group['val_query_weights'], row['bin_acc'])), row['val_acc'])
    for group in analysis['rl'].values():
        for row in group['rows']:
            values = [trace(runs[row['id']], 'eval/EpRet')[s] for s in group['last4_common_steps']]
            check(statistics.mean(values), row['last4_common_mean'])
    return checks


def table(filename, caption, label, columns, header, rows, wide=False, column_sep=None):
    env = 'table*' if wide else 'table'
    placement = 't' if wide or filename in ('aistats_mid_main.tex', 'aistats_ablation_main.tex') else 'H'
    caption = caption.replace('launches', 'runs').replace('launch', 'run')
    if any(r'\underline{' in row for row in rows):
        caption += ' Best observed scores are bold; second best are underlined.'
    source = '\n'.join([
        '% Generated offline by inference/aistats_revision_evidence.py.',
        rf'\begin{{{env}}}[{placement}]', r'\centering',
        *([rf'\setlength{{\tabcolsep}}{{{column_sep}pt}}'] if column_sep is not None else []),
        rf'\caption{{{caption}}}', rf'\label{{{label}}}',
        rf'\begin{{tabular}}{{{columns}}}', r'\toprule',
        header + r' \\', r'\midrule',
        *[row if row == r'\midrule' else row + r' \\' for row in rows],
        r'\bottomrule', r'\end{tabular}', rf'\end{{{env}}}', '',
    ])
    (PAPER / filename).write_text(source)


def main():
    historical = read('aistats2027_historical_quality')
    weekly = read('weekly_2026-09-28_2026-10-04_summary')
    frontier = read('weekly_2026-09-28_2026-10-04_snapshot')
    analysis = read('frontier_2026-10-04_late_analysis')
    evidence = {'historical_source': historical['retrieval_finished_utc'], 'weekly_source': frontier['retrieval_finished_utc'], 'historical': {}, 'mid': [], 'test_values_used': False}
    selected = [r for r in historical['runs'] if r['config'].split(' / ')[-1] in PARAMS]
    for task, metric in [('text8', 'val/BPC'), ('SDQ', 'Acc++')]:
        runs = [r for r in selected if r['task'] == task]
        traces = [trace(r, metric) for r in runs]
        groups = defaultdict(list)
        for run, t in zip(runs, traces):
            steps = sorted(t)[-1:] if task == 'text8' else sorted(t)[-5:]
            assert steps[-1] >= 950_000_000
            samples = [t[step] for step in steps]
            groups[run['config'].split(' / ')[-1]].append({
                'id': run['experiment_key'], 'value': statistics.mean(samples),
                'steps': steps, 'samples': samples,
            })
        evidence['historical'][task] = {'training_budget': 1_000_000_000, 'groups': {key: stats([r['value'] for r in rows]) | {'runs': rows} for key, rows in groups.items()}}

    rows = []
    text_candidates = [group['mean'] for group in evidence['historical']['text8']['groups'].values()]
    sdq_candidates = [group['mean'] for group in evidence['historical']['SDQ']['groups'].values()]
    for key in PARAMS:
        if key in ('grnn_L1C8', 'transformer'):
            rows.append(r'\midrule')
        text = evidence['historical']['text8']['groups'][key]
        sdq = evidence['historical']['SDQ']['groups'].get(key)
        label = LABELS[key]
        rows.append(f"{label} & {PARAMS[key]:.2f}M & {text['n']} & {pm(text, text_candidates)} & {pm(sdq, sdq_candidates, higher=True) if sdq else '---'}")
    table('aistats_historical_main.tex',
        r'SDQ and Text8 results at approximately ten million parameters with a one-billion-token training budget. Values are means and sample standard deviations; SDQ averages the final five online diagnostic measurements per run and has three runs per row. Parameter counts are for Text8.',
        'tab:historical', 'lrrrr', r'Model & Params & Text8 $n$ & Val. BPC $\downarrow$ & SDQ Acc++ $\uparrow$', rows, wide=True)

    runs_by_id = {r['metadata']['experimentKey']: r for r in frontier['runs']}
    evidence['pilot_scalar_checks_against_raw_logs'] = check_pilots(analysis, runs_by_id)
    baseline_rows = [r for r in weekly['text8_baselines_at980M'] if r['model'] in MID_KEYS]
    for label, model in [
        ('GRU L1', 'rnn.L1'), ('GRU L3', 'rnn.L3'),
        ('Grid-GRU L1C8', 'grnn.L1C8'), ('Grid-LRU L1C8', 'grnn_lru.L1C8'),
        ('Grid-LRU L2C8', 'grnn_lru.L2C8'),
    ]:
        group = [r for r in frontier['runs'] if r['parameters'].get('name', '').split('/seed')[0] == 'mid/' + model]
        assert len(group) == 2, 'Additional fixed two-seed cohort changed'
        baseline_rows.append({'model': label, 'ids': [r['metadata']['experimentKey'] for r in group]})
    for label, prefixes in [
        ('Grid-LRU L2C4', ['6c724575', '55de969d', '5c0724b5']),
        ('Transformer-256', ['ad72c9a0', '37180ab2']),
    ]:
        baseline_rows.append({'model': label, 'ids': [next(key for key in runs_by_id if key.startswith(prefix)) for prefix in prefixes]})
    order = ['Grid-GRU L3C4', 'Grid-GRU L2C4', 'Grid-GRU L2C8', 'Grid-GRU L1C8', 'GRU L1', 'GRU L2', 'GRU L3', 'Grid-LRU L1C8', 'Grid-LRU L2C4', 'Grid-LRU L2C8', 'Grid-LRU L3C4', 'Transformer-256', 'HGRN2', 'mLSTM', 'DeltaNet', 'RIMs', 'BRIMs']
    baseline_rows.sort(key=lambda r: order.index(r['model']))
    for row in baseline_rows:
        key = MID_KEYS[row['model']]
        params, state, hidden = accounting(runs_by_id[row['ids'][0]], key)
        values = []
        measured_steps = []
        seeds = []
        for key_id in row['ids']:
            run = runs_by_id[key_id]
            assert float(run['parameters']['eval|val_size']) == 5e6
            assert float(run['parameters']['eval|test_size']) == 5e6
            seeds.append(int(run['parameters']['seed']))
            sample = budget_score(run, 'val/Loss', math.log(2))
            values.append(sample['value'])
            measured_steps.append(sample['step'])
        score = stats(values)
        assert set(seeds) == ({42, 1337} if len(seeds) == 2 else {42, 1337, 31337})
        evidence['mid'].append({'model': row['model'], 'ids': row['ids'], 'n': score['n'], 'mean_bpc': score['mean'], 'sample_sd': score['sd'], 'params': params, 'state_scalars': state, 'hidden': hidden, 'seeds': seeds, 'measured_steps': measured_steps, 'values': values, 'training_budget': 1_000_000_000})
    rows = []
    boundaries = {'Grid-GRU L3C4', 'GRU L1', 'Grid-LRU L1C8', 'Transformer-256', 'RIMs'}
    for r in evidence['mid']:
        if rows and r['model'] in boundaries:
            rows.append(r'\midrule')
        value = pm({'mean': r['mean_bpc'], 'sd': r['sample_sd']}, [row['mean_bpc'] for row in evidence['mid']])
        rows.append(f"{r['model'].replace('Grid-GRU', 'MoSAIC')} & {r['params']/1e6:.3f}M & {r['n']} & {value}")
    table('aistats_mid_main.tex',
        'Text8 validation results at approximately one million parameters with a one-billion-token training budget. Values are means and sample standard deviations across runs. Models share the split and training recipe; state and computation differ.',
        'tab:mid', 'lrrr', 'Model & Size & Runs & BPC', rows, column_sep=4)
    rows = []
    for r in evidence['mid']:
        flops = dense_matrix_flops(r, runs_by_id[r['ids'][0]])
        r['forward_dense_matrix_flops_per_token'] = flops
        value = '---' if flops is None else f'{flops/1e6:.3f}'
        rows.append(f"{r['model'].replace('Grid-GRU', 'MoSAIC')} & {r['hidden']} & {r['params']:,} & {r['state_scalars']:,} & {4*r['state_scalars']/1024:.2f} & {value}")
    lru_l2 = next(r for r in frontier['runs'] if r['metadata']['experimentKey'].startswith('6c724575'))
    p, s, h = accounting(lru_l2, 'grnn_lru_L2C4')
    transformer = next(r for r in frontier['runs'] if r['metadata']['experimentKey'].startswith('ad72c9a0'))
    tp, ts, th = accounting(transformer, 'transformer')
    table('aistats_mid_state.tex',
        r'Text8 model accounting at approximately one million parameters, including token adapters ($V=27$). State includes recurrent/public-message or key/value caches; float32 storage excludes training intermediates and optimizer state. Dense MFLOPs/token count forward matrix products for GRU/MoSAIC only; other families are uncounted, not zero. Appendix~\ref{app:midprotocol} specifies the estimate.',
        'tab:midstate', 'lrrrrr', r'Model & Width & Params & State scalars & KiB & Dense MFLOPs/token', rows)
    evidence['mid_grid_lru_L2_accounting'] = {'params': p, 'state_scalars': s, 'hidden': h}
    evidence['mid_transformer_accounting'] = {'params': tp, 'state_scalars': ts, 'hidden': th}
    evidence['reporting_checks'] = write_reporting_checks(evidence, runs_by_id)
    evidence['test_values_used'] = True
    evidence['test_values_used_for_selection'] = False

    control = runs_by_id[analysis['architecture']['rows'][0]['id']]
    reference = budget_score(control, 'val/Loss', math.log(2))
    for row in analysis['architecture']['rows']:
        run = runs_by_id[row['id']]
        sample = budget_score(run, 'val/Loss', math.log(2))
        row.update(bpc=sample['value'], delta=sample['value']-reference['value'], measured_step=sample['step'])
        row['window1024_bpc'] = trace(run, 'val_w1024/Loss')[sample['step']] / math.log(2)
    analysis['architecture'].pop('common_step')
    analysis['architecture']['training_budget'] = 1_000_000_000
    control = next(r for key, r in runs_by_id.items() if key.startswith('6dbc5ce5'))
    reference = budget_score(control, 'val/Loss', math.log(2))
    evidence['sparse_control'] = {'id': control['metadata']['experimentKey'], 'bpc': reference['value']}
    for row in analysis['sparse_pairs']:
        sample = budget_score(runs_by_id[row['id']], 'val/Loss', math.log(2))
        row.update(bpc=sample['value'], delta=sample['value']-reference['value'], step=sample['step'])
    evidence['topk'] = []
    for model in ['dense', 'top2', 'top1']:
        if model == 'dense':
            run = next(r for key, r in runs_by_id.items() if key.startswith('9a7f6783'))
        else:
            run = next(r for r in frontier['runs'] if r['parameters'].get('name', '').split('/seed')[0] == 'mid/grnn.L2C4_' + ('topk2' if model == 'top2' else 'topk1'))
        evidence['topk'].append({'model': model, 'id': run['metadata']['experimentKey']} | budget_score(run, 'val/Loss', math.log(2)))

    names = {'full': 'Full', 'none': 'All three off', 'no_noise': 'No logit noise', 'no_comm': 'No communication cost', 'no_entropy': 'No entropy bonus'}
    rows = [f"{names[r['variant']]} & {quality_cell(r['bpc'], [x['bpc'] for x in analysis['ablation']['rows']])} & ${r['delta']:+.4f}$" for r in analysis['ablation']['rows']]
    table('aistats_ablation_main.tex',
        r'MoSAIC-L2C4 regularizer ablations in a 67M-token pilot, seed 42, at a shared early-training checkpoint. Widths, optimizer and capped validation are shared. $\Delta$ is BPC minus the full control, computed before rounding. One run per condition.',
        'tab:ablation', 'lrr', r'Condition & BPC $\downarrow$ & $\Delta$', rows)
    rows = [f"{r['variant'].replace('_', ' ')} & {quality_cell(r['bpc'], [x['bpc'] for x in analysis['architecture']['rows']])} & ${r['delta']:+.4f}$ & ${r['late_800M_mean_delta']:+.4f}$ & {quality_cell(r['window1024_bpc'], [x['window1024_bpc'] for x in analysis['architecture']['rows']])} & ${r['window1024_bpc']-r['bpc']:+.4f}$" for r in analysis['architecture']['rows']]
    table('aistats_lru_variants.tex',
        r'Five LRU mechanisms and their zero-query-initialized dynamic-hub control on Text8, seed 42, with a 1B-token training budget. Late $\Delta$ averages checkpoint-aligned differences. Reset-1024 resets state every 1,024 valid tokens; its penalty is relative to continuous BPC. Differences are computed before rounding. Widths are L2C4H180; projections and matrix-memory replacement change parameter/state counts.',
        'tab:lruvariants', 'lrrrrr', r'Variant & BPC $\downarrow$ & $\Delta$ & Late $\Delta$ & Reset-1024 & Reset penalty', rows)
    candidates = [reference['value']] + [r['bpc'] for r in analysis['sparse_pairs']]
    rows = [f"Dense control & {quality_cell(reference['value'], candidates)} & $+0.0000$"]
    rows += [f"{r['variant'].replace('_', ' ')} & {quality_cell(r['bpc'], candidates)} & ${r['delta']:+.4f}$" for r in analysis['sparse_pairs']]
    table('aistats_sparse_lru.tex',
        r'Sparse LRU pilots on Text8, seed 42, with a 1B-token training budget. $\Delta$ is relative to the displayed dense control, computed before rounding. Dynamic hub uses standard query initialization. Block has a 0.531M core versus approximately 1.05M for the control.',
        'tab:sparselru', 'lrr', r'Variant & BPC $\downarrow$ & $\Delta$', rows)
    rows = [f"{r['model'].replace('_', '-')} & {quality_cell(r['r64_bpc'], [x['r64_bpc'] for x in analysis['rollout']])} & {quality_cell(r['r512_bpc'], [x['r512_bpc'] for x in analysis['rollout']])} & ${r['delta']:+.4f}$" for r in analysis['rollout']]
    table('aistats_rollout.tex',
        r'Rollout pilots, seed 42; BPC with the same capped validation. Length 64 uses 512 training streams, length 512 uses 64: both have 32,768 tokens/update. Longer BPTT changes the number of independent streams and does not isolate context length alone.',
        'tab:rollout', 'lrrr', r'Model & Rollout 64 & Rollout 512 & $\Delta$', rows)
    mqar_rows = analysis['mqar']['rows']
    rows = [f"{r['variant'].replace('_', ' ')} & {quality_cell(100*r['val_acc'], [100*x['val_acc'] for x in mqar_rows], higher=True, decimals=3)} & {quality_cell(r['val_loss'], [x['val_loss'] for x in mqar_rows], decimals=3)} & " + ' & '.join(quality_cell(100*value, [100*x['bin_acc'][i] for x in mqar_rows], higher=True, decimals=3) for i, value in enumerate(r['bin_acc'])) for r in mqar_rows]
    table('aistats_mqar.tex',
        r'MQAR pilots, seed 42: query-weighted validation accuracy (\%) and cross-entropy (nats), followed by bin accuracies (\%). All sequence-level exact-match accuracies are zero. Each bin contains 1,000 sequences; no strong-baseline or repeated-seed comparison is available.',
        'tab:mqar', 'lrrrrrrr', r'Variant & Overall & CE & 64/4 & 128/8 & 256/16 & 256/32 & 256/64', rows)
    rows = []
    for task, group in analysis['rl'].items():
        if rows:
            rows.append(r'\midrule')
        for r in group['rows']:
            value = quality_cell(r['last4_common_mean'], [x['last4_common_mean'] for x in group['rows']], higher=True)
            rows.append(f"{task} & {r['model'].replace('_', '-')} & {value}")
    table('aistats_rl.tex',
        r'POPGym pilots, seed 42: mean return over four late evaluations. Each evaluation uses 32 stochastic-policy episodes with fresh environment seed 50000. Evaluations share one training trajectory; their mean is not a seed-level uncertainty estimate.',
        'tab:rl', 'llr', r'Environment & Model & Late return $\uparrow$', rows)
    evidence['pilots'] = {key: analysis[key] for key in ['ablation', 'architecture', 'sparse_pairs', 'rollout', 'mqar', 'rl']}

    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    for axis, task, metric, keys in [
        (axes[0], 'text8', 'val/BPC', ['rnn_L2', 'rnn_L3', 'grnn_L2C4', 'grnn_L3C4', 'transformer']),
        (axes[1], 'SDQ', 'Acc++', ['rnn_L1', 'rnn_L2', 'grnn_L2C4', 'grnn_L3C4']),
    ]:
        for key in keys:
            traces = [trace(r, metric) for r in selected if r['task'] == task and r['config'].endswith(' / ' + key)]
            steps = common(traces)
            values = np.asarray([[t[s] for s in steps] for t in traces])
            x, mean, sd = np.asarray(steps) / 1e9, values.mean(0), values.std(0, ddof=1)
            axis.plot(x, mean, label=LABELS[key], lw=1.3)
            axis.fill_between(x, mean-sd, mean+sd, alpha=.13)
        axis.set(xlabel='Training tokens (billions)', ylabel='Validation BPC' if task == 'text8' else 'Online Acc++', title='(a) Text8, ~10M' if task == 'text8' else '(b) SDQ, ~10M')
        axis.set_ylim((1.40, 1.86) if task == 'text8' else (.1, 1))
        axis.legend(fontsize=6.8, frameon=False)
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(PAPER / 'fig_aistats_historical_curves.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    for label in ['Grid-GRU L2C4', 'Grid-GRU L3C4', 'GRU L2', 'Grid-LRU L3C4']:
        row = next(r for r in evidence['mid'] if r['model'] == label)
        traces = [trace(runs_by_id[k], 'val/Loss') for k in row['ids']]
        steps = common(traces)
        values = np.asarray([[t[s]/math.log(2) for s in steps] for t in traces])
        x, mean, sd = np.asarray(steps)/1e9, values.mean(0), values.std(0, ddof=1)
        axes[0].plot(x, mean, label=label.replace('Grid-GRU', 'MoSAIC'), lw=1.3)
        axes[0].fill_between(x, mean-sd, mean+sd, alpha=.13)
    for r in analysis['mqar']['rows']:
        axes[1].plot(range(5), np.asarray(r['bin_acc'])*100, marker='o', ms=2.5, label=r['variant'].replace('_', ' '))
    axes[0].set(xlabel='Training tokens (billions)', ylabel='Validation BPC', title='(a) Text8, ~1M', ylim=(1.52, 2.03))
    axes[1].set(xlabel='Sequence length / key-value pairs', ylabel='Query accuracy (%)', title='(b) MQAR, seed 42', xticks=range(5), xticklabels=['64/4', '128/8', '256/16', '256/32', '256/64'])
    for axis in axes:
        axis.legend(fontsize=6.5, frameon=False)
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(PAPER / 'fig_aistats_mid_memory.pdf')
    plt.close(fig)
    (DATA / 'aistats2027_revision_evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print('Generated tables, vector figures, and provenance; no network or training.')


if __name__ == '__main__':
    main()
