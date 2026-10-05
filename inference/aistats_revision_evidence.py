"""Regenerate AISTATS tables and vector figures from frozen logs, offline.

Historical cohorts are fixed by the July report. New cohorts are fixed by the
weekly summary. No test values are read, and no experiments are launched.
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
    'Mamba': 'mamba', 'Grid-GRU L2C4': 'grnn_L2C4',
    'Grid-GRU L3C4': 'grnn_L3C4', 'Grid-GRU L2C8': 'grnn_L2C8',
    'GRU L2': 'rnn_L2', 'Grid-LRU L3C4': 'grnn_lru_L3C4',
    'HGRN2': 'hgrn2', 'mLSTM': 'mlstm', 'DeltaNet': 'delta_net',
    'RIMs': 'rims', 'BRIMs': 'brims',
    'GRU L1': 'rnn_L1', 'GRU L3': 'rnn_L3',
    'Grid-GRU L1C8': 'grnn_L1C8', 'Grid-LRU L1C8': 'grnn_lru_L1C8',
    'Grid-LRU L2C8': 'grnn_lru_L2C8',
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


def pm(row):
    return rf"${row['mean']:.4f}\pm{row['sd']:.4f}$"


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


def table(filename, caption, label, columns, header, rows, wide=False):
    env = 'table*' if wide else 'table'
    source = '\n'.join([
        '% Generated offline by inference/aistats_revision_evidence.py.',
        rf'\begin{{{env}}}[t]', r'\centering',
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
        if task == 'text8':
            endpoint = min(max(t) for t in traces)
            recurrent = [t for r, t in zip(runs, traces) if not r['config'].startswith('transformer')]
            horizon = max(s for s in common(recurrent) if s <= endpoint)
            steps = [horizon]
        else:
            steps = common(traces)[-5:]
            horizon = steps[-1]
        groups = defaultdict(list)
        for run, t in zip(runs, traces):
            samples = []
            interpolation = []
            for step in steps:
                assert min(t) <= step <= max(t), 'Extrapolation forbidden'
                samples.append(t[step] if step in t else float(np.interp(step, sorted(t), [t[s] for s in sorted(t)])))
                interpolation.append(step not in t)
            groups[run['config'].split(' / ')[-1]].append({
                'id': run['experiment_key'], 'value': statistics.mean(samples),
                'interpolated': any(interpolation), 'samples': samples,
            })
        evidence['historical'][task] = {'horizon': horizon, 'steps': steps, 'groups': {key: stats([r['value'] for r in rows]) | {'runs': rows} for key, rows in groups.items()}}

    rows = []
    for key in PARAMS:
        text = evidence['historical']['text8']['groups'][key]
        sdq = evidence['historical']['SDQ']['groups'].get(key)
        label = LABELS[key] + (r'$^{\dagger}$' if text['n'] == 2 else '')
        rows.append(f"{label} & {PARAMS[key]:.2f}M & {text['n']} & {pm(text)} & {pm(sdq) if sdq else '---'}")
    table('aistats_historical_main.tex',
        r'Historical $\sim$10M models, reaggregated at common logged budgets. Text8: 960,036,864 tokens; Transformer BPC is linearly interpolated between bracketing logs, recurrent rows are exact. SDQ: per-launch mean over the same five logs at 950.04--970.04M tokens. Values are mean $\pm$ sample SD; SDQ has three launches per row. Params are for text8. $\dagger$: two text8 launches. Acc++ is an online curriculum diagnostic.',
        'tab:historical', 'lrrrr', r'Model & Params & Text8 $n$ & Val. BPC $\downarrow$ & SDQ Acc++ $\uparrow$', rows, wide=True)

    runs_by_id = {r['metadata']['experimentKey']: r for r in frontier['runs']}
    evidence['pilot_scalar_checks_against_raw_logs'] = check_pilots(analysis, runs_by_id)
    baseline_rows = list(weekly['text8_baselines_at980M'])
    for label, model in [
        ('GRU L1', 'rnn.L1'), ('GRU L3', 'rnn.L3'),
        ('Grid-GRU L1C8', 'grnn.L1C8'), ('Grid-LRU L1C8', 'grnn_lru.L1C8'),
        ('Grid-LRU L2C8', 'grnn_lru.L2C8'),
    ]:
        group = [r for r in frontier['runs'] if r['parameters'].get('name', '').split('/seed')[0] == 'mid/' + model]
        assert len(group) == 2, 'Additional fixed two-seed cohort changed'
        values = [trace(r, 'val/Loss')[980_037_632] / math.log(2) for r in group]
        score = stats(values)
        baseline_rows.append({'model': label, 'step': 980_037_632, 'n': score['n'], 'mean_bpc': score['mean'], 'sample_sd': score['sd'], 'ids': [r['metadata']['experimentKey'] for r in group]})
    order = ['Mamba', 'Grid-GRU L3C4', 'Grid-GRU L2C4', 'Grid-GRU L2C8', 'Grid-GRU L1C8', 'GRU L1', 'GRU L2', 'GRU L3', 'Grid-LRU L1C8', 'Grid-LRU L2C8', 'Grid-LRU L3C4', 'HGRN2', 'mLSTM', 'DeltaNet', 'RIMs', 'BRIMs']
    baseline_rows.sort(key=lambda r: order.index(r['model']))
    for row in baseline_rows:
        key = MID_KEYS[row['model']]
        params, state, hidden = accounting(runs_by_id[row['ids'][0]], key)
        values = []
        seeds = []
        for key_id in row['ids']:
            run = runs_by_id[key_id]
            assert float(run['parameters']['eval|val_size']) == 5e6
            assert float(run['parameters']['eval|test_size']) == 5e6
            seeds.append(int(run['parameters']['seed']))
            values.append(trace(run, 'val/Loss')[row['step']] / math.log(2))
        score = stats(values)
        assert abs(score['mean'] - row['mean_bpc']) < 1e-10
        assert set(seeds) == ({42, 1337} if len(seeds) == 2 else {42, 1337, 31337})
        evidence['mid'].append(row | {'params': params, 'state_scalars': state, 'hidden': hidden, 'seeds': seeds})
    rows = [f"{r['model'].replace('Grid-GRU', 'MoSAIC')} & {r['params']/1e6:.3f}M & {r['n']} & ${r['mean_bpc']:.4f}\\pm{r['sample_sd']:.4f}$" for r in evidence['mid']]
    separate = []
    for label, model, prefixes, step in [
        ('Grid-LRU L2C4', 'grnn_lru_L2C4', ['6c724575', '55de969d', '5c0724b5'], 960_036_864),
        ('Transformer-256', 'transformer', ['ad72c9a0', '37180ab2'], 982_646_784),
    ]:
        group = [next(r for key, r in runs_by_id.items() if key.startswith(prefix)) for prefix in prefixes]
        seeds = [int(r['parameters']['seed']) for r in group]
        assert set(seeds) == ({42, 1337} if len(group) == 2 else {42, 1337, 31337})
        assert all(float(r['parameters']['eval|val_size']) == 5e6 and float(r['parameters']['eval|test_size']) == 5e6 for r in group)
        score = stats([trace(r, 'val/Loss')[step] / math.log(2) for r in group])
        params, state, hidden = accounting(group[0], model)
        separate.append(score | {'model': label, 'step': step, 'params': params, 'state_scalars': state, 'hidden': hidden, 'seeds': seeds, 'ids': [r['metadata']['experimentKey'] for r in group]})
    evidence['mid_separate_horizons'] = separate
    rows += [r'\midrule', r'\multicolumn{4}{l}{Separate logged horizons:}']
    for row, marker in zip(separate, [r'$^{\dagger}$', r'$^{\ddagger}$']):
        rows.append(f"{row['model']}{marker} & {row['params']/1e6:.3f}M & {row['n']} & {pm(row)}")
    table('aistats_mid_main.tex',
        r'Text8 $\sim$1M series: mean $\pm$ sample SD of validation BPC. Upper panel: exactly 980,037,632 logged tokens. Lower panel uses separate horizons: $\dagger$ 960,036,864; $\ddagger$ 982,646,784, without interpolation. Seeds 42/1337, adding 31337 for $n=3$. Split 90M/5M/5M. Local baselines share the training recipe; state and compute are unmatched.',
        'tab:mid', 'lrrr', r'Model & Params & $n$ & BPC $\downarrow$', rows)
    rows = [f"{r['model'].replace('Grid-GRU', 'MoSAIC')} & {r['hidden']} & {r['params']:,} & {r['state_scalars']:,} & {4*r['state_scalars']/1024:.2f}" for r in evidence['mid']]
    lru_l2 = next(r for r in frontier['runs'] if r['metadata']['experimentKey'].startswith('6c724575'))
    p, s, h = accounting(lru_l2, 'grnn_lru_L2C4')
    rows += [f'Grid-LRU L2C4 & {h} & {p:,} & {s:,} & {4*s/1024:.2f}']
    transformer = next(r for r in frontier['runs'] if r['metadata']['experimentKey'].startswith('ad72c9a0'))
    tp, ts, th = accounting(transformer, 'transformer')
    rows += [f'Transformer-256 & {th} & {tp:,} & {ts:,} & {4*ts/1024:.2f}']
    table('aistats_mid_state.tex',
        r'New Text8 model accounting from the retained widths and local implementations, including token adapters ($V=27$). State counts include necessary recurrent and public-message caches, or Transformer key/value caches; redundant views and bookkeeping are excluded. Float32 KiB exclude training intermediates and optimizer state.',
        'tab:midstate', 'lrrrr', r'Model & Width & Params & State scalars & KiB', rows)
    evidence['mid_grid_lru_L2_accounting'] = {'params': p, 'state_scalars': s, 'hidden': h}
    evidence['mid_transformer_accounting'] = {'params': tp, 'state_scalars': ts, 'hidden': th}

    names = {'full': 'Full', 'none': 'All three off', 'no_noise': 'No logit noise', 'no_comm': 'No communication cost', 'no_entropy': 'No entropy bonus'}
    rows = [f"{names[r['variant']]} & {r['bpc']:.4f} & ${r['delta']:+.4f}$" for r in analysis['ablation']['rows']]
    table('aistats_ablation_main.tex',
        r'MoSAIC-L2C4 regularizer pilot, seed 42: validation BPC at 50,331,648 tokens. Same widths, optimizer and 524,288-token validation cap; $\Delta$ is relative to the full model. One launch per condition; no variability estimate.',
        'tab:ablation', 'lrr', r'Condition & BPC $\downarrow$ & $\Delta$', rows)
    rows = [f"{r['variant'].replace('_', ' ')} & {r['bpc']:.4f} & ${r['delta']:+.4f}$ & ${r['late_800M_mean_delta']:+.4f}$ & {r['window1024_bpc']:.4f} & ${r['window1024_bpc']-r['bpc']:+.4f}$" for r in analysis['architecture']['rows']]
    table('aistats_lru_variants.tex',
        r'Five LRU mechanisms and their own dynamic-hub control on Text8, seed 42, at 980,037,632 tokens. Late $\Delta$ averages paired differences at the common 800.03--980.04M logs. Reset-1024 evaluates the same validation data with state reset every 1,024 valid tokens; reset penalty is its BPC minus continuous BPC. Core widths are L2C4H180; added projections or replacement of an LRU by matrix memory change parameter/state counts. This is a single-seed exploratory comparison.',
        'tab:lruvariants', 'lrrrrr', r'Variant & BPC $\downarrow$ & $\Delta$ & Late $\Delta$ & Reset-1024 & Reset penalty', rows)
    rows = [f"{r['variant'].replace('_', ' ')} & {r['step']/1e6:.2f} & {r['bpc']:.4f} & ${r['delta']:+.4f}$" for r in analysis['sparse_pairs']]
    table('aistats_sparse_lru.tex',
        r'Sparse LRU pilots, seed 42. Each row uses its latest common logged point with the separate dense control; horizons differ across rows and are not cross-row rankings. $\Delta$ is paired BPC difference. Block has a 0.531M core versus approximately 1.05M for its dense control.',
        'tab:sparselru', 'lrrr', r'Variant & Tokens (M) & BPC $\downarrow$ & $\Delta$', rows)
    rows = [f"{r['model'].replace('_', '-')} & {r['r64_bpc']:.4f} & {r['r512_bpc']:.4f} & ${r['delta']:+.4f}$" for r in analysis['rollout']]
    table('aistats_rollout.tex',
        r'Rollout pilots at 50,331,648 tokens, seed 42; BPC with the same capped validation. Length 64 uses 512 training streams, length 512 uses 64: both have 32,768 tokens/update. Longer BPTT changes the number of independent streams and does not isolate context length alone.',
        'tab:rollout', 'lrrr', r'Model & Rollout 64 & Rollout 512 & $\Delta$', rows)
    rows = [f"{r['variant'].replace('_', ' ')} & {100*r['val_acc']:.3f} & {r['val_loss']:.3f} & " + ' & '.join(f'{100*x:.3f}' for x in r['bin_acc']) for r in analysis['mqar']['rows']]
    table('aistats_mqar.tex',
        r'MQAR pilots at 173,598,720 logged training tokens, seed 42: query-weighted validation accuracy (\%) and cross-entropy (nats), followed by bin accuracies (\%). All sequence-level exact-match accuracies are zero. Each bin contains 1,000 sequences; no strong-baseline or repeated-seed comparison is available.',
        'tab:mqar', 'lrrrrrrr', r'Variant & Overall & CE & 64/4 & 128/8 & 256/16 & 256/32 & 256/64', rows)
    rows = []
    for task, group in analysis['rl'].items():
        for r in group['rows']:
            rows.append(f"{task} & {r['model'].replace('_', '-')} & {r['last4_common_mean']:.4f}")
    table('aistats_rl.tex',
        r'POPGym pilots, seed 42: mean evaluation return over the same four checkpoints at 6.291456, 6.815744, 7.340032, and 7.864320M vector slots. Each evaluation uses 32 stochastic-policy episodes with fresh environment seed 50000. Checkpoints share one training trajectory; their mean is not a seed-level uncertainty estimate.',
        'tab:rl', 'llr', r'Environment & Model & Late return $\uparrow$', rows)
    evidence['pilots'] = {key: analysis[key] for key in ['ablation', 'architecture', 'sparse_pairs', 'rollout', 'mqar', 'rl', 'depth_matched', 'transformer_separate_horizon']}

    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    for axis, task, metric, keys in [
        (axes[0], 'text8', 'val/BPC', ['rnn_L2', 'rnn_L3', 'grnn_L2C4', 'grnn_L3C4', 'transformer']),
        (axes[1], 'SDQ', 'Acc++', ['rnn_L1', 'rnn_L2', 'grnn_L2C4', 'grnn_L3C4']),
    ]:
        for key in keys:
            traces = [trace(r, metric) for r in selected if r['task'] == task and r['config'].endswith(' / ' + key)]
            steps = [s for s in common(traces) if s <= evidence['historical'][task]['horizon']]
            values = np.asarray([[t[s] for s in steps] for t in traces])
            x, mean, sd = np.asarray(steps) / 1e9, values.mean(0), values.std(0, ddof=1)
            axis.plot(x, mean, label=LABELS[key], lw=1.3)
            axis.fill_between(x, mean-sd, mean+sd, alpha=.13)
        axis.set(xlabel='Logged training tokens (billions)', ylabel='Validation BPC' if task == 'text8' else 'Online Acc++', title='(a) Text8, ~10M' if task == 'text8' else '(b) SDQ, ~10M')
        axis.set_ylim((1.40, 1.86) if task == 'text8' else (.1, 1))
        axis.legend(fontsize=6.8, frameon=False)
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(PAPER / 'fig_aistats_historical_curves.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    for label in ['Mamba', 'Grid-GRU L2C4', 'Grid-GRU L3C4', 'GRU L2', 'Grid-LRU L3C4']:
        row = next(r for r in evidence['mid'] if r['model'] == label)
        traces = [trace(runs_by_id[k], 'val/Loss') for k in row['ids']]
        steps = [s for s in common(traces) if s <= row['step']]
        values = np.asarray([[t[s]/math.log(2) for s in steps] for t in traces])
        x, mean, sd = np.asarray(steps)/1e9, values.mean(0), values.std(0, ddof=1)
        axes[0].plot(x, mean, label=label.replace('Grid-GRU', 'MoSAIC'), lw=1.3)
        axes[0].fill_between(x, mean-sd, mean+sd, alpha=.13)
    for r in analysis['mqar']['rows']:
        axes[1].plot(range(5), np.asarray(r['bin_acc'])*100, marker='o', ms=2.5, label=r['variant'].replace('_', ' '))
    axes[0].set(xlabel='Logged training tokens (billions)', ylabel='Validation BPC', title='(a) Text8, ~1M', ylim=(1.52, 2.03))
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
