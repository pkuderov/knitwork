"""Read-only resource accounting for the fixed historical paper cohort.

Fetch only run IDs recorded in results_aaai.md. Save a sanitized resource snapshot;
never save system environment variables, host names, or authentication material.
Run with --fetch to retrieve logs; without it, regenerate tables from the snapshot.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import threading


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'docs/experiments/aistats2027_resources.json'
REPORT = ROOT / 'docs/experiments/aistats2027_resources.md'
THREAD = threading.local()


def cohort():
    sections = {
        'Text8: completed-seed results at the comparable horizon': 'text8',
        'Text8: reduced-token, increased-update baselines': 'text8',
        'Store--Distract--Query: completed-replicate final-window results': 'SDQ',
        'Store--Distract--Query: reduced-budget baseline results': 'SDQ',
    }
    rows = []
    task = None
    for line in (ROOT / 'docs/experiments/results_aaai.md').read_text().splitlines():
        if line.startswith('## '):
            task = sections.get(line[3:])
        if task and line.startswith('| ') and re.search(r'`[a-f0-9]{8}`', line):
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            for prefix in re.findall(r'`([a-f0-9]{8})`', line):
                rows.append(dict(task=task, config=cells[0], run_prefix=prefix))
    assert len({r['run_prefix'] for r in rows}) == len(rows)
    return rows


def read_api(endpoint, params):
    import requests

    if not hasattr(THREAD, 'session'):
        THREAD.session = requests.Session()
        THREAD.session.headers['Authorization'] = os.environ['COMET_API_KEY']
    for attempt in range(2):
        try:
            response = THREAD.session.get(
                'https://www.comet.com/api/rest/v2/' + endpoint,
                params=params, timeout=(10, 30),
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt:
                raise


def numeric(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def integrate_fps(data):
    points = {}
    for item in data.get('metrics', data.get('values', [])):
        step = numeric(item.get('step'))
        rate = numeric(item.get('metricValue', item.get('value')))
        if step is not None and step > 0 and rate is not None and rate > 0:
            # The API returns chronological points; the last duplicate is retained.
            points[step] = rate
    assert points, 'No usable fps points'
    previous = 0
    seconds = 0
    tokens = 0
    for step, rate in sorted(points.items()):
        # Score comparisons concern only the prefix through the 1B target.
        if step > 1e9:
            break
        seconds += (step - previous) / rate
        previous = tokens = step
    assert seconds > 0
    return dict(
        accounted_tokens=tokens,
        loop_hours=seconds / 3600,
        effective_tokens_per_second=tokens / seconds,
        final_logged_tokens=max(points),
        fps_points=len(points),
    )


def fetch_run(row, metadata):
    run = dict(row)
    try:
        prefix = row['run_prefix']
        matches = [m for m in metadata if m['experimentKey'].startswith(prefix)]
        assert len(matches) == 1, 'Cohort ID missing or ambiguous'
        meta = matches[0]
        key = meta['experimentKey']
        params = {'experimentKey': key}
        values = read_api('experiment/parameters', params)['values']
        values = {v['name']: v.get('valueCurrent') for v in values}
        system = read_api('experiment/system-details', params)
        fps = read_api('experiment/metrics/get-metric', params | {'metricName': 'perf/fps'})
        run.update(integrate_fps(fps))
        for name in ['n_envs', 'rollout_len', 'n_steps']:
            run[name] = numeric(values.get(name))
        run['gpu_names'] = sorted({g['name'] for g in system.get('gpuStaticInfoList', []) if g.get('name')})
        duration = numeric(meta.get('durationMillis'))
        run['experiment_elapsed_hours'] = duration / 3.6e6 if duration is not None else None
        run['metadata_duration_inconsistent'] = (
            run['experiment_elapsed_hours'] is not None
            and run['loop_hours'] > 1.05 * run['experiment_elapsed_hours']
        )
        run['tokens_per_update'] = run['n_envs'] * run['rollout_len']
        run['accounted_updates'] = math.floor(run['accounted_tokens'] / run['tokens_per_update'])
        run['seed_recorded'] = str(values.get('seed')).lower() not in ['none', 'null', '']
        # Safe numeric provenance for rechecking the time integral offline.
        run['fps_trace'] = [
            {'step': numeric(x.get('step')), 'rate': numeric(x.get('metricValue', x.get('value')))}
            for x in fps.get('metrics', fps.get('values', []))
        ]
        print('Collected', row['task'], row['config'], prefix, flush=True)
    except Exception as error:
        run['error'] = type(error).__name__
        print('Unavailable', row['task'], prefix, type(error).__name__, flush=True)
    return run


def summarize(data):
    groups = {}
    for run in data['runs']:
        groups.setdefault((run['task'], run['config']), []).append(run)
    summaries = []
    for (task, config), runs in groups.items():
        valid = [r for r in runs if 'error' not in r]
        result = dict(task=task, config=config, n=len(runs), available=len(valid))
        for field in ['accounted_tokens', 'loop_hours', 'effective_tokens_per_second', 'experiment_elapsed_hours', 'accounted_updates']:
            numbers = [r[field] for r in valid if r.get(field) is not None]
            result[field] = dict(
                mean=statistics.mean(numbers) if numbers else None,
                sd=statistics.stdev(numbers) if len(numbers) > 1 else 0,
                min=min(numbers) if numbers else None,
                max=max(numbers) if numbers else None,
            )
        result['gpu_names'] = sorted({name for r in valid for name in r['gpu_names']})
        result['tokens_per_update'] = sorted({r['tokens_per_update'] for r in valid})
        summaries.append(result)
    return summaries


def write_report(data):
    groups = summarize(data)
    lines = [
        '# AISTATS 2027: historical resource accounting', '',
        f"Retrieved read-only at {data['retrieved_utc']}. The cohort is fixed by run IDs in `results_aaai.md`; this does not update its quality metrics or recruit newer runs.", '',
        'For each run, the loop time is Σ Δtokens / perf/fps through the last positive logged point at or below 1B tokens. This follows `Logger.flush` and `Timer.fps`: fps is the token increment divided by elapsed loop time since the preceding flush. It includes compilation, training, validation, inspection, logging overhead between flushes, and any resource contention. It is elapsed wall time, not exclusive GPU-hours or FLOPs. Finalization after the last flush is excluded. Endpoints are not extrapolated.', '',
        'Effective rate is accounted tokens divided by reconstructed loop time, not the arithmetic mean of logged rates. Updates are floor(tokens / (n_envs × rollout_len)); they count reached update boundaries, not confirmed successful optimizer steps. Experiment elapsed time is metadata durationMillis and covers the full recorded run, which can exceed the scored 1B prefix. GPU names are inventory labels; no contention control or peak-memory measurement is implied.', '',
        'Aggregates are mean ± sample standard deviation across available launches. Tokens and hours are per launch, not sums. Mixed-device group aggregates are inventory summaries, not performance comparisons. Two runs have metadata durations shorter than their integrated fps time; those durations are inconsistent and are not used as paper cost estimates.', '',
        '| Task | Config | Available / cohort n | Accounted tokens (M) | Update boundaries (k) | Loop elapsed (h) | Effective rate (k tokens/s) | Full experiment elapsed (h) | GPU |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    for group in groups:
        columns = [group['task'], group['config'], f"{group['available']}/{group['n']}"]
        for field,scale,digits in [('accounted_tokens',1e6,1),('accounted_updates',1e3,2),('loop_hours',1,3),('effective_tokens_per_second',1e3,1),('experiment_elapsed_hours',1,3)]:
            metric = group[field]
            columns.append('—' if metric['mean'] is None else f"{metric['mean']/scale:.{digits}f} ± {metric['sd']/scale:.{digits}f}")
        columns.append(', '.join(group['gpu_names']) or 'unverified')
        lines.append('| '+' | '.join(columns)+' |')
    lines += ['', '## Per-run provenance', '', '| Task | Config | Run prefix | Accounted tokens (M) | Loop elapsed (h) | Effective rate (k tokens/s) | GPU | Duration audit | Status |', '| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |']
    for run in data['runs']:
        columns = [run['task'], run['config'], '`'+run['run_prefix']+'`']
        columns += [f"{run.get('accounted_tokens',0)/1e6:.3f}", f"{run.get('loop_hours',0):.4f}", f"{run.get('effective_tokens_per_second',0)/1e3:.2f}"]
        columns.append(', '.join(run.get('gpu_names', [])) or 'unverified')
        columns.append('inconsistent' if run.get('metadata_duration_inconsistent') else 'no shorter-duration anomaly')
        columns.append(run.get('error', 'collected'))
        lines.append('| '+' | '.join(columns)+' |')
    REPORT.write_text('\n'.join(lines)+'\n')
    data['groups'] = groups
    SNAPSHOT.write_text(json.dumps(data, indent=2)+'\n')
    print('Wrote', SNAPSHOT.relative_to(ROOT), 'and', REPORT.relative_to(ROOT))


def model_label(config):
    model, name = config.split(' / ')
    if model == 'rnn':
        return 'GRU-' + name.removeprefix('rnn_')
    if model == 'grnn':
        return 'MoSAIC-' + name.removeprefix('grnn_')
    if model == 'transformer':
        return 'Transformer-' + ('64' if name.endswith('_64') else '256')
    return {'hgrn2': 'HGRN2', 'delta_net': 'DeltaNet', 'mlstm': 'mLSTM'}[model]


def latex_stat(runs, field, scale, digits):
    values = [run[field] / scale for run in runs]
    mean = statistics.mean(values)
    if len(values) == 1:
        return f'${mean:.{digits}f}$'
    sd = statistics.stdev(values)
    return f'${mean:.{digits}f}\\pm{sd:.{digits}f}$'


def write_latex(data):
    assert len(data['runs']) == 65 and all('error' not in r for r in data['runs']), 'Resolve incomplete cohort before producing paper tables'
    gpu_labels = {
        'NVIDIA H100 80GB HBM3': 'H100',
        'NVIDIA TITAN RTX': 'TITAN',
        'NVIDIA GeForce RTX 3080 Ti': '3080Ti',
        'Tesla V100-SXM3-32GB': 'V100',
    }
    groups = {}
    for run in data['runs']:
        if 'error' not in run:
            gpu = tuple(run['gpu_names'])
            assert len(gpu) == 1, 'Resolve multi-device inventories before reporting'
            groups.setdefault((run['task'], run['config'], gpu[0]), []).append(run)
    core_configs = {'rnn / rnn_L2', 'grnn / grnn_L2C4', 'grnn / grnn_L3C4', 'transformer / transformer'}
    main = [
        '% Generated offline by inference/aistats_resource_accounting.py.',
        r'\begin{table*}[t]', r'\centering',
        r'\caption{Historical log-derived costs for the H100 subset of the main references. Tokens, elapsed loop time, and effective rate are per-launch means $\pm$ sample standard deviations. $\dagger$: one launch, with no variability estimate. This hardware subset has fewer launches than the quality tables; workload contention was not controlled. Time includes training-loop overhead and is not exclusive accelerator usage.}',
        r'\label{tab:resources}',
        r'\begin{tabular}{llrrrr}', r'\toprule',
        r'Task & Model & $n$ & Logged tokens (M) & Loop time (h) & Rate (k tokens/s) \\', r'\midrule',
    ]
    for (task, config, gpu), runs in sorted(groups.items()):
        if config not in core_configs or gpu != 'NVIDIA H100 80GB HBM3':
            continue
        label = model_label(config) + (r'$\dagger$' if len(runs) == 1 else '')
        cells = [task, label, str(len(runs))]
        for field, scale, digits in [('accounted_tokens', 1e6, 1), ('loop_hours', 1, 3), ('effective_tokens_per_second', 1e3, 1)]:
            cells.append(latex_stat(runs, field, scale, digits))
        main.append(' & '.join(cells) + r' \\')
    main += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
    appendix = [
        '% Generated offline by inference/aistats_resource_accounting.py.',
        r'\section{Historical Resource Accounting}\label{app:resources}', '',
        r'This accounting covers the same 65 historical launches used in the main and reduced-token quality summaries. It was retrieved from their recorded run identifiers without adding new experiments. The hardware inventory is heterogeneous: 48 launches report H100 80GB HBM3, eight TITAN RTX, five RTX 3080 Ti, and four V100-SXM3-32GB. We therefore stratify rates and elapsed-time proxies by GPU model rather than pooling them into architecture comparisons.', '',
        r'For a run with processed-token counter $s_i$ and interval rate $f_i=\texttt{perf/fps}_i$, we reconstruct $T_{\mathrm{loop}}=\sum_i(s_i-s_{i-1})/f_i$, with $s_0=0$, through the last logged point at or below 1B. The effective rate is $s_{\mathrm{last}}/T_{\mathrm{loop}}$. The logger divides each token increment by elapsed time since its preceding flush. This includes compilation, training, validation, inspection, logging overhead between flushes, and any workload contention; it excludes finalization after the last flush. We do not extrapolate missing endpoints. These are log-derived elapsed-time proxies rather than dedicated throughput benchmarks or exclusive GPU-hours.', '',
        r'Update counts are $\lfloor s_{\mathrm{last}}/(n_{\mathrm{envs}}\,\ell_{\mathrm{TBPTT}})\rfloor$, counting reached update boundaries; a nonfinite-gradient skip need not perform an optimizer step. The full-run duration in tracker metadata is inconsistent with the fps-derived time in two historical launches (one GRU-L3 SDQ run and one Transformer-256 text8 run), so it is not used as the cost estimator. Tables below report means $\pm$ sample standard deviations per launch; single-launch rows have no uncertainty estimate. Their launch counts describe hardware strata, not additional experiments. H100 denotes H100 80GB HBM3, TITAN denotes TITAN RTX, 3080Ti denotes RTX 3080 Ti, and V100 denotes V100-SXM3-32GB.', '',
    ]
    for task in ['SDQ', 'text8']:
        for reduced in [False, True]:
            kind = 'Reduced-token context' if reduced else 'Main cohort'
            appendix += [r'\begin{table}[ht]', r'\centering', r'\caption{' + task + ': ' + kind + r' resources, stratified by GPU model. Reduced-token rows have different batch and update geometries.}', r'\begin{tabular}{llrrrrr}', r'\toprule', r'Model & GPU & $n$ & Tokens (M) & Updates (k) & Loop time (h) & Rate (k tokens/s) \\', r'\midrule']
            for (row_task, config, gpu), runs in sorted(groups.items()):
                is_reduced = config.split(' / ')[0] in ['hgrn2', 'delta_net', 'mlstm']
                if row_task != task or is_reduced != reduced:
                    continue
                cells = [model_label(config), gpu_labels[gpu], str(len(runs))]
                for field, scale, digits in [('accounted_tokens', 1e6, 1), ('accounted_updates', 1e3, 2), ('loop_hours', 1, 3), ('effective_tokens_per_second', 1e3, 1)]:
                    cells.append(latex_stat(runs, field, scale, digits))
                appendix.append(' & '.join(cells) + r' \\')
            appendix += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    (ROOT / 'article/latex/aistats_resources_main.tex').write_text('\n'.join(main))
    (ROOT / 'article/latex/aistats_resources_appendix.tex').write_text('\n'.join(appendix))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    if args.fetch:
        from dotenv import load_dotenv

        load_dotenv(ROOT / '.env')
        assert os.environ.get('COMET_API_KEY'), 'Comet API key unavailable'
        rows = cohort()
        metadata = []
        for task, project in [('text8','knitwork-text'),('SDQ','knitwork-sdq')]:
            cached = Path('/tmp/knitwork-comet-text-metadata.json')
            if task == 'text8' and cached.exists():
                project_metadata = json.loads(cached.read_text())['experiments']
            else:
                project_metadata = read_api('experiments', {'workspaceName':'team-rl-exp','projectName':project,'archived':'false','size':1000})['experiments']
            metadata.extend(project_metadata)
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(fetch_run, row, metadata) for row in rows]
            for future in as_completed(futures):
                results.append(future.result())
        data = dict(retrieved_utc=datetime.now(timezone.utc).isoformat(), source_cohort='docs/experiments/results_aaai.md', runs=sorted(results,key=lambda r:(r['task'],r['config'],r['run_prefix'])))
    else:
        data = json.loads(SNAPSHOT.read_text())
        for run in data['runs']:
            if 'error' not in run:
                run['metadata_duration_inconsistent'] = run['loop_hours'] > 1.05 * run['experiment_elapsed_hours']
    write_report(data)
    write_latex(data)


if __name__ == '__main__':
    main()
