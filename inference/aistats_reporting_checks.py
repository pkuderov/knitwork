"""Report existing Text8 test scores and uncertainty without training or network access."""

import csv
import json
import math
from pathlib import Path
import statistics

from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/experiments'
PAPER = ROOT / 'article/latex'


def quality_cell(value, candidates, *, higher=False, decimals=4, sd=None):
    """Highlight the two best distinct observed scores before rounding."""
    if value is None:
        return '---'
    values = sorted({x for x in candidates if x is not None}, reverse=higher)
    formatted = f'{value:.{decimals}f}'
    if sd is not None:
        formatted += rf'\pm{sd:.{decimals}f}'
    rank = values.index(value)
    if rank == 0:
        formatted = rf'\mathbf{{{formatted}}}'
    elif rank == 1:
        formatted = rf'\underline{{{formatted}}}'
    return '$' + formatted + '$'


def reporting_checks(evidence, runs):
    source = DATA / 'recent_metrics_2026-10-05.csv'
    with source.open(newline='') as stream:
        exported = list(csv.DictReader(stream))
    tests = []
    for group in evidence['mid']:
        samples = []
        for run_id in group['ids']:
            run = runs[run_id]
            name = run['parameters']['name']
            matched = [row for row in exported if row['experiment'] == name or row['experiment'].endswith(' ' + name)]
            if len(matched) != 1:
                raise ValueError('Expected one test-export row for ' + name)
            row = matched[0]
            value = None if row['test/BPC'] == '-' else float(row['test/BPC'])
            names = run['test_presence_audit']['test_metric_names_present']
            assert (value is not None) == ('test/BPC' in names)
            assert run['parameters']['eval|window'] == '1024'
            if value is not None:
                assert math.isfinite(value)
            samples.append({
                'id': run_id, 'name': name, 'seed': int(run['parameters']['seed']),
                'test_bpc': value, 'csv_experiment': row['experiment'],
            })
        values = [sample['test_bpc'] for sample in samples if sample['test_bpc'] is not None]
        tests.append({
            'model': group['model'], 'validation_runs': group['n'],
            'test_runs': len(values), 'mean_bpc': statistics.mean(values) if values else None,
            'sample_sd': statistics.stdev(values) if len(values) > 1 else None,
            'samples': samples,
        })

    large = evidence['historical']['text8']['groups']
    sdq = evidence['historical']['SDQ']['groups']
    mid = {group['model']: group for group in evidence['mid']}
    comparisons = []
    for scale, task, model, reference in [
        ('10M', 'Text8', 'grnn_L2C4', 'rnn_L3'), ('10M', 'Text8', 'grnn_L3C4', 'rnn_L3'),
        ('10M', 'SDQ', 'grnn_L2C4', 'rnn_L1'), ('10M', 'SDQ', 'grnn_L3C4', 'rnn_L1'),
        ('10M', 'Text8', 'grnn_L2C4', 'transformer'), ('10M', 'Text8', 'grnn_L3C4', 'transformer'),
        ('10M', 'Text8', 'grnn_L2C4', 'grnn_L2C8'), ('10M', 'Text8', 'grnn_L3C4', 'grnn_L2C4'),
        ('1M', 'Text8', 'Grid-GRU L2C4', 'GRU L3'), ('1M', 'Text8', 'Grid-GRU L3C4', 'GRU L3'),
    ]:
        groups = sdq if task == 'SDQ' else large
        left = [run['value'] for run in groups[model]['runs']] if scale == '10M' else mid[model]['values']
        right = [run['value'] for run in groups[reference]['runs']] if scale == '10M' else mid[reference]['values']
        qa = statistics.variance(left) / len(left)
        qb = statistics.variance(right) / len(right)
        df = (qa + qb)**2 / (qa**2 / (len(left) - 1) + qb**2 / (len(right) - 1))
        delta = statistics.mean(left) - statistics.mean(right)
        half_width = float(stats.t.ppf(0.975, df)) * math.sqrt(qa + qb)
        result = stats.ttest_ind(left, right, equal_var=False)
        comparisons.append({
            'scale': scale, 'task': task, 'model': model, 'reference': reference,
            'n_model': len(left), 'n_reference': len(right),
            'model_values': left, 'reference_values': right,
            'delta': delta, 'welch_df': df, 'p_two_sided': float(result.pvalue),
            'nominal_ci95': [delta - half_width, delta + half_width],
        })
    return {
        'test_source': str(source.relative_to(ROOT)), 'test_source_decimal_places': 6,
        'test_values_reported': True, 'test_values_used_for_selection': False,
        'test_checkpoint_rule': 'Minimum reset-1024 validation loss; reported test score uses continuous state.',
        'test_split_overlaps_10M_development_validation': True,
        'test_groups': tests, 'exploratory_welch_comparisons': comparisons,
        'welch_assumptions': 'Independent approximately normal run-level scores; no multiplicity correction; 10M seed independence is unverified.',
        'segment_bootstrap_performed': False,
        'segment_bootstrap_missing_inputs': 'Aligned per-segment losses for the compared final checkpoints are not retained locally.',
        'new_training_runs': 0, 'external_tracking_requests': 0,
    }


def write_reporting_checks(evidence, runs):
    report = reporting_checks(evidence, runs)
    test_rows = []
    candidates = [group['mean_bpc'] for group in report['test_groups']]
    for group in report['test_groups']:
        mean = group['mean_bpc']
        sd = group['sample_sd']
        value = quality_cell(mean, candidates, sd=sd)
        name = group['model'].replace('Grid-GRU', 'MoSAIC')
        test_rows.append(f"{name} & {group['test_runs']} & {value}" + r' \\')
    source = '\n'.join([
        '% Generated offline by inference/aistats_reporting_checks.py.',
        r'\begin{table}[H]', r'\centering',
        r'\caption{Text8 test BPC for models with approximately one million parameters. Values summarize available validation-selected checkpoints; multiple runs report means and sample SD. Missing evaluations are explicit. Best available mean is bold; second best is underlined.}',
        r'\label{tab:midtest}', r'\begin{tabular}{lrr}', r'\toprule',
        r'Model & Test runs & BPC \\', r'\midrule', *test_rows,
        r'\bottomrule', r'\end{tabular}', r'\end{table}', '',
    ])
    (PAPER / 'aistats_mid_test.tex').write_text(source)
    names = {
        'grnn_L2C4': 'MoSAIC-L2C4', 'grnn_L3C4': 'MoSAIC-L3C4', 'grnn_L2C8': 'MoSAIC-L2C8',
        'rnn_L1': 'GRU-L1', 'rnn_L3': 'GRU-L3', 'transformer': 'Transformer-256',
        'Grid-GRU L2C4': 'MoSAIC-L2C4', 'Grid-GRU L3C4': 'MoSAIC-L3C4', 'GRU L3': 'GRU-L3',
    }
    rows = []
    previous = None
    for comparison in report['exploratory_welch_comparisons']:
        block = (comparison['scale'], comparison['reference'] in ('transformer', 'grnn_L2C8', 'grnn_L2C4'))
        if previous is not None and block != previous:
            rows.append(r'\midrule')
        previous = block
        lower, upper = comparison['nominal_ci95']
        p = comparison['p_two_sided']
        p_text = '<0.001' if p < 0.001 else f'{p:.3f}'
        rows.append(
            f"{comparison['scale']} & {comparison['task']} & {names[comparison['model']]} & {names[comparison['reference']]} & "
            f"{comparison['delta']:+.4f} & {comparison['welch_df']:.2f} & ${p_text}$ & [{lower:+.4f}, {upper:+.4f}]" + r' \\'
        )
    source = '\n'.join([
        '% Generated offline by inference/aistats_reporting_checks.py.',
        r'\begin{table}[H]', r'\centering', r'\small',
        r'\caption{Exploratory Welch comparisons of final scores. Differences are model minus reference (Text8 validation BPC, lower is better; SDQ Acc++, higher is better). Intervals are nominal 95 percent intervals under independent-run assumptions, with no multiplicity adjustment; 10M seed independence is unverified.}',
        r'\label{tab:welch}', r'\begin{tabular}{llllrrrr}', r'\toprule',
        r'Scale & Task & Model & Reference & Difference & df & Two-sided $p$ & 95\% interval \\',
        r'\midrule', *rows, r'\bottomrule', r'\end{tabular}', r'\end{table}', '',
    ])
    (PAPER / 'aistats_uncertainty_checks.tex').write_text(source)
    (DATA / 'aistats2027_reporting_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    evidence = json.loads((DATA / 'aistats2027_revision_evidence.json').read_text())
    snapshot = json.loads((DATA / 'weekly_2026-09-28_2026-10-04_snapshot.json').read_text())
    runs = {run['metadata']['experimentKey']: run for run in snapshot['runs']}
    report = write_reporting_checks(evidence, runs)
    print('Test scores:', sum(group['test_runs'] for group in report['test_groups']))
    print('Exploratory Welch comparisons:', len(report['exploratory_welch_comparisons']))
    print('No training or network access.')


if __name__ == '__main__':
    main()
