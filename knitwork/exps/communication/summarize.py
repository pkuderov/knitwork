"""Collect complete local runs and report prespecified, seed-matched contrasts."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
import yaml

from knitwork.exps.communication.protocol import CONDITIONS, config_fingerprint


CONTRASTS = (
    ('dynamic_column0', 'static_column0', 'Content-dependent routing with column-0 readout'),
    ('dynamic_mean', 'static_mean', 'Content-dependent routing with all-column readout'),
    ('dynamic_mean', 'none_mean', 'Inter-column exchange with all state accessible'),
    ('none_mean', 'none_column0', 'Readout access to independent column states'),
)


def collect(root):
    groups = {}
    for path in sorted(Path(root).rglob('summary.json')):
        summary = json.loads(path.read_text())
        if not summary['completed']:
            continue
        cfg = yaml.safe_load((path.parent / 'config.yaml').read_text())
        metadata = json.loads((path.parent / 'metadata.json').read_text())
        if metadata['config_fingerprint'] != config_fingerprint(cfg):
            raise ValueError(f'Config provenance mismatch: {path}')
        key = (cfg['exp'], cfg['topology'], cfg['recipe'])
        label = (cfg['seed'], cfg['data_seed'])
        conditions = groups.setdefault(key, {}).setdefault(label, {})
        if cfg['condition'] in conditions:
            raise ValueError(f'Duplicate seed/condition: {path}')
        conditions[cfg['condition']] = (summary, cfg, metadata)
    return groups


def protocol_signature(cfg, metadata):
    core = cfg[cfg['model'].replace('.', '_')].copy()
    core.pop('routing')
    core.pop('readout')
    return json.dumps(
        {
            'core': core, 'data_identity': metadata['data_identity'], 'source_hashes': metadata['source_hashes'],
            'initialization': metadata['initialization_sha256'],
            'protocol': {key: cfg.get(key) for key in (
                'lr', 'reset_prob', 'generator', 'eval_generator', 'curriculum', 'eval',
                'communication', 'n_envs', 'rollout_len', 'effective_tokens', 'compile', 'deterministic',
            )},
        }, sort_keys=True,
    )


def summarize(root):
    reports = []
    for (task, topology, recipe), seeds in collect(root).items():
        for _, conditions in seeds.items():
            signatures = {protocol_signature(cfg, meta) for _, cfg, meta in conditions.values()}
            if len(signatures) != 1:
                raise ValueError(f'Unmatched conditions in {task}/{topology}/{recipe}')
        metrics = [('validation_BPC', lambda row: row['final_validation']['BPC'])] if task == 'text8' else [
            ('heldout_query_Acc', lambda row: row['final_validation']['Acc']),
            ('fixed_gap_macro_Acc', lambda row: row['fixed_gap_evaluation']['macro_Acc']),
        ]
        group = {
            'task': task, 'topology': topology, 'recipe': recipe,
            'missing_conditions_by_seed': {str(seed): sorted(set(CONDITIONS) - set(conditions)) for seed, conditions in seeds.items()},
            'skipped_updates_by_run': {row['run_id']: row['skipped_updates'] for conditions in seeds.values() for row, _, _ in conditions.values()},
            'contrasts': [],
        }
        for name, metric in metrics:
            for a, b, interpretation in CONTRASTS:
                matched = [(seed, rows) for seed, rows in seeds.items() if a in rows and b in rows]
                if not matched:
                    continue
                differences = np.array([metric(rows[a][0]) - metric(rows[b][0]) for _, rows in matched])
                n = len(differences)
                contrast = {
                    'metric': name, 'model': a, 'reference': b, 'interpretation': interpretation,
                    'n_matched_seeds': n, 'seed_labels': [list(seed) for seed, _ in matched],
                    'mean_difference': float(differences.mean()), 'differences': differences.tolist(),
                    'sample_sd': None, 'nominal_95_interval': None,
                }
                if n > 1:
                    sd = float(differences.std(ddof=1))
                    half = float(t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
                    contrast.update(sample_sd=sd, nominal_95_interval=[float(differences.mean() - half), float(differences.mean() + half)])
                else:
                    contrast['limitation'] = 'One training seed: descriptive difference only; no between-training uncertainty estimate.'
                group['contrasts'].append(contrast)
        reports.append(group)
    return {
        'groups': reports,
        'inference_scope': '1M controls. Paired seed intervals, when available, assume independent seed blocks and are nominal without multiplicity correction. Segment/query samples do not replace independent training seeds. Equal dense arithmetic is not equal total compute or wall time.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', default='runs/communication_1m')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    result = summarize(args.root)
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        Path(args.output).write_text(encoded)
    print(encoded)


if __name__ == '__main__':
    main()
