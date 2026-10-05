"""Freeze quality curves for the fixed historical paper cohort (GET only).

This creates a separate evidence artifact; the July quality report and October
resource snapshot remain unchanged. No test metrics or system details are read.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

from comet_ml.config import get_config

import fetch_frontier_snapshot as fetch
from aistats_resource_accounting import cohort


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs/experiments/aistats2027_historical_quality.json'


def main():
    fetch.KEY = get_config('comet.api_key')
    if not fetch.KEY:
        raise RuntimeError('Comet credential unavailable')
    snapshot = {
        'retrieval_started_utc': datetime.now(timezone.utc).isoformat(),
        'source_cohort': 'docs/experiments/results_aaai.md (2026-07-29)',
        'source_requests': 'Comet REST v2 GET; fixed-cohort quality curves only',
        'training_runs_launched': False,
    }
    metadata = {}
    for task, project in [('text8', 'knitwork-text'), ('SDQ', 'knitwork-sdq')]:
        listing = fetch.read('experiments', {
            'workspaceName': 'team-rl-exp', 'projectName': project,
            'archived': 'false', 'size': 1000,
        })
        metadata[task] = listing['experiments']

    def collect(row):
        result = dict(row, metrics={})
        matches = [m for m in metadata[row['task']] if m['experimentKey'].startswith(row['run_prefix'])]
        if len(matches) != 1:
            raise RuntimeError('Historical cohort ID missing or ambiguous')
        result['experiment_key'] = matches[0]['experimentKey']
        names = ['val/BPC'] if row['task'] == 'text8' else ['Acc++', 'T']
        for name in names:
            raw = fetch.read('experiment/metrics/get-metric', {
                'experimentKey': result['experiment_key'], 'metricName': name,
            })
            result['metrics'][name] = [
                {k: p.get(k) for k in ('step', 'metricValue', 'timestamp')}
                for p in raw.get('metrics', raw.get('values', []))
            ]
        return result

    with ThreadPoolExecutor(max_workers=10) as pool:
        snapshot['runs'] = list(pool.map(collect, cohort()))
    snapshot['retrieval_finished_utc'] = datetime.now(timezone.utc).isoformat()
    OUTPUT.write_text(json.dumps(snapshot, indent=2) + '\n')
    print('Saved fixed historical cohort:', len(snapshot['runs']), 'runs')


if __name__ == '__main__':
    main()
