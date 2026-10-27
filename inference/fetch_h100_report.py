"""Read-only Comet refresh restricted to experiment IDs present in H100 logs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference import fetch_frontier_snapshot as client

METRICS = {
    'val/Loss', 'val/BPC', 'val/Acc', 'val/Exact_match', 'val_w1024/Loss',
    'BPC', 'Acc', 'Loss', '|Grad|', 'Skipped', 'global_step', 'perf/fps', 'T',
    'L_comm', 'H_comm', 'eval/EpRet', 'train/EpRet', 'ApproxKL', 'KLStop',
    'state/RMS', 'state/PoleRadiusMax', 'train/Acc', 'train/Grad_norm',
    'train/skipped_steps', 'train/update', 'train/epoch',
}


def wanted(name):
    return name in METRICS or name.startswith('val/T') and name.endswith('/Acc')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='docs/experiments/h100_2026-10-12')
    args = parser.parse_args()
    directory = Path(args.directory)
    server = json.loads((directory/'server_snapshot.json').read_text())
    ids = {run_id for job in server['records'] for run_id in job['comet_ids']}
    load_dotenv(ROOT / '.env')
    client.KEY = os.getenv('COMET_API_KEY')
    if not client.KEY:
        raise RuntimeError('Comet credential unavailable')
    client.wanted = wanted
    selected = {}
    for project in ('knitwork-aistat', 'knitwork-mikasa'):
        for archived in ('false', 'true'):
            raw = client.read('experiments', {'workspaceName': 'team-rl-exp', 'projectName': project, 'archived': archived, 'size': 1000})
            for metadata in raw['experiments']:
                if metadata['experimentKey'] in ids:
                    selected[metadata['experimentKey']] = (project, metadata)
    result = {
        'retrieval_started_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'Comet REST GET only; experiment IDs strictly intersected with fetched aicenter3 queue logs',
        'server_ids': sorted(ids), 'unresolved_ids': sorted(ids-set(selected)), 'runs': [],
        'training_runs_launched': False,
    }
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, run in enumerate(pool.map(client.collect, selected.values()), 1):
            result['runs'].append(run)
            if i % 10 == 0:
                print(f'{i}/{len(selected)} experiments read', flush=True)
    result['retrieval_finished_utc'] = datetime.now(timezone.utc).isoformat()
    (directory/'comet_snapshot.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f"Fetched {len(result['runs'])}; unresolved IDs {len(result['unresolved_ids'])}; retrieval errors {sum(bool(r['errors']) for r in result['runs'])}")


if __name__ == '__main__':
    main()
