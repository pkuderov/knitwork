"""Fetch a sanitized frontier snapshot with Comet REST GET requests only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading

import requests
from comet_ml.config import get_config


LOCAL = threading.local()
KEY = None
BASE = 'https://www.comet.com/api/rest/v2/'
EXACT = {
    'Loss', 'BPC', 'Acc', 'LR', 'T', 'Upd', 'Skipped', '|Grad|', 'H',
    'ApproxKL', 'ClipFrac', 'ValidFraction', 'KLStop', 'L_pi', 'L_v',
    'global_step', 'perf/fps', 'L_comm', 'H_comm',
}
PREFIXES = ('val', 'train/', 'eval/', 'env/', 'state/', 'col_sim/', 'write_gate/', 'memory_write_rate/', 'slot/')


def read(endpoint, params):
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.headers['Authorization'] = KEY
    for attempt in range(3):
        try:
            response = LOCAL.session.get(BASE + endpoint, params=params, timeout=(10, 40))
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 2:
                raise


def wanted(name):
    return name in EXACT or name.startswith(PREFIXES)


def collect(item):
    project, meta = item
    selected_metadata = ('experimentKey', 'experimentName', 'durationMillis', 'startTimeMillis', 'endTimeMillis', 'running', 'archived')
    result = {
        'project': project, 'metadata': {k: meta.get(k) for k in selected_metadata},
        'parameters': {}, 'metric_summary': {}, 'metrics': {}, 'errors': [],
    }
    params = {'experimentKey': meta['experimentKey']}
    try:
        raw = read('experiment/parameters', params)['values']
        for p in raw:
            if not any(word in p['name'].lower() for word in ('path', 'api_key', 'password', 'secret', 'run_dir')):
                result['parameters'][p['name']] = p.get('valueCurrent')
        summary = read('experiment/metrics/summary', params)['values']
        result['test_presence_audit'] = {
            'test_metric_names_present': [p['name'] for p in summary if p['name'].startswith('test/')],
            'test_values_read_for_selection': False,
        }
        for p in summary:
            name = p['name']
            if not wanted(name):
                continue
            fields = ('valueCurrent', 'valueMin', 'valueMax', 'stepCurrent', 'stepMin', 'stepMax', 'timestampCurrent')
            result['metric_summary'][name] = {k: p.get(k) for k in fields}
            try:
                points = read('experiment/metrics/get-metric', params | {'metricName': name})
                result['metrics'][name] = [
                    {k: p.get(k) for k in ('step', 'metricValue', 'timestamp', 'epoch')}
                    for p in points.get('metrics', points.get('values', []))
                ]
            except Exception as error:
                result['errors'].append({'metric': name, 'error': type(error).__name__})
        if 'mqar_' in meta.get('experimentName', ''):
            output = read('experiment/output', params).get('output') or ''
            result['stdout_audit'] = {
                'available': bool(output), 'characters': len(output),
                'contains_traceback': 'Traceback' in output,
                'contains_best_update_line': 'Best update' in output,
                'status_complete': 'STATUS complete' in output,
                'status_failed': 'STATUS failed' in output,
            }
    except Exception as error:
        result['errors'].append({'error': type(error).__name__})
    return result


def main():
    global KEY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    KEY = get_config('comet.api_key')
    if not KEY:
        raise RuntimeError('Comet credential unavailable')
    projects = ['knitwork-aistat', 'knitwork-mikasa']
    snapshot = {
        'retrieval_started_utc': datetime.now(timezone.utc).isoformat(),
        'workspace': 'team-rl-exp', 'projects': projects,
        'source_requests': 'Comet REST v2 GET only; selected validation/training/RL curves. Test values and system metrics excluded.',
        'training_runs_launched': False,
    }
    items = []
    for project in projects:
        response = read('experiments', {'workspaceName': 'team-rl-exp', 'projectName': project, 'archived': 'false', 'size': 1000})
        metadata = [m for m in response['experiments'] if not m.get('archived')]
        print(project, len(metadata), flush=True)
        items.extend((project, m) for m in metadata)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        snapshot['runs'] = list(pool.map(collect, items))
    snapshot['retrieval_finished_utc'] = datetime.now(timezone.utc).isoformat()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=2) + '\n')
    print('Saved', target, 'runs', len(items), 'runs with errors', sum(bool(r['errors']) for r in snapshot['runs']), flush=True)


if __name__ == '__main__':
    main()
