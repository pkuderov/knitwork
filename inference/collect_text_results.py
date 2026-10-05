"""Collect final metrics of the finished text/MIKASA queue jobs from Comet (run on the server, needs COMET_API_KEY).

python inference/collect_text_results.py <queue_dir> <out.json>
For every job in queue/done: take the last Comet experiment key from its log and read the metric summary.
"""
import glob
import json
import os
import re
import sys
import time

from comet_ml import API

WANT = ('val/BPC', 'val_w1024/BPC', 'test/BPC', 'test_w1024/BPC', 'test_last/BPC', 'test_last_w1024/BPC',
        'val/Loss', 'val_w1024/Loss', 'perf/fps', 'skipped_steps')


def retry(fn, tries=6):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def main(queue, out):
    api = API(api_key=os.environ['COMET_API_KEY'])
    rows = []
    for jf in sorted(glob.glob(f'{queue}/done/*.json')):
        stem = os.path.basename(jf)[:-5]
        log = f'{queue}/logs/{stem}.log'
        if not os.path.exists(log):
            continue
        keys = re.findall(r'live on comet\.com \S+/([0-9a-f]{32})', open(log, errors='ignore').read())
        if not keys:
            continue
        try:
            exp = retry(lambda: api.get_experiment_by_key(keys[-1]))
            name = retry(exp.get_name)
            summary = {m['name']: m for m in retry(exp.get_metrics_summary)}
        except Exception as e:
            print('skip', stem, type(e).__name__, flush=True)
            continue
        metrics = {k: summary[k].get('valueCurrent') for k in WANT if k in summary}
        rows.append({'job': stem, 'name': name, 'key': keys[-1], 'metrics': metrics})
        print('ok', stem, flush=True)
    json.dump(rows, open(out, 'w'), indent=1)
    print(len(rows), 'rows ->', out)


if __name__ == '__main__':
    main(*sys.argv[1:3])
