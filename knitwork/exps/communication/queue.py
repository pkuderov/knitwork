"""Persistent local queue with an optional daily cutoff and manual restart."""

import argparse
from datetime import datetime
import fcntl
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / 'knitwork/exps/communication/config/local_queue.yaml'


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None


def deadline_guard(pid, hard_stop):
    identity = process_identity(pid)
    if identity is None:
        return
    while process_identity(pid) == identity:
        if time.time() >= hard_stop:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return
        time.sleep(min(0.5, max(0.01, hard_stop - time.time())))


def session_deadlines(cfg, now=None):
    if cfg.get('daily_cutoff', True) is False:
        return None, None
    now = now or datetime.now(ZoneInfo(cfg['timezone']))
    def endpoint(clock):
        hour, minute = map(int, clock.split(':'))
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0).timestamp()
    soft, hard = endpoint(cfg['graceful_stop']), endpoint(cfg['hard_stop'])
    if soft >= hard:
        raise ValueError('graceful_stop must precede hard_stop')
    return soft, hard


def paths(cfg):
    output = Path(cfg['output_dir'])
    if not output.is_absolute():
        output = ROOT / output
    directory = output / '_queue'
    directory.mkdir(parents=True, exist_ok=True)
    return output, directory


def prepare_jobs(cfg, output):
    from knitwork.exps.communication.protocol import config_fingerprint, resolve_config
    from knitwork.exps.communication.suite import cases
    matrix = Path(cfg['matrix'])
    if not matrix.is_absolute():
        matrix = ROOT / matrix
    def prepare(config, *, extra=False):
        run_output = Path(config['output_dir']) if extra else output
        if not run_output.is_absolute():
            run_output = ROOT / run_output
        config.update(output_dir=str(run_output), resume='auto', device=cfg['device'])
        config['log']['logger'] = None if cfg['logger'] == 'local' else cfg['logger']
        if config['exp'] == 'text8':
            config['dataset']['path'] = str(Path(cfg['text8_path']).expanduser().resolve())
            if not Path(config['dataset']['path']).is_file():
                raise FileNotFoundError(config['dataset']['path'])
        runner = config.get('training_runner', 'knitwork.exps.communication.run')
        if runner != 'knitwork.exps.communication.run':
            if runner not in ('knitwork.exps.communication.run_single_column', 'knitwork.exps.communication.run_concat'):
                raise ValueError(f'Unsupported queue runner: {runner}')
            extension = importlib.import_module(runner)
            config = extension.prepare_config(config)
            with extension.registered_model():
                resolved = resolve_config(config)
        else:
            resolved = resolve_config(config)
        return {
            'run_id': resolved['run_id'], 'config': config,
            'runner_module': runner,
            'config_fingerprint': config_fingerprint(resolved), 'status': 'pending',
        }
    jobs = [prepare(config) for _, config in cases(matrix)]
    # Interleave tasks; prioritize the original readout's learned/static contrast.
    order = ['dynamic_column0', 'static_column0', 'none_column0', 'dynamic_mean', 'static_mean', 'none_mean']
    jobs.sort(key=lambda job: (
        job['config']['topology'], job['config']['recipe'], job['config']['seed'],
        order.index(job['config']['model'].split('.', 1)[1]), job['config']['exp'] == 'sdq',
    ))
    # Supplementary controls follow the original matrix without expanding its six-way comparisons.
    for filename in cfg.get('additional_configs', []):
        path = Path(filename)
        if not path.is_absolute():
            path = ROOT / path
        jobs.append(prepare(yaml.safe_load(path.read_text()), extra=True))
    identities = [(job['config']['output_dir'], job['run_id']) for job in jobs]
    if len(identities) != len(set(identities)) or len({job['run_id'] for job in jobs}) != len(jobs):
        raise ValueError('Queue jobs must have unique run IDs and output paths')
    return jobs


def run_directory(job, output):
    return Path(job['config'].get('output_dir', output)) / job['run_id']


def reconcile_jobs(jobs, previous, output, *, retry_failed=False):
    saved = {job['run_id']: job for job in previous.get('jobs', [])}
    for job in jobs:
        old = saved.get(job['run_id'])
        if old:
            if old['config_fingerprint'] != job['config_fingerprint']:
                raise ValueError(f"Queue configuration changed: {job['run_id']}")
            if old.get('pid') and process_identity(old['pid']) == old.get('process_identity'):
                raise RuntimeError(f"An orphaned worker is still alive: {old['pid']}; stop it before restarting")
            for name in ('status', 'returncode', 'attempts', 'log_path', 'last_started_at'):
                if name in old:
                    job[name] = old[name]
            if job['status'] == 'running':
                job['status'] = 'paused'
            if retry_failed and job['status'] == 'failed':
                job['status'] = 'paused'
        run_dir = run_directory(job, output)
        summary = run_dir / 'summary.json'
        if summary.exists() and json.loads(summary.read_text())['completed']:
            metadata = json.loads((run_dir / 'metadata.json').read_text())
            if metadata['config_fingerprint'] != job['config_fingerprint']:
                raise ValueError(f'Completed configuration differs: {run_dir}')
            job['status'] = 'complete'


def supervise(cfg, *, retry_failed=False):
    output, directory = paths(cfg)
    with (directory / 'queue.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Queue supervisor is already running') from None
        soft, hard = session_deadlines(cfg)
        if soft is not None and time.time() >= soft:
            raise RuntimeError('Daily training window is closed; restart manually tomorrow before graceful_stop')
        if cfg.get('manual_restart') is not True:
            raise ValueError('This queue requires explicit manual restart')
        workers = cfg['workers']
        if type(workers) is not int or workers < 1:
            raise ValueError('workers must be a positive integer')
        state_path = directory / 'status.json'
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        jobs = prepare_jobs(cfg, output)
        reconcile_jobs(jobs, previous, output, retry_failed=retry_failed)
        state = {
            'pid': os.getpid(), 'process_identity': process_identity(os.getpid()),
            'status': 'running', 'workers': workers, 'jobs': jobs,
            'timezone': cfg['timezone'], 'soft_stop_epoch': soft, 'hard_stop_epoch': hard,
            'daily_cutoff': soft is not None,
            'manual_restart': True, 'session_started_at': time.time(),
        }
        active = {}
        stop_reason = None
        sent_pause = False
        next_launch = 0.0
        def event(kind, **values):
            record = {'time': datetime.now(ZoneInfo(cfg['timezone'])).isoformat(), 'event': kind, **values}
            with (directory / 'events.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
        def persist():
            state['updated_at'] = time.time()
            write_json(state_path, state)
        def request_stop(number, frame):
            nonlocal stop_reason
            stop_reason = signal.Signals(number).name
        for number in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(number, request_stop)
        event('session_started', workers=workers, jobs=len(jobs))
        persist()
        try:
            while True:
                now = time.time()
                if soft is not None and now >= soft and stop_reason is None:
                    stop_reason = 'daily_deadline'
                if stop_reason and not sent_pause:
                    for process, job in active.values():
                        if process.poll() is None:
                            process.send_signal(signal.SIGTERM)
                    sent_pause = True
                    state['status'] = 'stopping'
                    state['stop_reason'] = stop_reason
                    event('pause_requested', reason=stop_reason)
                    persist()
                for pid, (process, job) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        if hard is not None and now >= hard:
                            try:
                                os.killpg(pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        continue
                    summary_path = run_directory(job, output) / 'summary.json'
                    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
                    job['returncode'] = code
                    job['status'] = 'complete' if code == 0 and summary.get('completed') else ('paused' if stop_reason else 'failed')
                    job.pop('pid', None)
                    job.pop('process_identity', None)
                    del active[pid]
                    event('worker_exited', run_id=job['run_id'], status=job['status'], returncode=code)
                    if job['status'] == 'failed' and stop_reason is None:
                        stop_reason = 'worker_failed'
                    persist()
                if stop_reason and not active:
                    state['status'] = 'paused'
                    persist()
                    event('session_closed', reason=stop_reason, manual_restart=True)
                    return
                pending = [job for job in jobs if job['status'] in ('paused', 'pending')]
                if not active and not pending:
                    state['status'] = 'failed' if any(job['status'] == 'failed' for job in jobs) else 'complete'
                    persist()
                    event('session_closed', reason=state['status'])
                    return
                # Warm up the first full batch before launching the second compiler.
                warming = any(not (run_directory(job, output) / 'progress.json').exists() or
                    json.loads((run_directory(job, output) / 'progress.json').read_text()).get('updates', 0) < 1
                    for _, job in active.values())
                if not stop_reason and pending and len(active) < workers and not warming and now >= next_launch:
                    job = pending[0]
                    job_dir = directory / 'jobs' / job['run_id']
                    job_dir.mkdir(parents=True, exist_ok=True)
                    config = dict(job['config'], runtime={'deadline': soft})
                    config_path = job_dir / 'config.yaml'
                    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                    log_path = job_dir / 'train.log'
                    env = os.environ.copy()
                    env.update({
                        'PYTHONUNBUFFERED': '1', 'TORCHINDUCTOR_COMPILE_THREADS': '1',
                        'TORCHINDUCTOR_CACHE_DIR': str(output / '_compiler_cache'),
                        'COMET_TIMEOUT_HTTP': '30',
                    })
                    command = [sys.executable, '-m', job['runner_module'], str(config_path)]
                    with log_path.open('a') as log:
                        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        if hard is not None:
                            subprocess.Popen([sys.executable, '-m', 'knitwork.exps.communication.queue',
                                '--guard', str(process.pid), str(hard)], cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    job.update(status='running', pid=process.pid, process_identity=process_identity(process.pid),
                        attempts=job.get('attempts', 0) + 1, log_path=str(log_path), last_started_at=time.time())
                    active[process.pid] = (process, job)
                    next_launch = now + cfg['launch_spacing_seconds']
                    event('worker_started', run_id=job['run_id'], pid=process.pid)
                    persist()
                time.sleep(0.5)
        finally:
            for process, job in active.values():
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
            # When enabled, workers and independent guards also enforce the daily cutoff.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default=str(DEFAULT_CONFIG))
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--guard', nargs=2, metavar=('PID', 'HARD_STOP'))
    args = parser.parse_args()
    if args.guard:
        deadline_guard(int(args.guard[0]), float(args.guard[1]))
        return
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    _, directory = paths(cfg)
    state_path = directory / 'status.json'
    if args.status or args.stop:
        state = json.loads(state_path.read_text())
        if args.stop:
            if process_identity(state['pid']) != state['process_identity']:
                raise RuntimeError('Supervisor is not running')
            os.kill(state['pid'], signal.SIGUSR1)
            print('Graceful pause requested')
        else:
            print(json.dumps({key: value for key, value in state.items() if key != 'jobs'}, indent=2))
            for job in state['jobs']:
                print(f"{job['status']:8} {job['run_id']} pid={job.get('pid', '-')}")
        return
    if args.detach:
        command = [sys.executable, '-m', 'knitwork.exps.communication.queue', str(config_path)]
        if args.retry_failed:
            command.append('--retry-failed')
        with (directory / 'supervisor.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'Supervisor PID {process.pid}; log: {directory / "supervisor.log"}')
    else:
        supervise(cfg, retry_failed=args.retry_failed)


if __name__ == '__main__':
    main()
