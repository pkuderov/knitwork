import contextlib
import copy
from datetime import datetime
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import torch
import yaml

from test_communication_controls import CONFIG, load_checkpoint, tiny_config
from knitwork.exps.communication.protocol import config_fingerprint, resolve_config, tensor_digest
from knitwork.exps.communication.queue import deadline_guard, prepare_jobs, reconcile_jobs, session_deadlines
from knitwork.exps.communication.run import main
from knitwork.exps.communication.runtime import PauseRequested, StopController


class QueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_daily_window_and_order(self):
        cfg = yaml.safe_load((CONFIG / 'local_queue.yaml').read_text())
        cfg.pop('additional_configs', None)
        cfg['daily_cutoff'] = True
        now = datetime(2026, 10, 10, 22, 56, tzinfo=ZoneInfo('Europe/Moscow'))
        soft, hard = session_deadlines(cfg, now)
        self.assertLess(soft, now.timestamp())
        self.assertGreater(hard, now.timestamp())
        self.assertEqual(hard - soft, 300)
        with tempfile.TemporaryDirectory() as temporary:
            cfg['text8_path'] = str(Path(temporary) / 'text8.txt')
            Path(cfg['text8_path']).write_text('abcdefghijklmnopqrstuvwxyz ' * 100)
            jobs = prepare_jobs(cfg, Path(temporary))
            self.assertEqual(len(jobs), 12)
            self.assertEqual([job['run_id'].split('/')[0] for job in jobs[:2]], ['text8', 'sdq'])
            self.assertTrue(all('L2C4/clean/dynamic_column0' in job['run_id'] for job in jobs[:2]))
            saved = copy.deepcopy(jobs)
            saved[0]['status'] = 'running'
            saved[1]['status'] = 'failed'
            reconcile_jobs(jobs, {'jobs': saved}, Path(temporary))
            self.assertEqual(jobs[0]['status'], 'paused')
            self.assertEqual(jobs[1]['status'], 'failed')
            reconcile_jobs(jobs, {'jobs': saved}, Path(temporary), retry_failed=True)
            self.assertEqual(jobs[1]['status'], 'paused')

    def test_no_cutoff_after_23_and_on_next_day(self):
        cfg = yaml.safe_load((CONFIG / 'local_queue.yaml').read_text())
        self.assertIs(cfg['daily_cutoff'], False)
        for day, hour in ((12, 23), (13, 0)):
            now = datetime(2026, 10, day, hour, 30, tzinfo=ZoneInfo('Europe/Moscow'))
            soft, hard = session_deadlines(cfg, now)
            self.assertIsNone(soft)
            self.assertIsNone(hard)
            with patch('knitwork.exps.communication.runtime.time.time', return_value=now.timestamp()):
                control = StopController(soft)
                control.check()
                self.assertIsNone(control.reason)

    def test_legacy_config_keeps_daily_cutoff(self):
        cfg = yaml.safe_load((CONFIG / 'local_queue.yaml').read_text())
        cfg.pop('daily_cutoff')
        now = datetime(2026, 10, 12, 23, 30, tzinfo=ZoneInfo('Europe/Moscow'))
        soft, hard = session_deadlines(cfg, now)
        self.assertLess(soft, hard)
        self.assertLess(hard, now.timestamp())

    def test_guard_really_kills_only_its_worker_group(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        try:
            deadline_guard(process.pid, time.time() + 0.05)
            self.assertEqual(process.wait(timeout=5), -signal.SIGKILL)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_signal_handler_is_cooperative_and_restored(self):
        old = signal.getsignal(signal.SIGUSR1)
        control = StopController()
        control.install()
        try:
            signal.raise_signal(signal.SIGUSR1)
            with self.assertRaises(PauseRequested):
                control.check()
            self.assertEqual(control.reason, 'SIGUSR1')
        finally:
            control.restore()
        self.assertEqual(signal.getsignal(signal.SIGUSR1), old)

    def test_interrupted_evaluation_resumes_identically(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            dataset = root / 'text8.txt'
            dataset.write_text('abcdefghijklmnopqrstuvwxyz ' * 100)
            for task in ('text8', 'sdq'):
                full = tiny_config(task, root / f'{task}-full', dataset)
                interrupted = tiny_config(task, root / f'{task}-interrupted', dataset)
                main(full)
                original = StopController.check
                calls = 0
                def check(control):
                    nonlocal calls
                    calls += 1
                    # Check 7 is inside the evaluation triggered by update 2.
                    if calls == 7:
                        control.request('test_evaluation_pause')
                    original(control)
                with patch.object(StopController, 'check', check):
                    summary = main(interrupted)
                self.assertFalse(summary['completed'])
                self.assertEqual(summary['tokens'], 32)
                _, paused = load_checkpoint(interrupted)
                self.assertTrue(paused['pending_evaluation'])
                interrupted.update(resume='auto', runtime={'deadline': time.time() + 60})
                self.assertEqual(config_fingerprint(resolve_config(full)), config_fingerprint(resolve_config(interrupted)))
                self.assertTrue(main(interrupted)['completed'])
                _, expected = load_checkpoint(full)
                _, actual = load_checkpoint(interrupted)
                self.assertEqual(tensor_digest(actual['model']), tensor_digest(expected['model']))
                torch.testing.assert_close(actual['recurrent_state']['h'], expected['recurrent_state']['h'], rtol=0, atol=0)

    def test_deadline_pauses_before_training_and_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            cfg = tiny_config('sdq', Path(temporary))
            cfg['runtime'] = {'deadline': 0}
            paused = main(cfg)
            self.assertEqual(paused['tokens'], 0)
            self.assertFalse(paused['completed'])
            self.assertEqual(paused['pause_reason'], 'daily_deadline')
            cfg.update(resume='auto', runtime={'deadline': time.time() + 60})
            self.assertTrue(main(cfg)['completed'])


if __name__ == '__main__':
    unittest.main()
