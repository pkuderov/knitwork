import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from knitwork.common.config import load_config
from knitwork.common.dynamic_param import DynamicParameter
from knitwork.common.state_size import build
from knitwork.common.utils import count_learnable_params
from knitwork.exps.text.run import main, run_eval
from knitwork.models.lru_core import LruCore


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT = ROOT / 'knitwork/exps/text/config/rollout_mid'
ABLATION = ROOT / 'knitwork/exps/text/config/grnn_ablation_mid'


class RolloutPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_plain_lru_matches_direct_bank_stack_and_has_only_recurrent_state(self):
        model = LruCore(hidden_size=12, n_layers=2, horizon=[1, 1000], dtype=torch.float32, device='cpu')
        x, state = torch.randn(2, 1, 12), model.init_state(2)
        out, next_state, _ = model(x, state)
        expected, hidden = x, []
        for layer in range(2):
            expected, h = model.cells(layer, expected.transpose(0, 1), state['h'][layer].unsqueeze(0))
            hidden.append(h[0])
        torch.testing.assert_close(out, expected[:, 0])
        torch.testing.assert_close(next_state['h'], torch.stack(hidden))
        self.assertEqual(set(next_state), {'h'})
        cleared = model.reset_state(next_state, torch.tensor([True, False]))
        self.assertEqual(cleared['h'][:, 0].count_nonzero().item(), 0)
        torch.testing.assert_close(cleared['h'][:, 1], next_state['h'][:, 1])
        self.assertFalse(model.detach_state(next_state)['h'].requires_grad)
        compiled = torch.compile(model, backend='eager', fullgraph=True)
        actual, _, _ = compiled(x, state)
        torch.testing.assert_close(actual, out)
        actual.square().mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
        torch._dynamo.reset()

    def test_rollout_configs_match_token_budgets_reset_trajectories_and_sizes(self):
        expected = {'grnn_gru': 1050499, 'grnn_lru': 1049463, 'gru': 1014363, 'lru': 1046103}
        # Measure the controls explicitly: widths are fixed, rather than silently rounded at launch.
        for family in expected:
            reference = load_config(ROLLOUT / f'{family}_r64.yaml')
            model = build(reference, reference['model'].replace('.', '_'))
            self.assertEqual(count_learnable_params(model), expected[family])
            for directory, lengths, batch in [(ROLLOUT, [64, 256, 512, 1024], 32768), (ROLLOUT / 'low_memory', [64, 512], 16384)]:
                trajectories = []
                for length in lengths:
                    cfg = load_config(directory / f'{family}_r{length}.yaml')
                    self.assertEqual(cfg['rollout_len'] * cfg['n_envs'], batch)
                    self.assertEqual(cfg['n_steps'] % batch, 0)
                    self.assertEqual(cfg['n_steps'], 67108864)
                    self.assertEqual(cfg['eval']['n_envs'], 512)
                    self.assertEqual(cfg['eval']['max_tokens'], 524288)
                    self.assertIsNone(cfg['eval']['window'])
                    self.assertFalse(cfg['eval']['test_on_finish'])
                    self.assertIsNone(cfg['vis_inspect_schedule'])
                    reset = copy.deepcopy(cfg['gens']['text8']['reset_prob'])
                    reset['val'] /= length
                    reset['tar'] /= length
                    parameter = DynamicParameter(**reset)
                    values = []
                    for _ in range(2049 * (32768 // batch)):
                        values.append(parameter.val)
                        parameter.step()
                    trajectories.append(values[::32768 // batch])
                    self.assertEqual(cfg['lr']['schedule'], 250 * (32768 // batch))
                for values in trajectories[1:]:
                    self.assertEqual(values, trajectories[0])

    def test_ablation_factorial_is_unique_and_full_control_reuses_rollout(self):
        cfgs = [load_config(path) for path in ABLATION.glob('*.yaml')]
        combinations = set()
        control = load_config(ROLLOUT / 'grnn_gru_r64.yaml')
        for cfg in cfgs:
            options = cfg['grnn_L2C4']
            combinations.add((options['noise_std'], cfg['communication']['loss_weight'], cfg['communication']['entropy_weight']))
            for key in ('lr', 'gens', 'eval', 'rollout_len', 'n_envs', 'n_steps', 'seed'):
                self.assertEqual(cfg[key], control[key])
            self.assertEqual(options | {'noise_std': 0.05}, control['grnn_L2C4'])
        self.assertEqual(len(combinations), 8)
        full = load_config(ABLATION / 'full.yaml')
        full.pop('name')
        control.pop('name')
        self.assertEqual(full, control)

    def test_all_ablation_objectives_and_long_rollout_backpropagate(self):
        for path in ABLATION.glob('*.yaml'):
            cfg = load_config(path)
            model = build(cfg, 'grnn_L2C4', {'hidden_size': 12})
            model.train()
            state = model.init_state(2)
            loss = 0
            for _ in range(3):
                y, state, info = model(torch.randint(27, (1, 2)), state)
                loss += y.square().mean()
                loss += cfg['communication']['loss_weight'] * torch.stack(info['comm_loss']).mean()
                loss -= cfg['communication']['entropy_weight'] * torch.stack(info['comm_entropy']).mean()
            loss.backward()
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
        for family in ('grnn_gru', 'grnn_lru', 'gru', 'lru'):
            cfg = load_config(ROLLOUT / f'{family}_r512.yaml')
            model = build(cfg, cfg['model'].replace('.', '_'), {'hidden_size': 12})
            state = model.init_state(1)
            loss = 0
            for _ in range(512):
                y, state, _ = model(torch.randint(27, (1, 1)), state)
                loss += y.square().mean() / 512
            loss.backward()
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))

    def test_capped_fixed_stream_eval_does_not_access_test_or_repeat_final_validation(self):
        cfg = load_config(ROLLOUT / 'gru_r64.yaml')
        cfg.update(n_envs=2, rollout_len=4, n_steps=16, compile=False)
        cfg['rnn_L2']['hidden_size'] = 12
        cfg['eval'].update(n_envs=4, max_tokens=16, schedule=16, val_size=108, test_size=108, window=4)
        cfg['log'].update(logger=None, schedule=16)
        calls = []
        def record(*args, **kwargs):
            calls.append((kwargs['prefix'], kwargs['gen'].n_envs, kwargs['max_rollout']))
            return run_eval(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'text.txt'
            path.write_text('abcdefghijklmnopqrstuvwxyz ' * 30)
            cfg['gens']['text8']['path'] = str(path)
            with contextlib.redirect_stdout(io.StringIO()), patch('torch.compile', side_effect=lambda value, **kwargs: value), patch('knitwork.exps.text.run.run_eval', side_effect=record):
                main(cfg)
        self.assertEqual(calls, [('val', 4, 4), ('val_w4', 4, 4)])

    def test_default_eval_retains_full_pass_and_test_access(self):
        cfg = load_config(ROLLOUT / 'lru_r64.yaml')
        cfg.update(n_envs=2, rollout_len=4, n_steps=8, compile=False)
        cfg['lru_L2']['hidden_size'] = 12
        cfg['eval'].update(schedule=8, val_size=54, test_size=54, window=None)
        for key in ('n_envs', 'max_tokens', 'test_on_finish'):
            cfg['eval'].pop(key)
        cfg['log'].update(logger=None, schedule=8)
        calls = []
        def record(*args, **kwargs):
            calls.append((kwargs['prefix'], kwargs['gen'].n_envs, kwargs['max_rollout']))
            return run_eval(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'text.txt'
            path.write_text('abcdefghijklmnopqrstuvwxyz ' * 30)
            cfg['gens']['text8']['path'] = str(path)
            with contextlib.redirect_stdout(io.StringIO()), patch('torch.compile', side_effect=lambda value, **kwargs: value), patch('knitwork.exps.text.run.run_eval', side_effect=record):
                main(cfg)
        self.assertEqual(calls, [('val', 2, 27), ('test_last', 2, 27), ('test', 2, 27)])


if __name__ == '__main__':
    unittest.main()
