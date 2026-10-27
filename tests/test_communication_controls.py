import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from knitwork.exps.communication.evaluation import evaluate_text, fixed_gap_batch
from knitwork.exps.communication.protocol import CONDITIONS, accounting, make_model, resolve_config, tensor_digest
from knitwork.exps.communication.run import main
from knitwork.exps.communication.summarize import summarize
from knitwork.exps.communication.suite import cases, check_matrix
from knitwork.models.grnn_core import GridRnn as ProductionGridRnn
from knitwork.models.grnn_communication import GridRnn


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'knitwork/exps/communication/config'


def small_core(routing='dynamic', readout='column0'):
    return GridRnn(
        hidden_size=12, n_layers=2, n_columns=4, n_attn_heads=4,
        routing=routing, readout=readout, dtype=torch.float32, device='cpu', noise_std=0.0,
    )


def tiny_config(task, output, dataset=None, *, condition='dynamic_mean', recipe='paper'):
    cfg = yaml.safe_load((CONFIG / f'{task}.yaml').read_text())
    cfg.update(
        model=f'grnn_communication.{condition}', output_dir=str(output),
        device='cpu', compile=False, n_envs=4, rollout_len=4, n_steps=96, recipe=recipe,
    )
    cfg['topologies']['L2C4']['hidden_size'] = 12
    cfg['parameter_range'] = [1, 100000]
    cfg['checkpoint']['every_tokens'] = 20
    cfg['checkpoint']['keep_history'] = True
    cfg['log'].update(logger=None, schedule=32)
    cfg['eval'].update(schedule=32, n_envs=4)
    cfg['lr']['warmup'].update(n_linear_steps=2, schedule=1)
    cfg['lr']['schedule'] = 1
    if task == 'text8':
        cfg['dataset'].update(path=str(dataset), val_size=108, test_size=108)
        cfg['eval'].update(segment_len=8, max_tokens=None)
        cfg['reset_prob']['warmup'].update(n_linear_steps=2, schedule=1)
        cfg['reset_prob']['schedule'] = 1
    else:
        cfg['curriculum']['schedule'] = 8
        cfg['eval'].update(sdq_steps=8, fixed_gaps=[2, 4], fixed_gap_examples=8)
    return cfg


def load_checkpoint(cfg):
    resolved = resolve_config(cfg)
    directory = Path(cfg['output_dir']) / resolved['run_id']
    return directory, torch.load(directory / 'latest.pt', weights_only=False)


class CommunicationControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_dynamic_column0_is_bit_exact_production_and_preserves_initialization(self):
        kwargs = dict(hidden_size=12, n_layers=2, n_columns=4, n_attn_heads=4, dtype=torch.float32, device='cpu', bank=2, mha=2, noise_std=0.05)
        torch.manual_seed(42)
        production = ProductionGridRnn(**kwargs)
        torch.manual_seed(42)
        controlled = GridRnn(**kwargs)
        self.assertEqual(tensor_digest(production.state_dict()), tensor_digest(controlled.state_dict()))
        x = torch.randn(1, 2, 12)
        h = torch.randn(2, 4, 2, 12)
        state = {'h': h, 'out': h[-1]}
        torch.manual_seed(1337)
        expected, expected_state, expected_info = production(x, state, capture=True)
        torch.manual_seed(1337)
        actual, actual_state, actual_info = controlled(x, state, capture=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_state['h'], expected_state['h'], rtol=0, atol=0)
        for name in expected_info:
            for a, b in zip(actual_info[name], expected_info[name]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_static_routes_ignore_contents_but_values_and_router_train(self):
        model = small_core('static', 'mean')
        layer = model.attn[0]
        q = torch.randn(4, 2, 12, requires_grad=True)
        k = torch.randn(5, 2, 12, requires_grad=True)
        v = torch.randn(5, 2, 12, requires_grad=True)
        out, info = layer(q, k, v, return_weights=True)
        out2, info2 = layer(q * 100 + 99, k * 100 - 88, v, return_weights=True)
        torch.testing.assert_close(out, out2, rtol=0, atol=0)
        torch.testing.assert_close(info['attn_weights'], info2['attn_weights'], rtol=0, atol=0)
        out.square().mean().backward()
        self.assertIsNone(q.grad)
        self.assertIsNone(k.grad)
        self.assertGreater(float(v.grad.abs().sum()), 0)
        self.assertGreater(float(layer.ids.grad.abs().sum()), 0)
        self.assertGreater(float(layer.mha.in_proj_weight.grad.abs().sum()), 0)

    def test_none_has_local_feedback_external_access_and_zero_cross_column_jacobian(self):
        model = small_core('none', 'mean')
        h = torch.randn(2, 4, 2, 12, requires_grad=True)
        x = torch.randn(1, 2, 12, requires_grad=True)
        _, next_state, info = model(x, {'h': h, 'out': h[-1]}, capture=True)
        for column in range(4):
            grad_h, grad_x = torch.autograd.grad(next_state['h'][-1, column].square().sum(), (h, x), retain_graph=True)
            others = [c for c in range(4) if c != column]
            self.assertEqual(grad_h[:, others].count_nonzero().item(), 0)
            self.assertGreater(float(grad_h[:, column].abs().sum()), 0)
            self.assertGreater(float(grad_x.abs().sum()), 0)
        bottom, upper = info['attn_weights']
        allowed = torch.eye(4, 5, dtype=torch.bool)
        allowed[:, 4] = True
        self.assertEqual(bottom[~allowed].count_nonzero().item(), 0)
        self.assertTrue((bottom[:, 4] > 0).all())
        torch.testing.assert_close(upper, torch.eye(4), rtol=0, atol=0)

    def test_readout_preserves_state_and_mean_exposes_every_column(self):
        for routing in ('dynamic', 'static', 'none'):
            torch.manual_seed(42)
            column0 = small_core(routing, 'column0')
            torch.manual_seed(42)
            mean = small_core(routing, 'mean')
            x = torch.randn(1, 2, 12)
            state = column0.init_state(2)
            first, first_state, _ = column0(x, state)
            pooled, pooled_state, _ = mean(x, state)
            torch.testing.assert_close(first_state['h'], pooled_state['h'], rtol=0, atol=0)
            torch.testing.assert_close(first, first_state['h'][-1, 0])
            torch.testing.assert_close(pooled, pooled_state['h'][-1].mean(dim=0))
            pooled_state['h'].retain_grad()
            pooled.square().sum().backward()
            self.assertTrue((pooled_state['h'].grad[-1].abs().sum(dim=(1, 2)) > 0).all())

    def test_all_conditions_reset_detach_backward_and_compile(self):
        for condition in CONDITIONS:
            torch._dynamo.reset()
            routing, readout = condition.split('_', 1)
            model = small_core(routing, readout)
            compiled = torch.compile(model, backend='eager', fullgraph=True)
            state = model.init_state(2)
            loss = 0
            for _ in range(3):
                y, state, _ = compiled(torch.randn(1, 2, 12), state)
                loss = loss + y.square().mean()
            loss.backward()
            self.assertTrue(all(p.grad is None or p.grad.isfinite().all() for p in model.parameters()))
            cleared = model.reset_state(state, torch.tensor([True, False]))
            self.assertEqual(cleared['h'][:, :, 0].count_nonzero().item(), 0)
            torch.testing.assert_close(cleared['h'][:, :, 1], state['h'][:, :, 1])
            self.assertFalse(model.detach_state(state)['h'].requires_grad)
        torch._dynamo.reset()

    def test_matrix_matches_parameters_state_initial_weights_and_profiled_dense_work(self):
        report = check_matrix(CONFIG / 'matrix.yaml', profile=True)
        self.assertEqual(len(report), 12)
        expected = {('text8', 'L2C4'): 1050499, ('text8', 'L3C4'): 1068743, ('sdq', 'L2C4'): 1053338, ('sdq', 'L3C4'): 1071078}
        for row in report:
            task, topology = row['run_id'].split('/')[:2]
            self.assertEqual(row['parameters_nominal_trainable'], expected[task, topology])
            self.assertEqual(row['profiled_dense_forward_flops_per_token'], row['dense_forward_flops_per_token'])
        self.assertEqual(len(list(cases(CONFIG / 'matrix.yaml'))), 12)

    def test_fixed_gap_probe_uses_exact_native_sdq_targets(self):
        cfg = yaml.safe_load((CONFIG / 'sdq.yaml').read_text())['eval_generator']
        for gap in (1, 2, 32):
            tokens, targets = fixed_gap_batch(cfg, gap=gap, n_examples=20, seed=99)
            stored = np.zeros((20, 5), dtype=int)
            stores = np.zeros_like(stored)
            queries = np.zeros_like(stored)
            distract = np.zeros(20, dtype=int)
            initial_key = tokens[0] // 10
            for t, batch in enumerate(tokens):
                for b, token in enumerate(batch):
                    if token < 50:
                        k, v = divmod(token, 10)
                        if t:
                            self.assertNotEqual(k, initial_key[b])
                        stored[b, k] = v
                        stores[b, k] += 1
                    elif token < 60:
                        distract[b] = (distract[b] + token - 50) % 10
                    else:
                        queries[b, token - 60] += 1
            ixs = np.arange(20)
            expected = (stored[ixs, initial_key] + distract + stores[ixs, initial_key] - queries[ixs, initial_key]) % 10
            np.testing.assert_array_equal(tokens[-1] - 60, initial_key)
            np.testing.assert_array_equal(targets, expected)

    def test_segmented_text_evaluation_is_aligned_and_preserves_training_mode(self):
        cfg = resolve_config(yaml.safe_load((CONFIG / 'text8.yaml').read_text()))
        cfg[cfg['model'].replace('.', '_')]['hidden_size'] = 12
        model = make_model(cfg)
        data = np.tile(np.arange(27), 8)
        evaluation = {'n_envs': 4, 'segment_len': 8, 'seed': 1}
        with tempfile.TemporaryDirectory() as tmp:
            first = evaluate_text(model, data, evaluation, segment_path=Path(tmp) / 'a.npz')
            second = evaluate_text(model, data, evaluation, segment_path=Path(tmp) / 'b.npz')
            self.assertTrue(model.training)
            self.assertEqual(first, second)
            a, b = np.load(Path(tmp) / 'a.npz'), np.load(Path(tmp) / 'b.npz')
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
            self.assertAlmostEqual(first['Loss'], float(a['loss_sums'].sum()) / int(a['counts'].sum()))

    def test_text_and_sdq_resume_match_uninterrupted_weights_optimizer_generator_and_schedules(self):
        for task in ('text8', 'sdq'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = root / 'text.txt'
                data.write_text('abcdefghijklmnopqrstuvwxyz ' * 30)
                full = tiny_config(task, root / 'full', data)
                partial = tiny_config(task, root / 'resume', data)
                with contextlib.redirect_stdout(io.StringIO()):
                    main(full)
                    summary = main(partial, stop_after_updates=3)
                    self.assertFalse(summary['completed'])
                    partial['resume'] = 'auto'
                    main(partial)
                full_dir, a = load_checkpoint(full)
                resumed_dir, b = load_checkpoint(partial)
                self.assertEqual(a['step'], b['step'])
                self.assertEqual(a['updates'], b['updates'])
                self.assertEqual(a['lr'], b['lr'])
                self.assertEqual(a['reset'], b['reset'])
                self.assertEqual(a['curriculum'], b['curriculum'])
                self.assertEqual(a['eval_schedule'], b['eval_schedule'])
                for key in a['model']:
                    torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
                for index, values in a['optimizer']['state'].items():
                    for key, value in values.items():
                        torch.testing.assert_close(value, b['optimizer']['state'][index][key], rtol=0, atol=0)
                torch.testing.assert_close(a['recurrent_state']['h'], b['recurrent_state']['h'], rtol=0, atol=0)
                torch.testing.assert_close(a['rng']['torch'], b['rng']['torch'], rtol=0, atol=0)
                self.assertEqual(a['final_val'], b['final_val'])
                self.assertTrue((resumed_dir / 'previous.pt').exists())
                steps = [int(path.stem.split('-')[1]) for path in full_dir.glob('checkpoint-*.pt')]
                self.assertEqual(sorted(steps), [32, 48, 64, 80, 96])
                rows = [json.loads(line) for line in (resumed_dir / 'metrics.jsonl').read_text().splitlines()]
                self.assertEqual(rows[-1]['step'], 96)
                self.assertTrue((resumed_dir / 'final.pt').exists())

    def test_all_six_conditions_train_evaluate_and_checkpoint_on_both_tasks(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data = root / 'text.txt'
            data.write_text('abcdefghijklmnopqrstuvwxyz ' * 30)
            for task in ('text8', 'sdq'):
                for condition in CONDITIONS:
                    cfg = tiny_config(task, root / task / condition, data, condition=condition, recipe='clean')
                    cfg['n_steps'] = 32
                    result = main(cfg)
                    self.assertTrue(result['completed'])
                    self.assertEqual(result['skipped_updates'], 0)
                    directory, saved = load_checkpoint(cfg)
                    self.assertEqual(saved['format_version'], 1)
                    if task == 'sdq':
                        self.assertTrue((directory / 'fixed_gap_final.npz').exists())
                    else:
                        self.assertTrue((directory / 'test_final_segments.npz').exists())
            report = summarize(root)
            self.assertEqual(len(report['groups']), 2)
            for group in report['groups']:
                self.assertTrue(all(not missing for missing in group['missing_conditions_by_seed'].values()))
                for contrast in group['contrasts']:
                    self.assertEqual(contrast['n_matched_seeds'], 1)
                    self.assertIsNone(contrast['sample_sd'])
                    self.assertIsNone(contrast['nominal_95_interval'])

    def test_resume_rejects_changed_data_and_condition(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data = root / 'text.txt'
            data.write_text('abcdefghijklmnopqrstuvwxyz ' * 30)
            cfg = tiny_config('text8', root / 'run', data)
            main(cfg, stop_after_updates=2)
            directory, _ = load_checkpoint(cfg)
            cfg['resume'] = str(directory / 'latest.pt')
            changed = copy.deepcopy(cfg)
            changed['model'] = 'grnn_communication.static_mean'
            with self.assertRaisesRegex(ValueError, 'Checkpoint/config mismatch'):
                main(changed)
            data.write_text('abcdefghijklmnopqrstuvwxyz ' * 31)
            with self.assertRaisesRegex(ValueError, 'Dataset or generator changed'):
                main(cfg)


if __name__ == '__main__':
    unittest.main()
