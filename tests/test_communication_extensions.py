import contextlib
import copy
import importlib
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import torch
import yaml

from test_communication_controls import CONFIG, tiny_config
from knitwork.exps.communication.protocol import accounting, config_fingerprint, make_model, resolve_config, tensor_digest
from knitwork.exps.communication.queue import prepare_jobs, reconcile_jobs, run_directory
from knitwork.exps.communication.suite import profile_dense_forward
from knitwork.models.grnn_communication import GridRnn as FourColumnCore
from knitwork.models.grnn_single_column import GridRnn as SingleColumnCore


def extension_config(task, variant):
    return yaml.safe_load((CONFIG / f'{task}_none_{variant}.yaml').read_text())


class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_single_column_matches_active_column_of_original_none(self):
        kwargs = dict(hidden_size=12, n_layers=2, n_attn_heads=4, dtype=torch.float32, device='cpu', routing='none')
        with contextlib.redirect_stdout(io.StringIO()):
            original = FourColumnCore(n_columns=4, **kwargs)
            single = SingleColumnCore(n_columns=1, **kwargs)
        original_weights = original.state_dict()
        weights = single.state_dict()
        for name in weights:
            source = original_weights[name]
            if name.startswith('cells.'):
                source = source[:, :1]
            elif name.endswith('.ids'):
                indices = [0, 4] if name.startswith('attn.0.') else [0]
                source = source[:, indices]
            elif name.endswith('.pi_logtemp'):
                source = source[:, :, :1]
            weights[name] = source.clone()
        single.load_state_dict(weights)
        state_original = original.init_state(3)
        state_single = single.init_state(3)
        for _ in range(5):
            x = torch.randn(1, 3, 12)
            expected, state_original, _ = original(x, state_original)
            actual, state_single, info = single(x, state_single)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(state_single['h'], state_original['h'][:, :1], rtol=1e-5, atol=1e-6)
            self.assertTrue(all(torch.isfinite(value) for values in info.values() for value in values))
            self.assertEqual(info['comm_entropy'][1].item(), 0)
        actual.square().sum().backward()
        self.assertEqual(tuple(state_single['h'].shape), (2, 1, 3, 12))
        for layer in range(2):
            self.assertGreater(single.cells.weight_n.grad[layer].abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in single.parameters()))

    def test_configs_counts_state_width_selection_and_profile(self):
        expected = {
            ('text8', 'single_column'): (1057493, 456),
            ('sdq', 'single_column'): (1062264, 456),
            ('text8', 'concat'): (1061515, 1088),
            ('sdq', 'concat'): (1057418, 1088),
        }
        for (task, variant), (parameters, state) in expected.items():
            cfg = extension_config(task, variant)
            module = importlib.import_module(cfg['training_runner'])
            cfg = module.prepare_config(cfg)
            with module.registered_model(), contextlib.redirect_stdout(io.StringIO()):
                model = make_model(resolve_config(cfg))
                counts = module.accounting(model) if variant == 'concat' else accounting(model)
                self.assertEqual(counts['parameters_nominal_trainable'], parameters)
                self.assertEqual(counts['state_scalars'], state)
                self.assertEqual(counts['state_scalars_with_path_to_readout'], state)
                self.assertEqual(counts['dense_forward_flops_per_token'], profile_dense_forward(model))
                adapters = sum(p.numel() for block in (model.embedding, model.head) for p in block.parameters())
                self.assertEqual(counts['parameters_adapters'], adapters)
                if variant == 'single_column':
                    target = cfg['parameter_target']['nominal_trainable']
                    for width in (224, 232):
                        adjacent = copy.deepcopy(cfg)
                        adjacent['topologies']['L2C1']['hidden_size'] = width
                        neighbor = accounting(make_model(resolve_config(adjacent)))['parameters_nominal_trainable']
                        self.assertLess(abs(parameters - target), abs(neighbor - target))

    def test_concat_starts_as_mean_but_has_independent_column_weights(self):
        cfg = extension_config('text8', 'concat')
        module = importlib.import_module(cfg['training_runner'])
        base = yaml.safe_load((CONFIG / 'text8.yaml').read_text())
        base['model'] = 'grnn_communication.none_mean'
        with contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(42)
            mean = make_model(resolve_config(base))
            torch.manual_seed(42)
            with module.registered_model():
                concat = make_model(resolve_config(module.prepare_config(cfg)))
        self.assertEqual(tensor_digest(mean.rnn.state_dict()), tensor_digest(concat.rnn.state_dict()))
        h = torch.randn(2, 4, 3, 136, requires_grad=True)
        state = {'h': h, 'out': h[-1]}
        tokens = torch.tensor([[0, 1, 2]])
        expected, expected_state, _ = mean(tokens, state)
        actual, actual_state, _ = concat(tokens, state)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_state['h'], expected_state['h'], rtol=0, atol=0)
        self.assertEqual(concat.head.in_features, 544)
        torch.nn.functional.cross_entropy(actual, torch.tensor([0, 1, 2])).backward(retain_graph=True)
        blocks = concat.head.weight.grad.reshape(27, 4, 136)
        self.assertTrue((blocks.abs().sum(dim=(0, 2)) > 0).all())
        self.assertFalse(torch.allclose(blocks[:, 0], blocks[:, 1]))
        grad = torch.autograd.grad(actual_state['h'][-1, 0].square().sum(), h)[0]
        self.assertEqual(grad[:, 1:].count_nonzero().item(), 0)

    def test_full_training_and_exact_resume_for_both_extensions_and_tasks(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            dataset = root / 'text8.txt'
            dataset.write_text('abcdefghijklmnopqrstuvwxyz ' * 100)
            for task in ('text8', 'sdq'):
                for variant in ('single_column', 'concat'):
                    cfg = tiny_config(task, root / f'{task}-{variant}-full', dataset, condition='none_column0', recipe='clean')
                    extension = extension_config(task, variant)
                    module = importlib.import_module(extension['training_runner'])
                    cfg.update(model=extension['model'], topology=extension['topology'])
                    cfg['topologies'] = extension['topologies']
                    if variant == 'single_column':
                        cfg['topologies']['L2C1']['hidden_size'] = 12
                    else:
                        cfg['readout_initialization'] = 'mean_equivalent'
                        cfg['grnn_communication_none_concat'] = extension['grnn_communication_none_concat']
                        cfg['parameter_range'] = [1, 2000000]
                    interrupted = copy.deepcopy(cfg)
                    interrupted['output_dir'] = str(root / f'{task}-{variant}-resumed')
                    self.assertTrue(module.main(cfg)['completed'])
                    self.assertFalse(module.main(interrupted, stop_after_updates=3)['completed'])
                    interrupted['resume'] = 'auto'
                    self.assertTrue(module.main(interrupted)['completed'])
                    with module.registered_model():
                        resolved = resolve_config(module.prepare_config(cfg))
                    run_id = resolved['run_id']
                    full = torch.load(Path(cfg['output_dir']) / run_id / 'latest.pt', weights_only=False)
                    resumed = torch.load(Path(interrupted['output_dir']) / run_id / 'latest.pt', weights_only=False)
                    self.assertEqual(tensor_digest(full['model']), tensor_digest(resumed['model']))
                    self.assertEqual(full['skipped'], 0)
                    self.assertEqual(resumed['skipped'], 0)
                    torch.testing.assert_close(full['recurrent_state']['h'], resumed['recurrent_state']['h'], rtol=0, atol=0)
                    self.assertEqual(full['config_fingerprint'], config_fingerprint(resolved))
                    self.assertTrue(resolved['implementation']['source_hashes'])
                    tampered = module.prepare_config(cfg)
                    tampered['implementation']['source_hashes']['changed'] = 'invalid'
                    with self.assertRaisesRegex(ValueError, 'implementation changed'):
                        module.main(tampered)

    def test_extra_jobs_follow_original_matrix_and_use_separate_artifacts(self):
        cfg = yaml.safe_load((CONFIG / 'local_queue.yaml').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            cfg['additional_configs'] = []
            for variant in ('single_column', 'concat'):
                for task in ('text8', 'sdq'):
                    extra = extension_config(task, variant)
                    extra['output_dir'] = str(output / 'communication_extensions_1m')
                    path = output / f'{task}-{variant}.yaml'
                    path.write_text(yaml.safe_dump(extra))
                    cfg['additional_configs'].append(str(path))
            cfg['text8_path'] = str(output / 'text8.txt')
            Path(cfg['text8_path']).write_text('abcdefghijklmnopqrstuvwxyz ' * 100)
            jobs = prepare_jobs(cfg, output)
            self.assertEqual(len(jobs), 16)
            self.assertTrue(all(job['config']['topology'] == 'L2C4' for job in jobs[:12]))
            self.assertEqual([job['config']['topology'] for job in jobs[12:]], ['L2C1', 'L2C1', 'L2C4', 'L2C4'])
            self.assertTrue(all('none_concat' in job['run_id'] for job in jobs[14:]))
            self.assertTrue(all(Path(job['config']['output_dir']) == output for job in jobs[:12]))
            self.assertTrue(all('communication_extensions_1m' in str(run_directory(job, output)) for job in jobs[12:]))
            reconcile_jobs(jobs, {}, output)
            self.assertTrue(all(job['status'] == 'pending' for job in jobs))

    def test_real_queue_dispatches_extensions_and_reads_their_output_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = root / 'empty-matrix.yaml'
            matrix.write_text(yaml.safe_dump({'tasks': {}, 'topologies': [], 'recipes': [], 'seeds': [], 'conditions': []}))
            cfg = yaml.safe_load((CONFIG / 'local_queue.yaml').read_text())
            cfg.update(matrix=str(matrix), output_dir=str(root / 'queue'), logger='local', device='cpu', launch_spacing_seconds=0, additional_configs=[])
            for variant in ('single_column', 'concat'):
                extra = tiny_config('sdq', root / 'extensions', condition='none_column0', recipe='clean')
                template = extension_config('sdq', variant)
                extra.update(model=template['model'], topology=template['topology'], training_runner=template['training_runner'])
                extra['topologies'] = template['topologies']
                if variant == 'single_column':
                    extra['topologies']['L2C1']['hidden_size'] = 12
                else:
                    extra['readout_initialization'] = 'mean_equivalent'
                    extra['grnn_communication_none_concat'] = template['grnn_communication_none_concat']
                    extra['parameter_range'] = [1, 2000000]
                path = root / f'{variant}.yaml'
                path.write_text(yaml.safe_dump(extra))
                cfg['additional_configs'].append(str(path))
            path = root / 'queue.yaml'
            path.write_text(yaml.safe_dump(cfg))
            result = subprocess.run([sys.executable, '-m', 'knitwork.exps.communication.queue', str(path)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            state = json.loads((root / 'queue/_queue/status.json').read_text())
            self.assertEqual(state['status'], 'complete', result.stdout)
            self.assertIsNone(state['hard_stop_epoch'])
            for job in state['jobs']:
                self.assertEqual(job['status'], 'complete', result.stdout)
                run_dir = run_directory(job, root / 'queue')
                summary = json.loads((run_dir / 'summary.json').read_text())
                self.assertTrue(summary['completed'])
                self.assertEqual(summary['tokens'], 96)


if __name__ == '__main__':
    unittest.main()
