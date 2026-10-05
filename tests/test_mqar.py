import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from knitwork.common.config import load_config
from knitwork.common.utils import count_learnable_params
from knitwork.exps.mqar.run import allocated_state_floats, evaluate, main, model_state_info, sequence_objective
from knitwork.gens.mqar import IGNORE_INDEX, build_splits, epoch_batches, generate_mqar
from knitwork.models.utils import build_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'knitwork/exps/mqar/config'


def make_model(kind='grnn_lru', **overrides):
    cfg = dict(hidden_size=12, n_layers=2)
    if kind == 'grnn':
        cfg.update(n_columns=4, n_attn_heads=4, ln_msg=True, bank=2, mha=2, noise_std=0.05)
    elif kind != 'rnn':
        cfg.update(n_columns=4, horizon=[1, 1000], fb_norm=True)
    return build_model(
        wrapper_type='token', rnn_type=kind, rnn_cfg=cfg | overrides,
        wrapper_cfg=dict(input_size=64, output_size=64, dtype=torch.float32, device='cpu'),
    )


class MQARTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_pinned_zoology_reference_golden_arrays(self):
        # Hashes from the unmodified pinned upstream function with torch.manual_seed(seed).
        # This checks label alignment, weighted sampling, RNG draw order, and filler distribution.
        cases = [
            (32, 16, 3, 3, 7, False, 'd6065198adef13d253d49adddac4a99ef0d1d5b26b94e92f5d5328fe4179000c'),
            (32, 16, 3, 3, 7, True, 'a725360766c26453483f63fe90dca2da88c43ce5bd94806b95d002769608a359'),
            (8192, 256, 64, 4, 100, True, '393bdc3ffea811a316d5fedeb7ea490a94c62d8ae6d004e089b54b74912a3148'),
        ]
        for vocab, length, pairs, examples, seed, random, expected in cases:
            with self.subTest(vocab=vocab, random=random):
                data = generate_mqar(
                    vocab_size=vocab, input_seq_len=length, num_kv_pairs=pairs,
                    num_examples=examples, seed=seed, random_non_queries=random,
                )
                digest = hashlib.sha256(data['inputs'].numpy().tobytes() + data['labels'].numpy().tobytes()).hexdigest()
                self.assertEqual(digest, expected)

    def test_query_lookup_no_answer_leakage_and_rng_isolation(self):
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        data = generate_mqar(
            vocab_size=64, input_seq_len=32, num_kv_pairs=4, num_examples=12,
            seed=123, random_non_queries=False,
        )
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        np.testing.assert_array_equal(numpy_state[1], after[1])
        self.assertEqual(numpy_state[2:], after[2:])
        for inputs, labels in zip(data['inputs'], data['labels']):
            mapping = dict(zip(inputs[:8:2].tolist(), inputs[1:8:2].tolist()))
            self.assertEqual(len(mapping), 4)
            self.assertEqual(len(set(mapping.values())), 4)
            self.assertTrue(all(0 < key < 32 <= value < 64 for key, value in mapping.items()))
            query_positions = torch.where(labels != IGNORE_INDEX)[0]
            self.assertEqual(len(query_positions), 4)
            self.assertEqual(set(inputs[query_positions].tolist()), set(mapping))
            for position in query_positions.tolist():
                self.assertGreaterEqual(position, 8)
                self.assertEqual(position % 2, 0)
                self.assertEqual(labels[position].item(), mapping[inputs[position].item()])
                self.assertEqual(inputs[position + 1].item(), 0)
            self.assertTrue((labels[:8] == IGNORE_INDEX).all())
            self.assertTrue((inputs[8:][labels[8:] == IGNORE_INDEX] == 0).all())

    def test_invalid_generator_and_overlapping_split_seeds(self):
        defaults = dict(vocab_size=64, input_seq_len=32, num_kv_pairs=4, num_examples=2, seed=1)
        for override in (
            dict(input_seq_len=31), dict(input_seq_len=12), dict(vocab_size=32),
            dict(num_kv_pairs=0), dict(num_examples=0), dict(power_a=0),
            dict(power_a=float('nan')), dict(seed=-1), dict(num_examples=1.5),
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                generate_mqar(**(defaults | override))
        config = load_config(CONFIG_DIR / 'smoke.yaml')['data']
        config['val_seed'] = config['train_seed'] + 1
        with self.assertRaises(ValueError):
            build_splits(config)

    def test_epoch_covers_every_example_once_without_padding(self):
        splits = build_splits(load_config(CONFIG_DIR / 'smoke.yaml')['data'])
        first = list(epoch_batches(splits['train'], 3, np.random.default_rng(8)))
        second = list(epoch_batches(splits['train'], 3, np.random.default_rng(8)))
        for (index, indices), (other_index, other_indices) in zip(first, second):
            self.assertEqual(index, other_index)
            np.testing.assert_array_equal(indices, other_indices)
        for index, segment in enumerate(splits['train']):
            seen = np.concatenate([indices for batch_index, indices in first if batch_index == index])
            np.testing.assert_array_equal(np.sort(seen), np.arange(len(segment['inputs'])))

    def test_store_value_receives_gradient_from_late_query(self):
        # Distinct tokens and query-only loss: a gradient on token 40 must traverse stored state.
        inputs = torch.tensor([[3, 40, 0, 0, 0, 0, 3, 0]])
        labels = torch.tensor([[-100, -100, -100, -100, -100, -100, 40, -100]])
        for kind, options in (
            ('grnn_lru', {}), ('rnn', {}), ('grnn', {}),
            ('grnn_lru_sparse', {'topology': 'star'}),
            ('grnn_lru_sparse', {'update_periods': [1, 1, 2, 2], 'update_offsets': [0, 0, 0, 1]}),
            ('grnn_lru_sparse', {'projection_groups': 2, 'shuffle_between_layers': True}),
            ('grnn_lru_sparse', {'routing_activation': 'entmax15', 'routing_init': 'soft'}),
            ('grnn_lru_sparse', {'hub_query_size': 8}),
            ('grnn_lru_sparse', {'topology': 'star', 'hub_query_size': 8, 'hub_layers': [1]}),
            ('grnn_lru_sparse', {'topology': 'star', 'hub_query_size': 8, 'hub_query_init': 'zero'}),
            ('grnn_lru_sparse', {'topology': 'star', 'hub_query_size': 8, 'routing_init': 'self_heavy'}),
            ('grnn_lru_sparse', {'projection_groups': 2, 'dense_hub': True, 'shuffle_between_layers': True}),
            ('grnn_lru_sparse', {'update_periods': [1, 1, 1, 2]}),
            ('grnn_lru_sparse', {'update_periods': [1, 1, 2, 2], 'update_offsets': [0, 0, 0, 1], 'clockwork_mode': 'output'}),
        ):
            with self.subTest(kind=kind, options=options):
                model = make_model(kind, **options)
                loss, _, _ = sequence_objective(model, inputs, labels)
                loss.backward()
                self.assertTrue(loss.isfinite())
                self.assertGreater(model.embedding.weight.grad[40].abs().sum().item(), 0)
                self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))

    def test_eval_query_denominator_slices_exact_match_and_rng(self):
        model = make_model()
        # Constant prediction 40 permits hand-calculated metrics, with many ignored fillers.
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
            model.head.bias[40] = 2
        segments = [
            {'key': 'T8_K1', 'inputs': torch.tensor([[3, 40, 0, 0, 0, 0, 3, 0]]),
             'labels': torch.tensor([[-100, -100, -100, -100, -100, -100, 40, -100]])},
            {'key': 'T8_K2', 'inputs': torch.tensor([[3, 40, 4, 41, 3, 0, 4, 0]]),
             'labels': torch.tensor([[-100, -100, -100, -100, 40, -100, 41, -100]])},
        ]
        rng = torch.get_rng_state().clone()
        metrics = evaluate(model, segments, batch_size=2, device=torch.device('cpu'))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(metrics, evaluate(model, segments, batch_size=2, device=torch.device('cpu')))
        self.assertEqual(metrics['Acc'], 2 / 3)
        self.assertEqual(metrics['Exact_match'], 1 / 2)
        self.assertEqual(metrics['queries'], 3)
        self.assertEqual(metrics['by_segment']['T8_K1']['Acc'], 1)
        self.assertEqual(metrics['by_segment']['T8_K2']['Acc'], 1 / 2)
        expected_loss = np.log(np.exp(2) + 63) - 4 / 3
        self.assertAlmostEqual(metrics['Loss'], expected_loss, places=5)

    def test_mid_parameter_counts_and_yaml_variants(self):
        config = load_config(CONFIG_DIR / 'mid.yaml')
        for kind, key, expected in (
            ('grnn_lru', 'grnn_lru_L2C4', 3997028), ('rnn', 'rnn_L2', 4002000),
        ):
            model = build_model(
                wrapper_type='token', rnn_type=kind, rnn_cfg=config[key],
                wrapper_cfg=dict(input_size=8192, output_size=8192, dtype=torch.float32, device='cpu'),
            )
            self.assertEqual(count_learnable_params(model), expected)
        for variant in ('star', 'star_ring', 'clockwork', 'block', 'entmax', 'dynamic_hub', 'self_heavy'):
            self.assertEqual(config[f'grnn_lru_sparse_{variant}']['hidden_size'], 180)
        self.assertEqual(config['grnn_lru_sparse_dynamic_hub']['topology'], 'star')

    def test_mqar_suite_protocol_budget_controls_and_all_runner_paths(self):
        base = load_config(CONFIG_DIR / 'mid.yaml')
        smoke = load_config(CONFIG_DIR / 'smoke.yaml')
        paths = sorted((CONFIG_DIR / 'lru_mid').glob('*.yaml'))
        self.assertEqual(len(paths), 20)
        expected = {
            'grnn': 3935344, 'grnn_core_mid': 3279544, 'grnn_lru': 3997028,
            'gru': 4002000, 'dense_small': 3478100, 'block': 3478628,
            'block_wide': 4019684, 'block_dense_hub': 3608228,
        }
        with tempfile.TemporaryDirectory() as directory:
            for path in paths:
                with self.subTest(variant=path.stem):
                    cfg = load_config(path)
                    self.assertEqual(cfg['data'], base['data'])
                    self.assertEqual(cfg['training'], base['training'])
                    self.assertEqual(cfg['eval'], base['eval'])
                    self.assertIsNone(cfg['log']['logger'])
                    model_key = cfg['model'].replace('.', '_')
                    self.assertEqual(cfg[model_key], base[model_key])
                    if path.stem in expected:
                        model = build_model(
                            wrapper_type='token', rnn_type=cfg['model'].split('.', 1)[0],
                            rnn_cfg=cfg[model_key],
                            wrapper_cfg=dict(input_size=8192, output_size=8192, dtype=torch.float32, device='cpu'),
                        )
                        self.assertEqual(count_learnable_params(model), expected[path.stem])
                    # Preserve each variant's mechanism, but reduce data and width for the pipeline check.
                    cfg['data'] = copy.deepcopy(smoke['data'])
                    cfg['training'] = copy.deepcopy(smoke['training'])
                    cfg['eval'] = copy.deepcopy(smoke['eval'])
                    cfg['device'] = 'cpu'
                    cfg['output_dir'] = directory
                    cfg[model_key]['hidden_size'] = 12
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = main(cfg)
                    self.assertEqual(result['training_updates'], 2)
                    self.assertEqual(result['test']['queries'], 24)
                    self.assertTrue(np.isfinite(result['test']['Loss']))
            self.assertEqual(len(list(Path(directory).iterdir())), 20)

    def test_allocated_state_counts_grnn_views_once_and_lru_caches(self):
        grnn = make_model('grnn')
        lru = make_model()
        self.assertEqual(allocated_state_floats(grnn.init_state(1)), 2 * 4 * 12)
        self.assertEqual(allocated_state_floats(lru.init_state(1)), 2 * 2 * 4 * 12 + 2 * 4 * 12)
        _, state, _ = lru(torch.tensor([[3]]), lru.init_state(1))
        self.assertEqual(allocated_state_floats(state), 2 * 2 * 4 * 12 + 2 * 4 * 12 + 4 * 12)
        rng = torch.get_rng_state().clone()
        info = model_state_info(lru, torch.device('cpu'))
        self.assertEqual(info['initial_allocated_state_floats'], 288)
        self.assertEqual(info['allocated_state_floats'], 336)
        self.assertEqual(info['carried_state_floats'], 240)
        self.assertEqual(model_state_info(grnn, torch.device('cpu'))['allocated_state_floats'], 96)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_batched_objective_matches_per_step_reference_and_skips_nonfinite_steps(self):
        data = generate_mqar(vocab_size=64, num_examples=6, input_seq_len=24, num_kv_pairs=4, seed=1)
        inputs, labels = data['inputs'], data['labels']

        def per_step(model):
            state, losses, correct = model.init_state(len(inputs)), [], 0
            batch_first = getattr(model.rnn, 'batch_first', False)
            for step in range(inputs.shape[1]):
                tokens = inputs[:, step]
                feats, state, _ = model.rnn(model.embedding(tokens[:, None] if batch_first else tokens[None]), state)
                mask = labels[:, step] != IGNORE_INDEX
                if mask.any():
                    logits = model.head(feats[mask])
                    losses.append(torch.nn.functional.cross_entropy(logits, labels[mask, step], reduction='sum'))
                    correct += int((logits.argmax(-1) == labels[mask, step]).sum())
            return torch.stack(losses).sum(), correct

        for kind in ('grnn_lru', 'rnn'):
            with self.subTest(kind=kind):
                torch.manual_seed(0)
                ref_model = make_model(kind)
                ref_loss, ref_correct = per_step(ref_model)
                ref_loss.backward()
                torch.manual_seed(0)
                model = make_model(kind)
                loss, correct, _ = sequence_objective(model, inputs, labels)
                loss.backward()
                self.assertTrue(torch.allclose(loss, ref_loss, rtol=1e-5))
                self.assertEqual(int(correct), ref_correct)
                for a, b in zip(model.parameters(), ref_model.parameters()):
                    self.assertTrue(torch.allclose(a.grad, b.grad, rtol=1e-4, atol=1e-6))

        # a non-finite gradient must be skipped (and counted), not abort the run
        config = load_config(CONFIG_DIR / 'smoke.yaml')
        config['training']['max_updates'] = 2
        with tempfile.TemporaryDirectory() as directory:
            config['output_dir'] = directory
            from knitwork.exps.mqar import run as runner
            original = torch.nn.utils.clip_grad_norm_
            calls = []

            def flaky_clip(*args, **kwargs):
                norm = original(*args, **kwargs)
                calls.append(1)
                return torch.tensor(float('inf')) if len(calls) == 1 else norm

            torch.nn.utils.clip_grad_norm_ = flaky_clip
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = runner.main(config)
            finally:
                torch.nn.utils.clip_grad_norm_ = original
            self.assertEqual(result['skipped_steps'], 1)
            self.assertEqual(result['training_updates'], 2)

    def test_runner_checkpoint_selection_and_gru_control(self):
        config = load_config(CONFIG_DIR / 'smoke.yaml')
        original = copy.deepcopy(config)
        with tempfile.TemporaryDirectory() as directory:
            config['output_dir'] = directory
            for name in ('grnn_lru.L2C4', 'rnn.L2'):
                config['model'] = name
                before = copy.deepcopy(config)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = main(config)
                self.assertEqual(config, before)
                self.assertEqual(result['training_updates'], 2)
                self.assertIn(result['best_update'], (1, 2))
                self.assertEqual(result['test']['queries'], 24)
                self.assertEqual(result['training_tokens'], 8 * result['training_queries'])
                self.assertTrue(np.isfinite(result['test']['Loss']))
                self.assertGreater(result['training_seconds'], 0)
            runs = list(Path(directory).iterdir())
            self.assertEqual(len(runs), 2)
            for run in runs:
                self.assertEqual({path.name for path in run.iterdir()}, {'config.json', 'best.pt', 'last.pt', 'metrics.json', 'status.json'})
                metrics = json.loads((run / 'metrics.json').read_text())
                checkpoint = torch.load(run / 'best.pt', weights_only=True)
                self.assertEqual(checkpoint['update'], metrics['best_update'])
                self.assertEqual(checkpoint['validation']['Loss'], metrics['best_val_loss'])
                self.assertEqual(metrics['protocol']['bptt'], 'full_sequence')
                self.assertEqual(len(metrics['test']['by_segment']), 2)
        self.assertEqual(original['model'], 'grnn_lru.L2C4')


if __name__ == '__main__':
    unittest.main()
