"""Local correctness checks; no dataset, GPU, tracker or experiment launch required."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
import yaml

from knitwork.common.state_size import build, measure, state_floats
from knitwork.models.grnn_lru_core import GridRnn as DenseGridRnn, StaticMessagePassingLayer
from knitwork.models.grnn_lru_sparse import GridRnn, SparseCommunication, entmax15, shuffle_channels


CONFIG_DIR = Path(__file__).resolve().parents[1] / 'knitwork/exps/text/config/lru_mid'
VARIANTS = ('control', 'self_heavy', 'star', 'star_ring', 'clockwork', 'block', 'entmax', 'dynamic_hub')


def core(cls=GridRnn, **options):
    return cls(
        hidden_size=12, n_layers=2, n_columns=4, horizon=[1, 1000],
        dtype=torch.float32, device='cpu', **options,
    )


class SparseLruTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_control_is_original_including_initialization_and_backward(self):
        original = core(DenseGridRnn)
        torch.manual_seed(42)
        control = core()
        self.assertEqual(list(original.state_dict()), list(control.state_dict()))
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, control.state_dict()[name], rtol=0, atol=0)
        state = original.init_state(2)
        left, right = copy.deepcopy(state), copy.deepcopy(state)
        losses = [0, 0]
        for _ in range(8):
            x = torch.randn(2, 1, 12)
            y1, left, info1 = original(x, left, capture=True)
            y2, right, info2 = control(x, right, capture=True)
            torch.testing.assert_close(y1, y2, rtol=0, atol=0)
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
            for a, b in zip(info1['attn_weights'], info2['attn_weights']):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            losses[0] += y1.square().mean()
            losses[1] += y2.square().mean()
        for loss in losses:
            loss.backward()
        for a, b in zip(original.parameters(), control.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_star_masks_and_preserves_initial_self_and_input_mass(self):
        for topology in ('star', 'star_ring'):
            with self.subTest(topology=topology):
                original = StaticMessagePassingLayer(12, n_q=4, n_kv=5)
                old = original.pi_route_logits.detach().softmax(-1).clone()
                router = SparseCommunication(original, topology=topology)
                x = torch.randn(2, 5, 12)
                _, info = router(x[:, :4], x, x, True)
                weights = info['attn_weights']
                torch.testing.assert_close(weights.sum(-1), torch.ones(4))
                self.assertTrue((weights[~router.allowed] == 0).all())
                torch.testing.assert_close(weights.diagonal(), old.diagonal())
                torch.testing.assert_close(weights[:, 4], old[:, 4])
                torch.testing.assert_close(weights[0], old[0])
                expected = [0, 1, 4] if topology == 'star' else [0, 1, 2, 4]
                self.assertEqual(router.allowed[1].nonzero().flatten().tolist(), expected)
                self.assertTrue(router.allowed[3, 1] == (topology == 'star_ring'))

    def test_self_heavy_keeps_hub_and_input(self):
        original = StaticMessagePassingLayer(12, n_q=4, n_kv=5)
        hub = original.pi_route_logits.detach()[0].softmax(-1).clone()
        router = SparseCommunication(original, routing_init='self_heavy')
        weights = router.pi_route_logits.softmax(-1)
        torch.testing.assert_close(weights[0], hub)
        torch.testing.assert_close(weights.diagonal()[1:], torch.full((3,), 0.7))
        torch.testing.assert_close(weights[1:, 4], torch.full((3,), 0.15))

    def test_clockwork_freezes_state_and_skips_parameter_projections(self):
        model = core(update_periods=[1, 1, 2, 2], update_offsets=[0, 0, 0, 1])
        x = torch.randn(2, 1, 12)
        state = model.init_state(2)
        _, state, _ = model(x, state)
        self.assertEqual(state['clock_phase'], 0)
        for phase, frozen in ((0, 3), (1, 2), (0, 3)):
            previous = model.detach_state(state)
            model.zero_grad(set_to_none=True)
            with patch('torch.bmm', wraps=torch.bmm) as bmm:
                y, state, _ = model(x, previous)
            shapes = [tuple(call.args[1].shape) for call in bmm.call_args_list]
            self.assertEqual(shapes.count((3, 12, 24)), 2)
            self.assertEqual(shapes.count((3, 24, 12)), 2)
            torch.testing.assert_close(state['h'][:, frozen], previous['h'][:, frozen], rtol=0, atol=0)
            torch.testing.assert_close(state['outs'][:, :, frozen], previous['outs'][:, :, frozen], rtol=0, atol=0)
            y.square().sum().backward()
            self.assertEqual(model.cells.weight_in.grad[:, frozen].count_nonzero().item(), 0)
            self.assertEqual(model.cells.weight_out.grad[:, frozen].count_nonzero().item(), 0)
            self.assertEqual(getattr(model, f'active_{phase}').numel(), 3)
            self.assertEqual(state['clock_phase'], (phase + 1) % 2)

    def test_clockwork_reset_detach_and_accounting(self):
        model = core(update_periods=[1, 1, 2, 2], update_offsets=[0, 0, 0, 1])
        state = model.reset_state(None, bsz=2)
        _, state, _ = model(torch.randn(2, 1, 12), state)
        cleared = model.reset_state(state, torch.tensor([True, False]))
        self.assertEqual(cleared['clock_phase'], state['clock_phase'])
        self.assertEqual(cleared['h'][:, :, 0].count_nonzero().item(), 0)
        self.assertEqual(cleared['outs'][:, 0].count_nonzero().item(), 0)
        torch.testing.assert_close(cleared['h'][:, :, 1], state['h'][:, :, 1])
        self.assertEqual(model.detach_state(cleared)['clock_phase'], state['clock_phase'])
        self.assertFalse(model.detach_state(cleared)['h'].requires_grad)
        self.assertEqual(state_floats(model), 2 * 2 * 4 * 12 + 2 * 4 * 12)

    def test_grouped_projections_equal_expanded_dense_blocks(self):
        grouped = core(projection_groups=2)
        dense = core(DenseGridRnn)
        with torch.no_grad():
            dense.cells.log_r.copy_(grouped.cells.log_r)
            dense.cells.theta.copy_(grouped.cells.theta)
            dense.cells.weight_in.zero_()
            dense.cells.weight_out.zero_()
            for group in range(2):
                start, end = group * 6, (group + 1) * 6
                dense.cells.weight_in[:, :, start:end, start:end].copy_(grouped.cells.weight_in[:, :, group, :, :6])
                dense.cells.weight_in[:, :, start:end, 12 + start:12 + end].copy_(grouped.cells.weight_in[:, :, group, :, 6:])
                dense.cells.weight_out[:, :, start:end, start:end].copy_(grouped.cells.weight_out[:, :, group, :6])
                dense.cells.weight_out[:, :, 12 + start:12 + end, start:end].copy_(grouped.cells.weight_out[:, :, group, 6:])
        x, h = torch.randn(4, 2, 12), torch.randn(4, 2, 24)
        for layer in range(2):
            y1, h1 = grouped.cells(layer, x, h)
            y2, h2 = dense.cells(layer, x, h)
            torch.testing.assert_close(h1, h2)
            torch.testing.assert_close(y1, y2)

    def test_shuffle_mixes_groups_without_changing_columns(self):
        x = torch.arange(24).reshape(2, 2, 6)
        shuffled = shuffle_channels(x, 2)
        torch.testing.assert_close(shuffled, x[..., [0, 3, 1, 4, 2, 5]])

    def test_entmax_exact_zeros_normalization_and_gradcheck(self):
        logits = torch.tensor([[0.0, 0.2, -5.0, 0.8]], dtype=torch.float64, requires_grad=True)
        weights = entmax15(logits)
        torch.testing.assert_close(weights.sum(-1), torch.ones(1, dtype=torch.float64))
        self.assertEqual(weights[0, 2].item(), 0)
        self.assertTrue(torch.autograd.gradcheck(entmax15, (logits,)))
        (weights.square().sum()).backward()
        self.assertTrue(logits.grad.isfinite().all())

    def test_entmax_hub_stays_dense_and_inputs_live_at_initialization(self):
        model = core(routing_activation='entmax15', routing_init='soft')
        _, _, info = model(torch.randn(2, 1, 12), model.init_state(2), capture=True)
        weights = info['attn_weights'][0]
        self.assertTrue((weights[0] > 0).all())
        self.assertTrue((weights[:, -1] > 0).all())

    def test_dynamic_hub_changes_with_query_only(self):
        model = core(topology='star', hub_query_size=8)
        router = model.comm[0]
        keys = torch.randn(2, 5, 12)
        q = torch.randn(2, 4, 12)
        out1, info1 = router(q, keys, keys, True)
        _, info2 = router(q + 4, keys, keys, True)
        self.assertFalse(torch.allclose(info1['attn_weights'][0], info2['attn_weights'][0]))
        torch.testing.assert_close(info1['attn_weights'][1:], info2['attn_weights'][1:], rtol=0, atol=0)
        out1.square().sum().backward()
        self.assertGreater(router.query.weight.grad.abs().sum().item(), 0)
        self.assertGreater(router.key.weight.grad.abs().sum().item(), 0)
        self.assertEqual(state_floats(model), 252)

    def test_invalid_options_fail_early(self):
        for options in (
                {'projection_groups': 5}, {'update_periods': [2, 1, 1, 1]},
                {'update_periods': [1, 1, 0, 1]}, {'update_offsets': [0, 0, 1, 0]},
                {'topology': 'missing'}, {'topology': 'star', 'routing_activation': 'entmax15'},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                core(**options)

    def test_all_mid_configs_forward_backward_and_common_protocol(self):
        baseline_protocol = None
        for name in VARIANTS:
            with self.subTest(variant=name):
                config = yaml.safe_load((CONFIG_DIR / f'{name}.yaml').read_text())
                key = config['model'].replace('.', '_')
                protocol = {k: v for k, v in config.items() if k not in ('model', key)}
                if baseline_protocol is None:
                    baseline_protocol = protocol
                self.assertEqual(protocol, baseline_protocol)
                self.assertEqual(config[key]['hidden_size'], 180)
                params, floats, _ = measure(config, key)
                expected_params = 531063 if name == 'block' else 1055223 if name == 'dynamic_hub' else 1049463
                self.assertEqual(params, expected_params)
                expected_state = 4320 if name == 'clockwork' else 3960 if name == 'dynamic_hub' else 3600
                self.assertEqual(floats, expected_state)
                model = build(config, key)
                state = model.init_state(2)
                loss = 0
                for _ in range(8):
                    logits, state, _ = model(torch.randint(27, (2, 1)), state)
                    loss += F.cross_entropy(logits, torch.randint(27, (2,)))
                self.assertTrue(loss.isfinite())
                loss.backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(parameter.grad.isfinite().all())

    def test_compile_fullgraph_matches_eager_for_all_variants_and_phases(self):
        for name in VARIANTS:
            with self.subTest(variant=name):
                config = yaml.safe_load((CONFIG_DIR / f'{name}.yaml').read_text())
                key = config['model'].replace('.', '_')
                model = build(config, key, {'hidden_size': 12})
                compiled = torch.compile(model, backend='eager', fullgraph=True)
                eager_state = model.init_state(2)
                compiled_state = copy.deepcopy(eager_state)
                loss = 0
                for _ in range(4):
                    tokens = torch.randint(27, (2, 1))
                    eager_y, eager_state, _ = model(tokens, eager_state)
                    compiled_y, compiled_state, _ = compiled(tokens, compiled_state)
                    torch.testing.assert_close(eager_y, compiled_y)
                    torch.testing.assert_close(eager_state['h'], compiled_state['h'])
                    loss += compiled_y.square().mean()
                loss.backward()
                self.assertTrue(all(p.grad.isfinite().all() for p in model.parameters()))
                torch._dynamo.reset()

    def test_feedback_norm_matches_updated_control_and_custom_forward_paths(self):
        original = core(DenseGridRnn, fb_norm=True)
        control = core(fb_norm=True)
        control.load_state_dict(original.state_dict())
        state = original.init_state(2)
        left, right = copy.deepcopy(state), copy.deepcopy(state)
        x = torch.randn(2, 1, 12)
        for _ in range(4):
            y1, left, _ = original(x, left)
            y2, right, _ = control(x, right)
            torch.testing.assert_close(y1, y2, rtol=0, atol=0)
            torch.testing.assert_close(left['out'], right['out'], rtol=0, atol=0)
        for options in (
                {'projection_groups': 2, 'shuffle_between_layers': True},
                {'update_periods': [1, 1, 2, 2], 'update_offsets': [0, 0, 0, 1]},
        ):
            with self.subTest(options=options):
                model = core(fb_norm=True, **options)
                state = model.init_state(2)
                for _ in range(4):
                    y, state, _ = model(x, state)
                    torch.testing.assert_close(y, state['outs'][-1][:, 0])
                    torch.testing.assert_close(state['out'], F.rms_norm(state['outs'][-1], (12,)))
                reset = model.reset_state(state, torch.tensor([True, False]))
                torch.testing.assert_close(reset['out'], F.rms_norm(reset['outs'][-1], (12,)))
                self.assertEqual(reset['out'][0].count_nonzero().item(), 0)


if __name__ == '__main__':
    unittest.main()
