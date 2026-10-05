"""Behavioral ablations for the follow-up LRU experiment series."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from knitwork.common.config import load_config
from knitwork.common.state_size import build, measure, state_floats
from knitwork.models.grnn_lru_core import GridRnn as DenseGridRnn
from knitwork.models.grnn_lru_sparse import GridRnn


CONFIG_DIR = Path(__file__).resolve().parents[1] / 'knitwork/exps/text/config/lru_mid'
COUNTS = {
    'dynamic_hub_top': (1052343, 3780),
    'dynamic_hub_zero': (1055223, 3960),
    'dynamic_hub_self_heavy': (1055223, 3960),
    'dense_small': (533311, 2560),
    'block_wide': (1050099, 5080),
    'block_dense_hub': (660663, 3600),
    'clockwork_light': (1049463, 4320),
    'clockwork_output': (1049463, 4320),
}


def core(**options):
    return GridRnn(
        hidden_size=12, n_layers=2, n_columns=4, horizon=[1, 1000],
        fb_norm=True, dtype=torch.float32, device='cpu', **options,
    )


class LruFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_dynamic_layer_selection_and_required_query_cache(self):
        model = core(topology='star', hub_query_size=8, hub_layers=[1])
        self.assertFalse(hasattr(model.comm[0], 'query'))
        self.assertTrue(hasattr(model.comm[1], 'query'))
        for layer in range(2):
            count = 5 if layer == 0 else 4
            keys, query = torch.randn(2, count, 12), torch.randn(2, 4, 12)
            _, first = model.comm[layer](query, keys, keys, True)
            _, second = model.comm[layer](query + 3, keys, keys, True)
            if layer == 0:
                torch.testing.assert_close(first['attn_weights'], second['attn_weights'], rtol=0, atol=0)
            else:
                self.assertFalse(torch.allclose(first['attn_weights'][0], second['attn_weights'][0]))
                torch.testing.assert_close(first['attn_weights'][1:], second['attn_weights'][1:], rtol=0, atol=0)
        self.assertEqual(state_floats(model), 252)
        model.fb_norm = False
        self.assertEqual(state_floats(model), 240)

    def test_zero_query_matches_static_star_including_wrapper_initialization(self):
        config = load_config(CONFIG_DIR / 'star.yaml')
        key = config['model'].replace('.', '_')
        star = build(config, key, {'hidden_size': 12})
        rng_after_star = torch.get_rng_state().clone()
        torch.manual_seed(42)
        dynamic = build(config, key, {'hidden_size': 12, 'hub_query_size': 8, 'hub_query_init': 'zero'})
        torch.testing.assert_close(torch.get_rng_state(), rng_after_star, rtol=0, atol=0)
        for name, parameter in star.state_dict().items():
            torch.testing.assert_close(parameter, dynamic.state_dict()[name], rtol=0, atol=0)
        state = star.init_state(2)
        left, right = copy.deepcopy(state), copy.deepcopy(state)
        loss = 0
        for _ in range(6):
            tokens = torch.randint(27, (2, 1))
            y1, left, _ = star(tokens, left)
            y2, right, _ = dynamic(tokens, right)
            torch.testing.assert_close(y1, y2, rtol=0, atol=0)
            torch.testing.assert_close(left['h'], right['h'], rtol=0, atol=0)
            loss += y2.square().mean()
        loss.backward()
        for comm in dynamic.rnn.comm:
            self.assertGreater(comm.query.weight.grad.abs().sum().item(), 0)
            self.assertEqual(comm.key.weight.grad.count_nonzero().item(), 0)
        torch.optim.SGD(dynamic.parameters(), lr=0.1).step()
        dynamic.zero_grad()
        logits, _, _ = dynamic(torch.randint(27, (2, 1)), dynamic.init_state(2))
        logits.square().sum().backward()
        for comm in dynamic.rnn.comm:
            self.assertGreater(comm.key.weight.grad.abs().sum().item(), 0)

    def test_dynamic_self_heavy_keeps_hub_access_and_peripheral_self_mass(self):
        model = core(topology='star', hub_query_size=8, routing_init='self_heavy')
        for layer, comm in enumerate(model.comm):
            weights = comm.pi_route_logits.masked_fill(~comm.allowed, float('-inf')).softmax(-1)
            torch.testing.assert_close(weights.diagonal()[1:], torch.full((3,), 0.7))
            torch.testing.assert_close(weights[1:, 0], torch.full((3,), 0.15 if layer == 0 else 0.3))
            if layer == 0:
                torch.testing.assert_close(weights[1:, -1], torch.full((3,), 0.15))

    def test_dense_hub_and_block_periphery_equal_expanded_dense_bank(self):
        mixed = core(projection_groups=2, dense_hub=True)
        dense = DenseGridRnn(
            hidden_size=12, n_layers=2, n_columns=4, horizon=[1, 1000],
            fb_norm=True, dtype=torch.float32, device='cpu',
        )
        with torch.no_grad():
            dense.cells.log_r.copy_(mixed.cells.log_r)
            dense.cells.theta.copy_(mixed.cells.theta)
            dense.cells.weight_in.zero_()
            dense.cells.weight_out.zero_()
            dense.cells.weight_in[:, 0].copy_(mixed.cells.hub_weight_in)
            dense.cells.weight_out[:, 0].copy_(mixed.cells.hub_weight_out)
            for group in range(2):
                start, end = group * 6, (group + 1) * 6
                dense.cells.weight_in[:, 1:, start:end, start:end].copy_(mixed.cells.weight_in[:, :, group, :, :6])
                dense.cells.weight_in[:, 1:, start:end, 12 + start:12 + end].copy_(mixed.cells.weight_in[:, :, group, :, 6:])
                dense.cells.weight_out[:, 1:, start:end, start:end].copy_(mixed.cells.weight_out[:, :, group, :6])
                dense.cells.weight_out[:, 1:, 12 + start:12 + end, start:end].copy_(mixed.cells.weight_out[:, :, group, 6:])
        x, h = torch.randn(4, 2, 12), torch.randn(4, 2, 24)
        for layer in range(2):
            y1, h1 = mixed.cells(layer, x, h)
            y2, h2 = dense.cells(layer, x, h)
            torch.testing.assert_close(h1, h2)
            torch.testing.assert_close(y1, y2)
            columns = torch.tensor([0, 1, 3])
            selected_y, selected_h = mixed.cells(layer, x[columns], h[columns], columns)
            torch.testing.assert_close(selected_h, h2[columns])
            torch.testing.assert_close(selected_y, y2[:, columns])

    def test_output_clock_records_all_tokens_but_skips_inactive_output_projection(self):
        model = core(
            update_periods=[1, 1, 2, 2], update_offsets=[0, 0, 0, 1],
            clockwork_mode='output',
        )
        _, state, _ = model(torch.randn(2, 1, 12), model.init_state(2))
        for phase, frozen in ((0, 3), (1, 2)):
            state = model.detach_state(state)
            x = torch.randn(2, 1, 12)
            expected_h = []
            messages = torch.cat((state['out'], x), dim=1)
            # An ordinary full cell call is the oracle for continuous writes.
            for layer in range(2):
                message, _ = model.comm[layer](state['outs'][layer], messages, messages)
                full_out, h = model.cells(layer, message, state['h'][layer])
                expected_h.append(h)
                messages = full_out.clone()
                messages[:, frozen] = state['outs'][layer, :, frozen]
            model.zero_grad()
            with patch('torch.bmm', wraps=torch.bmm) as bmm:
                y, new_state, _ = model(x, state)
            shapes = [tuple(call.args[1].shape) for call in bmm.call_args_list]
            self.assertEqual(shapes.count((4, 12, 24)), 2)
            self.assertEqual(shapes.count((3, 24, 12)), 2)
            torch.testing.assert_close(new_state['h'], torch.stack(expected_h))
            torch.testing.assert_close(new_state['outs'][:, :, frozen], state['outs'][:, :, frozen], rtol=0, atol=0)
            self.assertFalse(torch.equal(new_state['h'][:, frozen], state['h'][:, frozen]))
            (y.square().mean() + new_state['h'].square().mean()).backward()
            self.assertGreater(model.cells.weight_in.grad[:, frozen].abs().sum().item(), 0)
            self.assertEqual(model.cells.weight_out.grad[:, frozen].count_nonzero().item(), 0)
            self.assertEqual(new_state['clock_phase'], (phase + 1) % 2)
            state = new_state
        cleared = model.reset_state(state, torch.tensor([True, False]))
        self.assertEqual(cleared['h'][:, :, 0].count_nonzero().item(), 0)
        self.assertEqual(cleared['outs'][:, 0].count_nonzero().item(), 0)
        self.assertEqual(cleared['clock_phase'], state['clock_phase'])

    def test_followup_configs_keep_training_protocol_and_parameter_budgets(self):
        base = load_config(CONFIG_DIR / 'control.yaml')
        protocol_keys = ('n_envs', 'rollout_len', 'n_steps', 'lr', 'gens', 'eval', 'dtype', 'compile', 'communication', 'seed')
        for variant, expected in COUNTS.items():
            with self.subTest(variant=variant):
                cfg = load_config(CONFIG_DIR / f'{variant}.yaml')
                self.assertEqual({k: cfg[k] for k in protocol_keys}, {k: base[k] for k in protocol_keys})
                self.assertEqual(measure(cfg, cfg['model'].replace('.', '_'))[:2], expected)

    def test_new_paths_compile_and_backpropagate_in_all_clock_phases(self):
        for variant in COUNTS:
            with self.subTest(variant=variant):
                cfg = load_config(CONFIG_DIR / f'{variant}.yaml')
                model = build(cfg, cfg['model'].replace('.', '_'), {'hidden_size': 12})
                compiled = torch.compile(model, backend='eager', fullgraph=True)
                left = model.init_state(2)
                right = copy.deepcopy(left)
                loss = 0
                for _ in range(5):
                    tokens = torch.randint(27, (2, 1))
                    y1, left, _ = model(tokens, left)
                    y2, right, _ = compiled(tokens, right)
                    torch.testing.assert_close(y1, y2)
                    torch.testing.assert_close(left['h'], right['h'])
                    torch.testing.assert_close(left['outs'], right['outs'])
                    torch.testing.assert_close(y2, model.head(right['outs'][-1][:, 0]))
                    loss += y2.square().mean()
                loss.backward()
                self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
                torch._dynamo.reset()

    def test_invalid_followup_controls_are_rejected(self):
        for options in (
            {'hub_layers': [1]}, {'hub_query_size': 8, 'hub_layers': []},
            {'hub_query_size': 8, 'hub_layers': [2]},
            {'hub_query_size': 8, 'hub_layers': [1, 1]},
            {'hub_query_size': 8, 'hub_layers': [True]},
            {'hub_query_init': 'missing'}, {'dense_hub': True},
            {'clockwork_mode': 'missing'},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                core(**options)


if __name__ == '__main__':
    unittest.main()
