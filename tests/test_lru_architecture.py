import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from knitwork.common.config import load_config
from knitwork.common.state_size import build, state_floats
from knitwork.common.utils import count_learnable_params
from knitwork.exps.aliased_cubes.run import main as cubes_main
from knitwork.exps.mqar.run import main as mqar_main, model_state_info, sequence_objective
from knitwork.exps.text.run import main as text_main
from knitwork.gens.mqar import generate_mqar
from knitwork.models.grnn_lru_experimental import GridRnn, MECHANISMS, delta_write
from knitwork.models.grnn_lru_sparse import GridRnn as Control
from knitwork.models.utils import build_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'knitwork/exps'
TEXT_COUNTS = {
    'dynamic_control': 1055223, 'write_gate': 1057389,
    'query_read': 1112823, 'associative': 830321,
    'rotations': 1055247, 'two_reads': 1055223,
}


def core(mechanism, **options):
    torch.manual_seed(42)
    return GridRnn(
        mechanism=mechanism, hidden_size=12, n_layers=options.pop('n_layers', 2),
        n_columns=4, horizon=[1, 1000], fb_norm=True,
        dtype=torch.float32, device='cpu', **options,
    )


def control(n_layers=2):
    torch.manual_seed(42)
    return Control(
        hidden_size=12, n_layers=n_layers, n_columns=4, horizon=[1, 1000],
        fb_norm=True, topology='star', hub_query_size=8, hub_query_init='zero',
        dtype=torch.float32, device='cpu',
    )


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gate_extremes_preserve_memory_or_recover_control(self):
        model, base = core('write_gate'), control()
        state = base.init_state(2)
        x = torch.randn(2, 1, 12)
        with torch.no_grad():
            for gate in model.write_gates:
                gate.weight.zero_()
                gate.bias.fill_(100)
        y1, s1, _ = model(x, copy.deepcopy(state))
        y2, s2, _ = base(x, copy.deepcopy(state))
        torch.testing.assert_close(y1, y2)
        torch.testing.assert_close(s1['h'], s2['h'])
        with torch.no_grad():
            for gate in model.write_gates:
                gate.bias.fill_(-100)
        _, frozen, info = model(x, copy.deepcopy(state), capture=True)
        torch.testing.assert_close(frozen['h'][:, 1:], state['h'][:, 1:], rtol=0, atol=0)
        self.assertFalse(torch.equal(frozen['h'][:, 0], state['h'][:, 0]))
        self.assertFalse(torch.equal(frozen['outs'][:, :, 1:], state['outs'][:, :, 1:]))
        self.assertTrue(all(gate[0].item() == 1 for gate in info['write_gate']))

    def test_query_changes_private_read_without_changing_peripheral_update(self):
        model = core('query_read', n_layers=1)
        state = model.init_state(2)
        changed = copy.deepcopy(state)
        changed['outs'] = changed['outs'].clone()
        changed['outs'][0, :, 0] += 3
        x = torch.randn(2, 1, 12)
        y1, s1, _ = model(x, state)
        y2, s2, _ = model(x, changed)
        self.assertFalse(torch.allclose(y1, y2))
        torch.testing.assert_close(s1['h'][:, 1:], s2['h'][:, 1:])
        torch.testing.assert_close(s1['outs'][:, :, 1:], s2['outs'][:, :, 1:])

    def test_delta_rule_overwrites_one_binding_and_preserves_orthogonal_keys(self):
        memory = torch.randn(2, 4, 3, requires_grad=True)
        keys = torch.tensor([[0., 1., 0., 0.], [0., 0., 0., 1.]])
        values = torch.randn(2, 3, requires_grad=True)
        updated = delta_write(memory, keys, values, torch.ones(2, 1))
        retrieved = torch.einsum('bk,bkv->bv', keys, updated)
        torch.testing.assert_close(retrieved, values)
        torch.testing.assert_close(updated[0, [0, 2, 3]], memory[0, [0, 2, 3]])
        torch.testing.assert_close(updated[1, :3], memory[1, :3])
        torch.testing.assert_close(delta_write(memory, keys, values, torch.zeros(2, 1)), memory)
        updated.square().sum().backward()
        self.assertGreater(values.grad.abs().sum().item(), 0)
        self.assertTrue(memory.grad.isfinite().all())

    def test_associative_column_is_replaced_and_has_cross_token_gradient(self):
        model = core('associative', memory_key_size=4, memory_value_size=6)
        state = model.init_state(2)
        self.assertEqual(model.cells.weight_in.shape, (2, 3, 12, 24))
        self.assertEqual(state['h'].shape, (2, 3, 2, 24))
        self.assertEqual(state['memory'].shape, (2, 2, 4, 6))
        _, state, _ = model(torch.randn(2, 1, 12), state)
        written = state['memory']
        written.retain_grad()
        self.assertGreater(written.abs().sum().item(), 0)
        y, _, _ = model(torch.randn(2, 1, 12), state)
        y.square().mean().backward()
        self.assertGreater(written.grad.abs().sum().item(), 0)
        for modules in (model.memory_writes, model.memory_queries, model.memory_outputs):
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in modules.parameters()))

    def test_rotations_preserve_norm_and_zero_angles_recover_control(self):
        model, base = core('rotations'), control()
        state = base.init_state(2)
        x = torch.randn(2, 1, 12)
        y1, s1, _ = model(x, copy.deepcopy(state))
        y2, s2, _ = base(x, copy.deepcopy(state))
        torch.testing.assert_close(y1, y2)
        torch.testing.assert_close(s1['h'], s2['h'])
        with torch.no_grad():
            model.rotation_angles.uniform_(-3, 3)
        h = torch.randn(4, 2, 24)
        for layer in range(2):
            rotated = model._rotate(layer, h)
            torch.testing.assert_close(rotated.square().sum(0), h.square().sum(0))
        y, _, _ = model(x, state)
        y.square().mean().backward()
        self.assertGreater(model.rotation_angles.grad.abs().sum().item(), 0)

    def test_two_reads_update_periphery_once_and_refine_hub_from_original_state(self):
        model, base = core('two_reads', n_layers=1), control(n_layers=1)
        state = base.init_state(2)
        x = torch.randn(2, 1, 12)
        _, ordinary, _ = base(x, copy.deepcopy(state))
        with patch('torch.bmm', wraps=torch.bmm) as bmm:
            y, refined, info = model(x, state, capture=True)
        shapes = [tuple(call.args[1].shape) for call in bmm.call_args_list]
        self.assertEqual(shapes.count((4, 12, 24)), 1)
        self.assertEqual(shapes.count((1, 12, 24)), 1)
        self.assertEqual(shapes.count((4, 24, 12)), 1)
        self.assertEqual(shapes.count((1, 24, 12)), 1)
        torch.testing.assert_close(refined['h'][:, 1:], ordinary['h'][:, 1:])
        torch.testing.assert_close(refined['outs'][:, :, 1:], ordinary['outs'][:, :, 1:])
        sources = torch.cat((ordinary['outs'][0], x), dim=1)
        message, _ = model._hub_message(0, sources, ordinary['outs'][0, :, 0] + x[:, 0], state['h'][0])
        expected_h = model._advance_cells(0, message, state['h'][0, :1], model.hub_column)
        torch.testing.assert_close(refined['h'][0, :1], expected_h)
        torch.testing.assert_close(y, refined['outs'][-1, :, 0])
        self.assertIn('second_attn_weights', info)

    def test_same_seed_embedding_and_head_are_preserved(self):
        cfg = dict(hidden_size=12, n_layers=2, n_columns=4, horizon=[1, 1000], fb_norm=True)
        wrapper = dict(input_size=64, output_size=64, dtype=torch.float32, device='cpu')
        torch.manual_seed(42)
        base = build_model(
            wrapper_type='token', wrapper_cfg=wrapper, rnn_type='grnn_lru_sparse',
            rnn_cfg=cfg | dict(topology='star', hub_query_size=8, hub_query_init='zero'),
        )
        rng_after = torch.get_rng_state().clone()
        for mechanism in MECHANISMS:
            torch.manual_seed(42)
            model = build_model(
                wrapper_type='token', wrapper_cfg=wrapper, rnn_type='grnn_lru_experimental',
                rnn_cfg=cfg | dict(mechanism=mechanism),
            )
            torch.testing.assert_close(torch.get_rng_state(), rng_after, rtol=0, atol=0)
            torch.testing.assert_close(model.embedding.weight, base.embedding.weight, rtol=0, atol=0)
            torch.testing.assert_close(model.head.weight, base.head.weight, rtol=0, atol=0)

    def test_all_paths_compile_reset_detach_and_backpropagate(self):
        for mechanism in MECHANISMS:
            with self.subTest(mechanism=mechanism):
                model = core(mechanism)
                compiled = torch.compile(model, backend='eager', fullgraph=True)
                state = model.init_state(2)
                right = copy.deepcopy(state)
                loss = 0
                for step in range(4):
                    x = torch.randn(2, 1, 12)
                    y, state, _ = model(x, state, capture=bool(step % 2))
                    z, right, _ = compiled(x, right, capture=bool(step % 2))
                    torch.testing.assert_close(y, z)
                    torch.testing.assert_close(z, right['outs'][-1, :, 0])
                    for key in state:
                        torch.testing.assert_close(state[key], right[key])
                    loss += z.square().mean()
                loss.backward()
                self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
                detached = model.detach_state(state)
                self.assertTrue(all(not value.requires_grad for value in detached.values()))
                cleared = model.reset_state(detached, torch.tensor([True, False]))
                for key, batch_dim in [('h', 2), ('outs', 1), ('out', 0)] + ([('memory', 1)] if mechanism == 'associative' else []):
                    self.assertEqual(cleared[key].select(batch_dim, 0).count_nonzero().item(), 0)
                    torch.testing.assert_close(cleared[key].select(batch_dim, 1), detached[key].select(batch_dim, 1))
                self.assertEqual(model.inspection_hidden(state).shape[1], 4)
                torch._dynamo.reset()

    def test_configs_keep_protocols_and_record_state_and_parameter_budgets(self):
        for task in ('text', 'mqar', 'aliased_cubes'):
            directory = CONFIG / task / 'config/lru_architecture_mid'
            control_cfg = load_config(directory / 'dynamic_control.yaml')
            for name in (*MECHANISMS, 'dynamic_control'):
                cfg = load_config(directory / f'{name}.yaml')
                keys = ('data', 'training', 'eval') if task != 'text' else ('n_envs', 'rollout_len', 'n_steps', 'gens', 'eval', 'lr', 'communication')
                for key in keys:
                    self.assertEqual(cfg[key], control_cfg[key])
                if task == 'text':
                    model = build(cfg, cfg['model'].replace('.', '_'))
                    self.assertEqual(count_learnable_params(model), TEXT_COUNTS[name])
                    sizes = model_state_info(model, torch.device('cpu'))
                    self.assertEqual(state_floats(model.rnn), 4264 if name == 'associative' else 3960)
                    self.assertEqual(sizes['allocated_state_floats'], 5344 if name == 'associative' else 5040)

    def test_mqar_late_queries_reach_stored_value_embeddings(self):
        data = generate_mqar(vocab_size=64, input_seq_len=32, num_kv_pairs=4, num_examples=2, seed=10)
        cfg = load_config(CONFIG / 'mqar/config/lru_architecture_mid/smoke.yaml')
        for mechanism in MECHANISMS:
            with self.subTest(mechanism=mechanism):
                model = build_model(
                    wrapper_type='token', rnn_type='grnn_lru_experimental',
                    wrapper_cfg=dict(input_size=64, output_size=64, dtype=torch.float32, device='cpu'),
                    rnn_cfg=cfg[f'grnn_lru_experimental_{mechanism}'],
                )
                loss, _, _ = sequence_objective(model, data['inputs'], data['labels'])
                loss.backward()
                values = data['inputs'][:, 1:8:2].unique()
                self.assertGreater(model.embedding.weight.grad[values].abs().sum().item(), 0)

    def test_all_mqar_and_cube_pipelines_save_held_out_results(self):
        for mechanism in MECHANISMS:
            with self.subTest(mechanism=mechanism), tempfile.TemporaryDirectory() as directory:
                cfg = load_config(CONFIG / 'mqar/config/lru_architecture_mid/smoke.yaml')
                cfg['model'] = f'grnn_lru_experimental.{mechanism}'
                cfg['output_dir'] = directory
                with contextlib.redirect_stdout(io.StringIO()):
                    result = mqar_main(cfg)
                outputs = list(Path(directory).iterdir())
                self.assertEqual(len(outputs), 1)
                self.assertTrue((outputs[0] / 'best.pt').exists())
                self.assertTrue((outputs[0] / 'last.pt').exists())
                self.assertTrue((outputs[0] / 'metrics.json').exists())
                recorded = json.loads((outputs[0] / 'config.json').read_text())
                self.assertEqual(recorded['model'], cfg['model'])
                self.assertTrue(torch.isfinite(torch.tensor(result['best_val_loss'])))
                cube = load_config(CONFIG / f'aliased_cubes/config/lru_architecture_mid/{mechanism}.yaml')
                cube[cube['model'].replace('.', '_')]['hidden_size'] = 12
                cube['data'].update(sequence_length=12, train_sequences=4, val_sequences=2, test_sequences=2)
                cube['training'].update(iterations=2, batch_size=2, rollout_len=4)
                cube['eval'].update(schedule=1, batch_size=2, report_after=3)
                cube['output_dir'] = directory
                with contextlib.redirect_stdout(io.StringIO()):
                    cubes_main(cube)
                self.assertEqual(len(list(Path(directory).iterdir())), 2)

    def test_text_pipeline_includes_compact_memory_inspection(self):
        for mechanism in MECHANISMS:
            with self.subTest(mechanism=mechanism), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'text.txt'
                path.write_text('abcdefghijklmnopqrstuvwxyz ' * 40)
                cfg = load_config(CONFIG / f'text/config/lru_architecture_mid/{mechanism}.yaml')
                cfg[cfg['model'].replace('.', '_')]['hidden_size'] = 12
                cfg['gens']['text8']['path'] = str(path)
                cfg.update(n_envs=2, rollout_len=4, n_steps=16, compile=False, inspect_schedule=2, vis_inspect_schedule=2)
                cfg['eval']['enabled'] = False
                cfg['log'].update(logger=None, schedule=8)
                with contextlib.redirect_stdout(io.StringIO()), patch('torch.compile', side_effect=lambda value, **kwargs: value):
                    text_main(cfg)

    def test_invalid_controls_are_rejected(self):
        for options in (
            dict(mechanism='unknown'), dict(n_inputs=2), dict(n_outputs=2),
            dict(hub_query_size=0), dict(read_rank=0), dict(memory_key_size=True),
            dict(memory_value_size=-1), dict(memory_column=0), dict(memory_column=4),
            dict(rotation_groups=13), dict(write_gate_bias=float('nan')),
        ):
            mechanism = options.pop('mechanism', 'write_gate')
            with self.subTest(options=options), self.assertRaises(ValueError):
                core(mechanism, **options)


if __name__ == '__main__':
    unittest.main()
