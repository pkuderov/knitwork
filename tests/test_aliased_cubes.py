import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest

from gymnasium.utils.env_checker import check_env
import numpy as np
import torch
import yaml

from knitwork.env.aliased_cube import AliasedCube, cube_surface_transitions
from knitwork.exps.aliased_cubes.model import ActionObservationModel, reactive_probabilities
from knitwork.exps.aliased_cubes.run import evaluate, evaluate_reactive, main


ROOT = Path(__file__).resolve().parents[1]


class AliasedCubeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_cube_topology_seams_and_connectivity(self):
        for edge in (2, 3, 6):
            with self.subTest(edge=edge):
                table = cube_surface_transitions(edge)
                n_states = 6 * edge ** 2
                self.assertEqual(table.shape, (n_states, 4))
                self.assertGreaterEqual(table.min(), 0)
                self.assertLess(table.max(), n_states)
                for state in range(n_states):
                    self.assertEqual(len(set(table[state])), 4)
                    for neighbor in table[state]:
                        self.assertIn(state, table[neighbor])
                visited, frontier = {0}, [0]
                while frontier:
                    for neighbor in table[frontier.pop()]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            frontier.append(neighbor)
                self.assertEqual(len(visited), n_states)
        # Face 0 interior east/west preserve row and change column by one.
        table = cube_surface_transitions(6)
        self.assertEqual(table[14, 0], 15)
        self.assertEqual(table[14, 1], 13)
        self.assertNotEqual(table[5, 0] // 36, 0)

    def test_gym_api_and_no_privileged_observation(self):
        env = AliasedCube()
        check_env(env, skip_render_check=True)
        observation, info = env.reset(seed=7)
        state = env.state
        result = env.step(0)
        self.assertEqual(env.state, env.transitions[state, 0])
        self.assertEqual(result, (int(env.observations[env.state]), 0.0, False, False, {}))
        self.assertEqual(info, {})
        self.assertTrue(env.observation_space.contains(observation))
        with self.assertRaises(ValueError):
            env.step(-1)

    def test_layout_and_walk_seeds_are_separate_and_reproducible(self):
        env1, env2, env3 = AliasedCube(map_seed=0), AliasedCube(map_seed=0), AliasedCube(map_seed=1)
        self.assertEqual(env1.fingerprint, env2.fingerprint)
        self.assertNotEqual(env1.fingerprint, env3.fingerprint)
        self.assertEqual(np.bincount(env1.observations).tolist(), [18] * 12)
        first = env1.sample_walks(4, 10, 100)
        second = env2.sample_walks(4, 10, 100)
        other = env1.sample_walks(4, 10, 101)
        self.assertEqual(set(first), {'observations', 'actions'})
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertFalse(np.array_equal(first['actions'], other['actions']))
        self.assertEqual(first['observations'].shape, (4, 10))
        self.assertEqual(first['actions'].shape, (4, 9))

    def test_external_graph_import_and_invalid_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.npz'
            np.savez(path, transitions=np.array([[1, 0], [0, 1]]), observations=np.array([0, 1]))
            env = AliasedCube(graph_path=path)
            self.assertEqual((env.n_states, env.n_actions, env.n_observations), (2, 2, 2))
            self.assertEqual(env.source, 'external_transition_table')
            env.reset(options={'state': 0})
            self.assertEqual(env.step(0)[0], 1)
            np.savez(path, transitions=np.array([[2, 0], [0, 1]]), observations=np.array([0, 1]))
            with self.assertRaises(ValueError):
                AliasedCube(graph_path=path)

    def test_reactive_model_uses_observation_action_conditioning(self):
        train = {'observations': np.array([[0, 1, 0], [0, 1, 0]]), 'actions': np.array([[0, 1], [0, 1]])}
        probabilities = reactive_probabilities(train, 2, 2)
        torch.testing.assert_close(probabilities[0, 0], torch.tensor([0.25, 0.75], dtype=torch.float64))
        torch.testing.assert_close(probabilities[1, 1], torch.tensor([0.75, 0.25], dtype=torch.float64))
        torch.testing.assert_close(probabilities[0, 1], torch.tensor([0.5, 0.5], dtype=torch.float64))
        self.assertEqual(evaluate_reactive(probabilities, train, 0)['Acc'], 1.0)

    def test_adapter_lru_and_gru_forward_backward_and_compile(self):
        for name, cfg in (
                ('grnn_lru', dict(hidden_size=12, n_layers=2, n_columns=4, horizon=[1, 1000], fb_norm=True)),
                ('rnn', dict(hidden_size=12, n_layers=2)),
        ):
            with self.subTest(model=name):
                model = ActionObservationModel(
                    n_observations=12, n_actions=4, rnn_type=name, rnn_cfg=cfg,
                    dtype=torch.float32, device=torch.device('cpu'),
                )
                state = model.rnn.init_state(2)
                observation, action = torch.tensor([1, 2]), torch.tensor([0, 3])
                logits, next_state = model(observation, action, state)
                self.assertEqual(logits.shape, (2, 12))
                logits.square().mean().backward()
                self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
                if name == 'grnn_lru':
                    compiled = torch.compile(model, backend='eager', fullgraph=True)
                    compiled_logits, compiled_state = compiled(observation, action, state)
                    torch.testing.assert_close(compiled_logits, logits)
                    torch.testing.assert_close(compiled_state['h'], next_state['h'])
                    torch._dynamo.reset()

    def test_eval_is_deterministic_and_preserves_training_rng(self):
        model = ActionObservationModel(
            n_observations=12, n_actions=4, rnn_type='grnn_lru',
            rnn_cfg=dict(hidden_size=12, n_layers=2, n_columns=4, horizon=[1, 1000], fb_norm=True),
            dtype=torch.float32, device=torch.device('cpu'),
        )
        walks = AliasedCube().sample_walks(4, 10, 100)
        kwargs = dict(batch_size=2, device=torch.device('cpu'), report_after=3)
        rng_before = torch.get_rng_state().clone()
        result1 = evaluate(model, walks, **kwargs)
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        result2 = evaluate(model, walks, **kwargs)
        self.assertEqual(result1, result2)
        self.assertEqual(len(result1['Acc_by_step']), 9)
        self.assertEqual(result1['context_steps'], 3)

    def test_mid_runner_selection_checkpointing_and_gru_control(self):
        config = yaml.safe_load((ROOT / 'knitwork/exps/aliased_cubes/config/mid.yaml').read_text())
        config['device'], config['output_dir'] = 'cpu', None
        config['data'].update(train_sequences=4, val_sequences=2, test_sequences=2, sequence_length=9)
        config['training'].update(iterations=2, batch_size=2, rollout_len=4)
        config['eval'].update(schedule=1, batch_size=2, report_after=3)
        with tempfile.TemporaryDirectory() as directory:
            config['output_dir'] = directory
            with contextlib.redirect_stdout(io.StringIO()):
                first = main(config)
            self.assertEqual(first['training_transitions'], 32)
            self.assertIn(first['best_iteration'], (1, 2))
            self.assertTrue(np.isfinite(first['test']['Loss']))
            run_dir, = Path(directory).iterdir()
            self.assertTrue((run_dir / 'config.json').exists())
            self.assertTrue((run_dir / 'best.pt').exists())
            self.assertTrue((run_dir / 'last.pt').exists())
            self.assertTrue((run_dir / 'metrics.json').exists())
            # Repeat from the same config: existing artifacts are retained in a unique run directory.
            with contextlib.redirect_stdout(io.StringIO()):
                second = main(config)
            self.assertEqual(first, second)
            self.assertEqual(len(list(Path(directory).iterdir())), 2)
        gru_cfg = copy.deepcopy(config)
        gru_cfg['model'], gru_cfg['output_dir'] = 'rnn.L2', None
        with contextlib.redirect_stdout(io.StringIO()):
            gru_result = main(gru_cfg)
        self.assertTrue(np.isfinite(gru_result['test']['Loss']))


if __name__ == '__main__':
    unittest.main()
