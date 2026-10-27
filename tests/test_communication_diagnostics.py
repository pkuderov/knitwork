import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from inference.communication_diagnostics import DEFAULTS, analyze_run, cka_matrices, linear_cka, readout_features
from knitwork.exps.communication.protocol import make_model, resolve_config, tensor_digest
from test_communication_controls import tiny_config


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cka_alignment_translation_scale_rotation_and_degeneracy(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(128, 8))
        rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        self.assertAlmostEqual(linear_cka(x, x * 3 @ rotation + 17), 1, places=10)
        self.assertLess(linear_cka(x, rng.normal(size=x.shape)), 0.2)
        self.assertIsNone(linear_cka(np.ones_like(x), x))
        matrices = cka_matrices(np.stack((x, x, x, x), axis=1)[:, None])
        np.testing.assert_allclose(matrices, np.ones((1, 4, 4)))

    def test_readout_shapes_and_concat_order(self):
        top = np.arange(24).reshape(2, 4, 3)
        np.testing.assert_array_equal(readout_features(top, 'C2'), top[:, 2])
        np.testing.assert_array_equal(readout_features(top, 'Concat'), top.reshape(2, 12))
        np.testing.assert_array_equal(readout_features(top, 'Mean'), top.mean(1))

    def test_diagnostics_freeze_backbone_and_preserve_none_causal_isolation(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            dataset = root / 'text8.txt'
            dataset.write_text('abcdefghijklmnopqrstuvwxyz ' * 100)
            settings = dict(DEFAULTS, streams=4, warmup_steps=8, text_steps=64, sdq_steps=128, probe_updates=20, probe_batch=16)
            for task in ('text8', 'sdq'):
                cfg = resolve_config(tiny_config(task, root, dataset, condition='none_column0', recipe='clean'))
                model = make_model(cfg).eval().requires_grad_(False)
                before = tensor_digest(model.state_dict())
                directory = root / task
                directory.mkdir()
                record = {'config': cfg, 'checkpoint': {'path': 'synthetic', 'sha256': 'test'}, 'step': 96}
                text = np.tile(np.arange(27), 100)
                result = analyze_run(record, model, (text, text), settings, directory)
                self.assertEqual(before, tensor_digest(model.state_dict()))
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertEqual(np.asarray(result['cka']).shape, (2, 4, 4))
                self.assertEqual(len(result['probes']['targets']), 6)
                self.assertEqual(len(result['concat_column_replacement']), 4)
                if task == 'sdq':
                    self.assertEqual(len(result['probes']['stored_values']), 6)
                    for column in (1, 2, 3):
                        self.assertEqual(result['query_state_lesions'][column]['delta_Acc'], 0)


if __name__ == '__main__':
    unittest.main()
