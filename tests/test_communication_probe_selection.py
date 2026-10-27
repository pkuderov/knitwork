import unittest

import numpy as np
import torch

from inference.communication_diagnostics import DEFAULTS
from inference.select_communication_readouts import fit_selected


class ProbeSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_selection_does_not_use_main_validation_labels(self):
        rng = np.random.default_rng(13)
        train = rng.normal(size=(160, 8)).astype(np.float32)
        validation = rng.normal(size=(40, 8)).astype(np.float32)
        target = (train[:, 0] > 0).astype(int)
        val_target = (validation[:, 0] > 0).astype(int)
        groups = np.tile(np.arange(4), 40)
        settings = dict(DEFAULTS, streams=4, probe_updates=60, probe_batch=32, probe_penalties=[0, 0.1])
        scores, head, _ = fit_selected(train, target, validation, val_target, groups, 2, settings)
        changed_scores, changed_head, _ = fit_selected(train, target, validation, 1 - val_target, groups, 2, settings)
        self.assertEqual(scores['selection'], changed_scores['selection'])
        torch.testing.assert_close(head['weight'], changed_head['weight'], rtol=0, atol=0)
        torch.testing.assert_close(head['bias'], changed_head['bias'], rtol=0, atol=0)
        self.assertGreater(scores['validation']['Acc'], 0.8)
        self.assertLess(changed_scores['validation']['Acc'], 0.2)

    def test_native_initialization_and_raw_coordinates_are_preserved_at_step_zero(self):
        rng = np.random.default_rng(4)
        train = rng.normal(size=(80, 4)).astype(np.float32)
        validation = rng.normal(size=(20, 4)).astype(np.float32)
        initial = {'weight': torch.tensor([[3., 0, 0, 0], [-3., 0, 0, 0]]), 'bias': torch.tensor([0.2, -0.2])}
        target = (torch.as_tensor(train) @ initial['weight'].T + initial['bias']).argmax(-1).numpy()
        val_target = (torch.as_tensor(validation) @ initial['weight'].T + initial['bias']).argmax(-1).numpy()
        settings = dict(DEFAULTS, streams=4, probe_updates=0, probe_penalties=[0])
        scores, head, predictions = fit_selected(train, target, validation, val_target, np.tile(np.arange(4), 20), 2, settings, initial)
        torch.testing.assert_close(head['weight'], initial['weight'])
        torch.testing.assert_close(head['bias'], initial['bias'])
        np.testing.assert_array_equal(predictions, val_target)
        self.assertEqual(scores['selection']['update'], 0)


if __name__ == '__main__':
    unittest.main()
