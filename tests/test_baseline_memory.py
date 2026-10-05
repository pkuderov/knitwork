import unittest

import torch
from torch.utils.checkpoint import checkpoint

from knitwork.models.baseline.delta_net import DeltaNetCore
from knitwork.models.baseline.hgrn2 import HGRN2Core
from knitwork.models.baseline.mlstm import mLSTMCore

B, T, H = 5, 14, 24
KW = dict(dtype=torch.float32, device='cpu')


def make(kind):
    torch.manual_seed(0)
    return {
        'delta_net': lambda: DeltaNetCore(hidden_size=H, n_layers=2, **KW),
        'mlstm': lambda: mLSTMCore(hidden_size=H, n_layers=2, **KW),
        'hgrn2': lambda: HGRN2Core(hidden_size=H, n_layers=2, expand=2, **KW),
    }[kind]()


def explicit_reset(kind, state, reset):
    """Reference: the original semantics, multiply every state tensor by the keep mask before the step."""
    keep = (~reset)[:, None, None].float()
    if kind == 'delta_net':
        return {'S': [S * keep for S in state['S']]}
    if kind == 'mlstm':
        return {'C': [C * keep for C in state['C']], 'n': [n * keep[:, :, 0] for n in state['n']],
                'm': [m * keep[:, :, 0] for m in state['m']]}
    return {'h': [h * keep for h in state['h']]}


class BaselineMemoryTests(unittest.TestCase):
    def inputs(self):
        g = torch.Generator().manual_seed(1)
        xs = torch.randn(T, 1, B, H, generator=g)
        resets = torch.rand(T, B, generator=g) < 0.3
        return xs, resets

    def rollout(self, kind, lazy, segment=0):
        core, (xs, resets) = make(kind), self.inputs()

        def run(state, ts):
            ys = []
            for t in ts:
                state = core.reset_state(state, resets[t]) if lazy else explicit_reset(kind, state, resets[t])
                y, state, _ = core(xs[t], state)
                ys.append(y)
            return torch.stack(ys), state

        state, outs = core.init_state(B), []
        for s in range(0, T, segment or T):
            ts = list(range(s, min(T, s + (segment or T))))
            y, state = checkpoint(run, state, ts, use_reentrant=False) if segment else run(state, ts)
            outs.append(y)
        y = torch.cat(outs)
        (y ** 2).sum().backward()
        return y.detach(), torch.cat([p.grad.flatten() for p in core.parameters()])

    def test_lazy_reset_matches_explicit_state_multiplication(self):
        for kind in ('delta_net', 'mlstm', 'hgrn2'):
            with self.subTest(kind=kind):
                y0, g0 = self.rollout(kind, lazy=False)
                y1, g1 = self.rollout(kind, lazy=True)
                self.assertTrue(torch.allclose(y0, y1, rtol=1e-4, atol=1e-5))
                self.assertTrue(torch.allclose(g0, g1, rtol=1e-3, atol=1e-5))

    def test_segment_checkpointing_gives_the_same_gradients(self):
        for kind in ('delta_net', 'mlstm', 'hgrn2'):
            with self.subTest(kind=kind):
                y0, g0 = self.rollout(kind, lazy=True)
                y1, g1 = self.rollout(kind, lazy=True, segment=7)
                self.assertTrue(torch.allclose(y0, y1, rtol=1e-5, atol=1e-6))
                self.assertTrue(torch.allclose(g0, g1, rtol=1e-4, atol=1e-6))

    def test_reset_state_detach_and_carried_state_size(self):
        from knitwork.common.state_size import state_floats
        for kind in ('delta_net', 'mlstm'):
            core = make(kind)
            state = core.reset_state(core.init_state(B), torch.tensor([True, False, False, False, False]))
            self.assertIn('keep', state)
            y, state, _ = core(torch.randn(1, B, H), state)
            self.assertNotIn('keep', state)  # the pending reset is consumed by the step
            self.assertNotIn('keep', core.detach_state(state))
            self.assertEqual(core.reset_state(None, torch.zeros(B, dtype=torch.bool))['S' if kind == 'delta_net' else 'C'][0].shape[0], B)
            # one more reset, then detach keeps the pending mask
            pending = core.detach_state(core.reset_state(state, torch.ones(B, dtype=torch.bool)))
            self.assertIn('keep', pending)
            self.assertGreater(state_floats(core), 0)


if __name__ == '__main__':
    unittest.main()
