import unittest

import torch

from knitwork.models.grnn_lru_slots import GridRnn, SLOT_MODES

H, S, K, V = 12, 8, 16, 32
W = K + V + 2


def core(mode, **kw):
    torch.manual_seed(0)
    return GridRnn(mode=mode, hidden_size=H, n_layers=2, n_columns=4, horizon=[1, 1000],
                   dtype=torch.float32, device='cpu', **kw)


class SlotTests(unittest.TestCase):
    def test_shapes_and_reset(self):
        for mode in SLOT_MODES:
            m = core(mode)
            s = m.init_state(3)
            out, s, _ = m(torch.randn(3, 1, H), s)
            self.assertEqual(out.shape, (3, H))
            self.assertEqual(s['memory'].shape, (2, 3, S, W))
            s = m.reset_state(s, torch.tensor([[True], [False], [False]]))
            self.assertEqual(float(s['memory'][:, 0].abs().sum()), 0.0)

    def test_closed_gate_keeps_memory(self):
        for mode in SLOT_MODES:
            m = core(mode, slot_write_bias=-40.0)
            s = m.init_state(2)
            _, s2, _ = m(torch.randn(2, 1, H), s)
            self.assertTrue(torch.allclose(s2['memory'][..., :K + V], s['memory'][..., :K + V]))
            self.assertEqual(float(s2['memory'][..., K + V].sum()), 0.0)  # nothing became occupied

    def test_first_write_touches_one_slot(self):
        for mode in SLOT_MODES:
            m = core(mode, slot_write_bias=40.0)
            new, _ = m._write(0, torch.zeros(2, S, W), torch.randn(2, 2 * H))
            changed = (new[..., :K + V].abs().sum(-1) > 1e-6).sum(-1)
            expected = S if mode == 'slot_soft' else 1  # softmax is the dense control
            self.assertTrue((changed == expected).all(), (mode, changed))
            if mode != 'slot_soft':
                self.assertTrue((new[..., K + V].sum(-1) == 1).all())

    def test_two_facts_survive_distractors(self):
        for mode in ('slot_alloc', 'slot_entmax'):
            m = core(mode, slot_write_bias=40.0)
            memory = torch.zeros(1, S, W)
            for _ in range(2):
                memory = m._write(0, memory, torch.randn(1, 2 * H))[0]
            self.assertEqual(int(memory[0, :, K + V].sum()), 2)
            stored = memory[0, :2, :K + V].clone()
            m.slot_writes[0].bias.data[K + V] = -40.0  # distractors: closed gate
            for _ in range(20):
                memory = m._write(0, memory, torch.randn(1, 2 * H))[0]
            self.assertTrue(torch.equal(memory[0, :2, :K + V], stored))

    def test_oldest_slot_overwritten_when_full(self):
        m = core('slot_alloc', slot_write_bias=40.0)
        memory = torch.zeros(1, S, W)
        for _ in range(S):
            memory = m._write(0, memory, torch.randn(1, 2 * H))[0]
        self.assertEqual(int(memory[0, :, K + V].sum()), S)
        new = m._write(0, memory, torch.randn(1, 2 * H))[0]
        self.assertFalse(torch.equal(new[0, 0, :K + V], memory[0, 0, :K + V]))  # slot 0 is the oldest
        self.assertTrue(torch.equal(new[0, 1:, :K + V], memory[0, 1:, :K + V]))

    def test_gradient_from_late_query_to_past_write(self):
        for mode in SLOT_MODES:
            m = core(mode, slot_write_bias=2.0)
            s = m.init_state(2)
            first = torch.randn(2, 1, H, requires_grad=True)
            _, s, _ = m(first, s)
            for _ in range(7):  # distractors, then the late read is the last output
                out, s, _ = m(torch.randn(2, 1, H), s)
            out.pow(2).sum().backward()
            self.assertGreater(float(first.grad.abs().sum()), 0.0, mode)

    def test_gradients_reach_modules(self):
        for mode in SLOT_MODES:
            m = core(mode)
            s = m.init_state(2)
            loss = 0
            for _ in range(6):
                out, s, _ = m(torch.randn(2, 1, H), s)
                loss = loss + out.pow(2).sum()
            loss.backward()
            for name in ('slot_writes', 'slot_queries', 'slot_outputs'):
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in getattr(m, name).parameters()), (mode, name))


if __name__ == '__main__':
    unittest.main()
