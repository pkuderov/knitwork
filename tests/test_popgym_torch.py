import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

import popgym  # noqa: F401
from knitwork.env.popgym_torch import TASKS, CardMemoryEnv


def deal_order_types(deck, field):
    # popgym deals from the end of the shuffled idx array
    return torch.as_tensor(getattr(deck, field)[deck.idx[::-1].copy()])[None]


@pytest.mark.parametrize('task', sorted(TASKS))
def test_step_equivalence_with_popgym(task):
    rng = np.random.default_rng(0)
    for seed in range(3):
        ref = gym.make(f'popgym-{task}-v0').unwrapped
        obs, _ = ref.reset(seed=seed)
        env = CardMemoryEnv(task, 1, device='cpu', prev_action_reward=False)
        if env.kind == 'count_recall':
            field = {2: 'colors_idx', 4: 'suits_idx', 13: 'ranks_idx'}[env.kw['n_types']]
            cards = deal_order_types(ref.value_deck, field)
            queries = deal_order_types(ref.query_deck, field)
        else:
            deck = ref.deck
            cards, queries = deal_order_types(deck, 'suits_idx'), None
        codes, targets = env.episodes(cards, queries)
        space = ref.observation_space
        total = 0.0
        for t in range(env.ep_len):
            ours = np.concatenate([np.eye(size)[c] for size, c in zip(env.obs_sizes, codes[0, t].tolist())])
            np.testing.assert_array_equal(ours, spaces.flatten(space, obs))
            action = int(rng.integers(env.n_actions))
            obs, reward, terminated, truncated, _ = ref.step(action)
            target = int(targets[0, t])
            expected = 0.0 if target < 0 else env.reward_scale * (1 if action == target else -1)
            assert reward == pytest.approx(expected), (task, t)
            assert terminated == (t == env.ep_len - 1) and not truncated
            total += reward
        assert -1 - 1e-6 <= total <= 1 + 1e-6


def test_autoreset_and_features():
    env = CardMemoryEnv('RepeatFirstEasy', 8, device='cpu', seed=1)
    obs = env.reset(desync=False)
    assert obs.shape == (8, env.obs_dim)
    for t in range(env.ep_len):
        target = env.target()
        obs, reward, done = env.step(target)
        assert torch.allclose(reward, torch.full_like(reward, env.reward_scale))
        assert done.all() == (t == env.ep_len - 1)
    # new episode: previous action/reward features are cleared
    assert obs[:, 4:].abs().sum() == 0
    assert (env.t == 0).all()


@pytest.mark.parametrize('task', ['AutoencodeEasy', 'BattleshipEasy'])
def test_env_state_roundtrip(task):
    from knitwork.env.popgym_batched import PopgymBatchEnv
    Env = CardMemoryEnv if task in TASKS else PopgymBatchEnv
    env, twin = Env(task, 4, device='cpu', seed=3), Env(task, 4, device='cpu', seed=99)
    env.reset()
    actions = torch.randint(env.n_actions, (10, 4), generator=torch.Generator().manual_seed(0))
    for a in actions[:5]:
        env.step(a)
    twin.load_state_dict(env.state_dict())
    for a in actions[5:]:
        out, out_twin = env.step(a), twin.step(a)
        for x, y in zip(out, out_twin):
            assert torch.equal(x, y)
