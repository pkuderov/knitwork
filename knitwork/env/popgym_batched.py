"""Original POPGym environments behind the CardMemoryEnv batch API (CPU stepping, tensors on device).

Used for action-dependent tasks (Battleship, MineSweeper, Concentration, ...).
SAME_STEP autoreset; time-limit truncation is treated as episode end without bootstrap.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import popgym  # noqa: F401


class PopgymBatchEnv:
    def __init__(self, task, n_envs, *, device, seed=0, prev_action_reward=True):
        env_id = task if task.startswith('popgym-') else f'popgym-{task}-v0'
        self.envs = [gym.make(env_id).unwrapped for _ in range(n_envs)]
        self.n_envs = n_envs
        self.device = torch.device(device)
        self.prev_action_reward = prev_action_reward
        self.rng = np.random.default_rng(seed)

        e = self.envs[0]
        self.obs_space = e.observation_space
        act = e.action_space
        # MultiDiscrete coordinates -> one categorical index (same bijection as popgym DiscreteAction)
        self.act_nvec = tuple(act.nvec.flat) if isinstance(act, spaces.MultiDiscrete) else None
        self.n_actions = int(np.prod(self.act_nvec)) if self.act_nvec else int(act.n)
        self.obs_base = spaces.flatdim(self.obs_space)
        self.obs_dim = self.obs_base + (self.n_actions + 1 if prev_action_reward else 0)
        # step cap for evaluation episodes
        self.ep_len = int(getattr(e, 'max_episode_length', None) or getattr(e, 'episode_length', None) or 1000)
        self.obs = np.zeros((n_envs, self.obs_dim), dtype=np.float32)

    def _set_obs(self, i, obs, action=None, reward=0.0):
        row = self.obs[i]
        row[:] = 0.0
        row[:self.obs_base] = spaces.flatten(self.obs_space, obs)
        if self.prev_action_reward and action is not None:
            row[self.obs_base + action] = 1.0
            row[-1] = np.sign(reward)

    def reset(self, *, desync=True):
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=int(self.rng.integers(2**31)))
            self._set_obs(i, obs)
        return torch.tensor(self.obs, device=self.device)

    def target(self):
        # no oracle action for these tasks
        return torch.full((self.n_envs,), -1, dtype=torch.long, device=self.device)

    def step(self, actions):
        actions = actions.cpu().numpy()
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            env_a = np.array(np.unravel_index(a, self.act_nvec)) if self.act_nvec else int(a)
            obs, r, term, trunc, _ = env.step(env_a)
            rewards[i], dones[i] = r, term or trunc
            if dones[i]:
                obs, _ = env.reset(seed=int(self.rng.integers(2**31)))
                self._set_obs(i, obs)
            else:
                self._set_obs(i, obs, a, r)
        dev = self.device
        return torch.tensor(self.obs, device=dev), torch.tensor(rewards, device=dev), torch.tensor(dones, device=dev)
