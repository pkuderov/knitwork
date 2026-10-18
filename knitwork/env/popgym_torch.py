"""Batched GPU re-implementations of POPGym card-memory tasks.

Actions never affect observations in these tasks, so an episode is a fixed-length
pair of sequences: observation codes and per-step targets (-1 = unrewarded step).
Semantics (episode length, targets, reward scale) follow popgym 1.x exactly; see
tests/test_popgym_torch.py for the step-by-step equivalence check.
Autoreset is SAME_STEP: the transition with done=True returns the first
observation of the next episode.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

N_SUITS = 4

TASKS = {
    'RepeatPreviousEasy': ('repeat_previous', dict(num_decks=1, k=4)),
    'RepeatPreviousMedium': ('repeat_previous', dict(num_decks=2, k=32)),
    'RepeatPreviousHard': ('repeat_previous', dict(num_decks=3, k=64)),
    'RepeatFirstEasy': ('repeat_first', dict(num_decks=1)),
    'RepeatFirstMedium': ('repeat_first', dict(num_decks=8)),
    'RepeatFirstHard': ('repeat_first', dict(num_decks=16)),
    'AutoencodeEasy': ('autoencode', dict(num_decks=1)),
    'AutoencodeMedium': ('autoencode', dict(num_decks=2)),
    'AutoencodeHard': ('autoencode', dict(num_decks=3)),
    'CountRecallEasy': ('count_recall', dict(num_decks=1, n_types=2)),
    'CountRecallMedium': ('count_recall', dict(num_decks=2, n_types=4)),
    'CountRecallHard': ('count_recall', dict(num_decks=4, n_types=13)),
}


def shuffled_types(n_envs, n_cards, n_types, *, generator, device):
    # balanced multiset of card types, independently shuffled per env: [B, N]
    order = torch.rand(n_envs, n_cards, generator=generator, device=device).argsort(dim=1)
    return order % n_types


class CardMemoryEnv:
    """POPGym card-memory task for n_envs parallel episodes on one device."""

    def __init__(self, task, n_envs, *, device, seed=0, prev_action_reward=True):
        kind, kw = TASKS[task.removeprefix('popgym-').removesuffix('-v0')]
        self.kind, self.kw = kind, kw
        self.n_envs = n_envs
        self.device = torch.device(device)
        self.prev_action_reward = prev_action_reward
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        n_cards = 52 * kw['num_decks']
        if kind == 'repeat_previous':
            self.ep_len, self.reward_scale = n_cards - 1, 1 / (n_cards - kw['k'])
            self.obs_sizes, self.n_actions = (N_SUITS,), N_SUITS
        elif kind == 'repeat_first':
            self.ep_len, self.reward_scale = n_cards - 1, 1 / (n_cards - 1)
            self.obs_sizes, self.n_actions = (N_SUITS,), N_SUITS
        elif kind == 'autoencode':
            self.ep_len, self.reward_scale = 2 * n_cards - 1, 1 / n_cards
            self.obs_sizes, self.n_actions = (2, N_SUITS), N_SUITS
        else:
            self.ep_len, self.reward_scale = n_cards - 1, 1 / (n_cards - 1)
            self.obs_sizes, self.n_actions = (kw['n_types'],) * 2, n_cards // kw['n_types']
        self.n_cards = n_cards
        self.obs_dim = sum(self.obs_sizes) + (self.n_actions + 1 if prev_action_reward else 0)

        B, T = n_envs, self.ep_len
        self.codes = torch.zeros(B, T, len(self.obs_sizes), dtype=torch.long, device=self.device)
        self.targets = torch.zeros(B, T, dtype=torch.long, device=self.device)
        self.t = torch.zeros(B, dtype=torch.long, device=self.device)
        self.prev_action = torch.zeros(B, self.n_actions, device=self.device)
        self.prev_reward = torch.zeros(B, device=self.device)
        self.arange = torch.arange(B, device=self.device)

    def _generate(self, n):
        g, dev = self.generator, self.device
        n_types = self.kw.get('n_types', N_SUITS)
        cards = shuffled_types(n, self.n_cards, n_types, generator=g, device=dev)
        queries = shuffled_types(n, self.n_cards, n_types, generator=g, device=dev) if self.kind == 'count_recall' else None
        return self.episodes(cards, queries)

    def episodes(self, cards, queries=None):
        """Card types in deal order [n, N] -> (codes [n, T, K], targets [n, T])."""
        kw, N, T = self.kw, self.n_cards, self.ep_len
        n = cards.shape[0]
        j = torch.arange(T, device=cards.device)
        if self.kind == 'repeat_previous':
            # decision j sees cards[0..j]; it is rewarded for cards[j + 1 - k] once j + 1 >= k
            src = (j + 1 - kw['k']).clamp(min=0)
            targets = cards[:, src].masked_fill(j + 1 < kw['k'], -1)
            codes = cards[:, :T, None]
        elif self.kind == 'repeat_first':
            targets = cards[:, :1].expand(n, T)
            codes = cards[:, :T, None]
        elif self.kind == 'autoencode':
            # watch: decisions 0..N-2 see (1, cards[j]); decision N-1 sees (0, cards[N-1]); then (0, 0)
            suit = torch.cat([cards, cards.new_zeros(n, T - N)], dim=1)
            mode = (j < N - 1).long().expand(n, T)
            codes = torch.stack([mode, suit], dim=-1)
            src = (2 * N - 2 - j).clamp(0, N - 1)
            targets = cards[:, src].masked_fill(j < N - 1, -1)
        else:
            # decision j: count of queries[j] among cards[0..j]
            counts = F.one_hot(cards[:, :T], kw['n_types']).cumsum(dim=1)
            targets = counts.gather(2, queries[:, :T, None]).squeeze(-1)
            codes = torch.stack([cards[:, :T], queries[:, :T]], dim=-1)
        return codes, targets

    def _encode(self):
        codes = self.codes[self.arange, self.t]
        parts = [F.one_hot(codes[:, i], size).float() for i, size in enumerate(self.obs_sizes)]
        if self.prev_action_reward:
            parts += [self.prev_action, self.prev_reward[:, None]]
        return torch.cat(parts, dim=-1)

    def _new_episodes(self, mask):
        n = int(mask.sum())
        if n == 0:
            return
        self.codes[mask], self.targets[mask] = self._generate(n)
        self.t[mask] = 0
        self.prev_action[mask] = 0.0
        self.prev_reward[mask] = 0.0

    def reset(self, *, desync=True):
        self._new_episodes(torch.ones(self.n_envs, dtype=torch.bool, device=self.device))
        if desync:
            # spread episode phases so rollouts are not phase-locked across envs
            self.t = torch.randint(self.ep_len, (self.n_envs,), generator=self.generator, device=self.device)
        return self._encode()

    def target(self):
        """Rewarded action for the current decision, -1 if unrewarded."""
        return self.targets[self.arange, self.t]

    def step(self, actions):
        target = self.target()
        sign = torch.where(actions == target, 1.0, -1.0).masked_fill(target < 0, 0.0)
        reward = sign * self.reward_scale
        self.prev_action = F.one_hot(actions, self.n_actions).float()
        # reward feature is the sign, POPGym rewards are O(1 / episode length)
        self.prev_reward = sign
        self.t = self.t + 1
        done = self.t >= self.ep_len
        self._new_episodes(done)
        return self._encode(), reward, done
