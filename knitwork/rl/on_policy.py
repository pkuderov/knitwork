"""Reusable vectorized on-policy sampling and PPO updates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from gymnasium import spaces

from knitwork.common.torch import to_numpy, to_torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    values: torch.Tensor
    state_init: object
    previous_episode_end: torch.Tensor
    episode_end_before_previous: torch.Tensor


class EpisodeStats:
    """Episode summaries accumulated from valid vector-environment samples."""

    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.returns = np.zeros(n_envs, dtype=np.float64)
        self.lengths = np.zeros(n_envs, dtype=np.int64)
        self.completed_returns = []
        self.completed_lengths = []

    def update(self, batch):
        valid, _ = get_rollout_masks(batch)
        valid = to_numpy(valid)
        rewards = to_numpy(batch.rewards)
        dones = to_numpy(batch.terminated | batch.truncated)
        for t in range(valid.shape[0]):
            self.returns[valid[t]] += rewards[t, valid[t]]
            self.lengths[valid[t]] += 1
            for i in np.where(valid[t] & dones[t])[0]:
                self.completed_returns.append(float(self.returns[i]))
                self.completed_lengths.append(int(self.lengths[i]))
                self.returns[i] = 0.0
                self.lengths[i] = 0

    def get(self):
        if not self.completed_returns:
            return {}
        return {
            'ep_return': float(np.mean(self.completed_returns[-self.n_envs:])),
            'ep_length': float(np.mean(self.completed_lengths[-self.n_envs:])),
        }


def flatten_obs(obs, obs_space):
    """Convert a Gymnasium vector observation to the model's batched input."""
    if isinstance(obs_space, spaces.Discrete):
        return obs.reshape(-1, 1).astype(np.int64)
    if isinstance(obs_space, spaces.MultiDiscrete):
        return obs.reshape(obs.shape[0], -1).astype(np.float32)
    if isinstance(obs_space, spaces.Tuple):
        return np.stack(obs, axis=-1).astype(np.float32)
    if isinstance(obs_space, spaces.Box):
        return obs.reshape(obs.shape[0], -1).astype(np.float32)
    raise ValueError(f'Unsupported observation space: {obs_space}')


def get_rollout_masks(batch):
    """Derive NEXT_STEP masks from episode-end flags and boundary history."""
    episode_end = batch.terminated | batch.truncated
    invalid_reset_step = torch.empty_like(episode_end)
    episode_start = torch.empty_like(episode_end)

    invalid_reset_step[0] = batch.previous_episode_end
    episode_start[0] = batch.episode_end_before_previous
    if len(episode_end) > 1:
        invalid_reset_step[1:] = episode_end[:-1]
        episode_start[1] = batch.previous_episode_end
    if len(episode_end) > 2:
        episode_start[2:] = episode_end[:-2]

    return ~invalid_reset_step, episode_start


@torch.no_grad()
def compute_gae(
        rewards, values, terminated, truncated, valid_masks,
        *, gamma, lambda_,
):
    """Compute GAE and lambda-return targets for vectorized trajectories.

    Reset-only NEXT_STEP slots are excluded. The following model value is the
    terminal-observation value for a time-limit truncation.
    """
    advs = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        valid = valid_masks[t].to(values.dtype)
        bootstrap_mask = (~terminated[t]).to(values.dtype)
        continue_mask = (
            valid_masks[t] & ~(terminated[t] | truncated[t])
        ).to(values.dtype)
        delta = valid * (
            rewards[t] + gamma * values[t + 1] * bootstrap_mask - values[t]
        )
        gae = delta + gamma * lambda_ * continue_mask * gae
        advs[t] = gae
    return advs, advs + values[:-1]


@torch.no_grad()
def sample_batch(
        env, model, state, obs, previous_episode_end, episode_end_before_previous,
        *, obs_space, rollout_len, dtype, is_discrete,
        capture_fn=None, on_step=None,
):
    """Collect one vectorized on-policy rollout and its bootstrap values."""
    n_envs = env.num_envs
    obs_shape = (rollout_len, n_envs, 1) if is_discrete else (
        rollout_len, n_envs, obs.shape[1]
    )
    obs_dtype = torch.int64 if is_discrete else dtype
    device = next(model.parameters()).device
    obs_buf = torch.zeros(obs_shape, dtype=obs_dtype, device=device)
    act_buf = torch.zeros(rollout_len, n_envs, dtype=torch.int64, device=device)
    lp_buf = torch.zeros(rollout_len, n_envs, dtype=dtype, device=device)
    rew_buf = torch.zeros(rollout_len, n_envs, dtype=dtype, device=device)
    term_buf = torch.zeros(rollout_len, n_envs, dtype=torch.bool, device=device)
    trunc_buf = torch.zeros(rollout_len, n_envs, dtype=torch.bool, device=device)
    value_buf = torch.zeros(rollout_len + 1, n_envs, dtype=dtype, device=device)

    state_init = state
    batch_previous_episode_end = to_torch(
        previous_episode_end, device=device,
    ).bool()
    batch_episode_end_before_previous = to_torch(
        episode_end_before_previous, device=device,
    ).bool()
    for t in range(rollout_len):
        recurrent_reset = to_torch(
            episode_end_before_previous, device=device,
        ).bool()
        model_obs = to_torch(obs, device=device)
        model_obs = model_obs.view(-1, 1) if is_discrete else model_obs.to(dtype)

        state = model.reset_state(state, recurrent_reset)
        capture = capture_fn() if capture_fn is not None else False
        logits, value, state, info = model(model_obs, state, capture=capture)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        next_obs, rewards, terminated, truncated, _ = env.step(to_numpy(actions))

        terminated_mask = to_torch(terminated, device=device).bool()
        truncated_mask = to_torch(truncated, device=device).bool()
        obs_buf[t] = model_obs
        act_buf[t] = actions
        lp_buf[t] = dist.log_prob(actions)
        rew_buf[t] = to_torch(rewards, device=device).to(dtype)
        term_buf[t] = terminated_mask
        trunc_buf[t] = truncated_mask
        value_buf[t] = value

        if on_step is not None:
            on_step(state, info, capture)

        obs = flatten_obs(next_obs, obs_space)
        episode_end_before_previous = previous_episode_end
        previous_episode_end = terminated | truncated

    model_obs = to_torch(obs, device=device)
    model_obs = model_obs.view(-1, 1) if is_discrete else model_obs.to(dtype)
    bootstrap_reset = to_torch(episode_end_before_previous, device=device)
    bootstrap_state = model.reset_state(state, bootstrap_reset)
    _, value_buf[-1], _, _ = model(model_obs, bootstrap_state, capture=False)

    batch = RolloutBatch(
        obs=obs_buf,
        actions=act_buf,
        log_probs=lp_buf,
        rewards=rew_buf,
        terminated=term_buf,
        truncated=trunc_buf,
        values=value_buf,
        state_init=state_init,
        previous_episode_end=batch_previous_episode_end,
        episode_end_before_previous=batch_episode_end_before_previous,
    )
    return batch, state, obs, previous_episode_end, episode_end_before_previous


def train_batch(
        model, batch, optimizer, *, ppo_epochs, clip_eps, value_coef,
        entropy_coef, max_grad_norm, gamma, gae_lambda,
        comm_loss_weight=0.0, comm_entropy_weight=0.0,
):
    """Run recurrent PPO updates with BPTT across the complete rollout."""
    valid_masks, state_reset_masks = get_rollout_masks(batch)
    advs, returns = compute_gae(
        batch.rewards,
        batch.values,
        batch.terminated,
        batch.truncated,
        valid_masks,
        gamma=gamma,
        lambda_=gae_lambda,
    )
    n_valid = int(valid_masks.sum().item())
    if n_valid == 0:
        return dict(
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            comm_loss=0.0,
            comm_entropy=0.0,
            grad_norm=0.0,
            mean_reward=0.0,
            n_optimizer_steps=0,
        )

    valid_advs = advs[valid_masks]
    advs = (advs - valid_advs.mean()) / (valid_advs.std(unbiased=False) + 1e-8)

    metrics = dict(
        policy_loss=0.0,
        value_loss=0.0,
        entropy=0.0,
        comm_loss=0.0,
        comm_entropy=0.0,
    )
    grad_norm = None
    n_optimizer_steps = 0
    n_comm = 0
    for _ in range(ppo_epochs):
        state = batch.state_init
        loss_sum = 0.0
        for t in range(batch.obs.shape[0]):
            state = model.reset_state(state, state_reset_masks[t])
            logits, value, state, info = model(batch.obs[t], state, capture=False)
            dist = Categorical(logits=logits)
            valid = valid_masks[t]
            if not valid.any():
                continue
            entropy = dist.entropy()
            ratio = (dist.log_prob(batch.actions[t]) - batch.log_probs[t]).exp()
            policy_loss = -torch.min(
                ratio * advs[t],
                ratio.clamp(1 - clip_eps, 1 + clip_eps) * advs[t],
            )
            value_loss = F.mse_loss(value, returns[t], reduction='none')
            loss_sum = loss_sum + (
                policy_loss[valid].sum()
                + value_coef * value_loss[valid].sum()
                - entropy_coef * entropy[valid].sum()
            )

            comm_loss = comm_entropy = None
            if valid.all() and 'comm_loss' in info:
                comm_loss = torch.stack(info['comm_loss']).mean()
                comm_entropy = torch.stack(info['comm_entropy']).mean()
                loss_sum = loss_sum + valid.sum() * (
                    comm_loss_weight * comm_loss - comm_entropy_weight * comm_entropy
                )
                n_comm += int(valid.sum().item())

            metrics['policy_loss'] += policy_loss[valid].detach().sum().item()
            metrics['value_loss'] += value_loss[valid].detach().sum().item()
            metrics['entropy'] += entropy[valid].detach().sum().item()
            if comm_loss is not None:
                metrics['comm_loss'] += valid.sum().item() * comm_loss.detach().item()
                metrics['comm_entropy'] += valid.sum().item() * comm_entropy.detach().item()

        optimizer.zero_grad()
        (loss_sum / n_valid).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if torch.isfinite(grad_norm):
            optimizer.step()
            n_optimizer_steps += 1
        else:
            print('Nan/Inf grad — step skipped')

    metrics = {
        key: value / (ppo_epochs * n_valid)
        for key, value in metrics.items()
    }
    if n_comm:
        metrics['comm_loss'] *= ppo_epochs * n_valid / n_comm
        metrics['comm_entropy'] *= ppo_epochs * n_valid / n_comm
    metrics['grad_norm'] = grad_norm
    metrics['mean_reward'] = batch.rewards[valid_masks].mean()
    metrics['n_optimizer_steps'] = n_optimizer_steps
    return metrics
