"""Reusable vectorized on-policy sampling and PPO updates."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from gymnasium import spaces

from knitwork.common.torch import to_loggable_metrics, to_numpy, to_torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    term: torch.Tensor
    trunc: torch.Tensor
    reset: torch.Tensor
    # for the batch start
    state_init: object
    # for the batch continuation
    prev_batch_done: torch.Tensor


class EpisodeStats:
    """Episode summaries accumulated from valid vector-environment samples."""

    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.returns = None
        self.lengths = None
        self.completed_returns = deque(maxlen=n_envs)
        self.completed_lengths = deque(maxlen=n_envs)

    @torch.no_grad()
    def update(self, batch: RolloutBatch):
        rollout_len, n_envs = batch.actions.shape[:2]
        reset, rewards = batch.reset, batch.rewards

        if self.returns is None:
            self.returns = batch.rewards.new_zeros(n_envs)
            self.lengths = batch.rewards.new_zeros(n_envs, dtype=torch.int64)

        for t in range(rollout_len):
            ix_reset = torch.nonzero(reset[t]).flatten()
            self.completed_returns.extend(self.returns[ix_reset].cpu().tolist())
            self.completed_lengths.extend(self.lengths[ix_reset].cpu().tolist())
            self.returns += rewards[t]
            self.lengths += 1
            self.returns[ix_reset] = 0.0
            self.lengths[ix_reset] = 0

    def get(self):
        if not self.completed_returns:
            return {}
        return {
            'EpRet': np.mean(self.completed_returns),
            'EpLen': np.mean(self.completed_lengths),
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


def prep_obs(obs, obs_space, device, is_discrete, dtype):
    obs = to_torch(flatten_obs(obs, obs_space), device=device)
    obs = obs.view(-1, 1) if is_discrete else obs.to(dtype)
    return obs


@torch.no_grad()
def sample_batch(
        env, model, state, obs, done,
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
    value_buf = torch.zeros(rollout_len + 1, n_envs, dtype=dtype, device=device)
    term_buf = torch.zeros(rollout_len, n_envs, dtype=torch.bool, device=device)
    trunc_buf = torch.zeros(rollout_len, n_envs, dtype=torch.bool, device=device)
    reset_buf = torch.zeros(rollout_len, n_envs, dtype=torch.bool, device=device)

    state_init = state
    # state = model.reset_state(state, done)

    for t in range(rollout_len):
        # from the last step, now it means reset
        reset_mask = done
        capture = capture_fn() if capture_fn is not None else False

        logits, value, state, info = model(obs, state, capture=capture)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        next_obs, rewards, term, trunc, _ = env.step(to_numpy(actions))

        term = to_torch(term, device=device).bool()
        trunc = to_torch(trunc, device=device).bool()
        next_obs = prep_obs(next_obs, obs_space, device, is_discrete, dtype)

        obs_buf[t] = obs
        act_buf[t] = actions
        lp_buf[t] = dist.log_prob(actions)
        rew_buf[t] = to_torch(rewards, device=device).to(dtype)
        value_buf[t] = value
        term_buf[t] = term
        trunc_buf[t] = trunc
        reset_buf[t] = reset_mask

        if on_step is not None:
            on_step(state, info, capture)

        state = model.reset_state(state, reset_mask)
        obs = next_obs
        done = term | trunc

    _, value_buf[-1], _, _ = model(obs, state, capture=False)

    batch = RolloutBatch(
        obs=obs_buf,
        actions=act_buf,
        log_probs=lp_buf,
        rewards=rew_buf,
        values=value_buf,
        term=term_buf,
        trunc=trunc_buf,
        reset=reset_buf,

        state_init=state_init,
        prev_batch_done=done,
    )
    return batch, state, obs, done


def train_batch(
        model, batch: RolloutBatch, optimizer, *, ppo_epochs, clip_eps, value_coef,
        entropy_coef, max_grad_norm, gamma, gae_lambda,
        comm_loss_weight=0.0, comm_entropy_weight=0.0,
):
    """Run recurrent PPO updates with BPTT across the complete rollout."""
    rollout_len = batch.obs.shape[0]
    comm_loss_enabled = None

    is_valid = ~batch.reset
    advs, returns = compute_gae(
        batch.rewards, batch.values,
        batch.term, batch.trunc, is_valid,
        gamma=gamma, lambda_=gae_lambda,
    )

    valid_advs = advs[is_valid]
    advs = (advs - valid_advs.mean()) / (valid_advs.std(unbiased=False) + 1e-8)

    n_optimizer_steps = 0
    for _ in range(ppo_epochs):
        policy_loss=0.0
        value_loss=0.0
        entropy=0.0
        comm_loss=0.0
        comm_entropy=0.0
        state = batch.state_init

        for t in range(batch.obs.shape[0]):
            logits, value, state, info = model(batch.obs[t], state, capture=False)
            state = model.reset_state(state, batch.reset[t])
            dist = Categorical(logits=logits)
            valid = is_valid[t]
            if not valid.any():
                continue

            ratio = (dist.log_prob(batch.actions[t]) - batch.log_probs[t]).exp()
            policy_loss += -torch.min(
                ratio * advs[t],
                ratio.clamp(1 - clip_eps, 1 + clip_eps) * advs[t],
            )[valid].mean()

            entropy += dist.entropy()[valid].mean()
            value_loss += F.mse_loss(value, returns[t], reduction='none')[valid].mean()

            if comm_loss_enabled or (comm_loss_enabled is None and 'comm_loss' in info):
                comm_loss_enabled = True
                comm_loss += torch.stack(info['comm_loss']).mean()
                comm_entropy += torch.stack(info['comm_entropy']).mean()

        policy_loss /= rollout_len
        value_loss /= rollout_len
        entropy /= rollout_len
        comm_loss /= rollout_len
        comm_entropy /= rollout_len

        loss = (
            policy_loss
            + value_coef * value_loss
            - entropy_coef * entropy
            + comm_loss_weight * comm_loss
            - comm_entropy_weight * comm_entropy
        )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if torch.isfinite(grad_norm):
            optimizer.step()
            n_optimizer_steps += 1
        else:
            print('Nan/Inf grad — step skipped')

    metrics = {
        'L_pi': policy_loss,
        'L_v': value_loss,
        'H': entropy,
        'L_comm': comm_loss,
        'H_comm': comm_entropy,
        '|Grad|': grad_norm,
        'Rew': batch.rewards[is_valid].mean(),
        'Upd': n_optimizer_steps,
    }
    metrics = to_loggable_metrics(metrics)
    return metrics
