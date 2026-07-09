from __future__ import annotations

import importlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from datetime import datetime

from knitwork.common.entrypoint import run_experiment
from knitwork.common.logging import create_logger
from knitwork.common.scheduler import create_scheduler
from knitwork.common.torch import DynamicLearningRate
from knitwork.common.tracker import Tracker
from knitwork.common.utils import (
    FpsCounter, flatten_dict,
    format_readable_num, get_device, get_dtype,
    to_numpy, to_torch,
)

from knitwork.env.treasure_hunt import TreasureHuntEnv, TreasureHuntConfig

_REGISTRY: dict[str, tuple[str, str]] = {
    'rnn':            ('knitwork.models.gru',           'GruBaseline'),
    'grnn':           ('knitwork.models.grnn',           'GridRnn'),
    'grnn_err':       ('knitwork.models.grnn_err',       'GridRnn'),
    'grnn2':          ('knitwork.models.grnn2',          'GridRnn2'),
    'grnn_lru':       ('knitwork.models.grnn_lru',       'GridLRU'),
    'grnn_lru_wide':  ('knitwork.models.grnn_lru',       'GridLRU'),
    'hgrnn':          ('knitwork.models.hgrnn',          'HopfieldGridRnn'),
    'hgrnn_lru':      ('knitwork.models.hgrnn_lru',      'HopfieldGridLRU'),
    'hgrn_grnn':      ('knitwork.models.hgrn_grnn',      'HGRN_GridRnn'),
    'grnn_fw':        ('knitwork.models.grnn_fw',        'GridRnnFW'),
    'grnn_reservoir': ('knitwork.models.grnn_reservoir', 'GridRnnReservoir'),
    'grnn_loss':      ('knitwork.models.grnn_loss',      'GridRnnLoss'),
    'grnn_engram':    ('knitwork.models.engram_grnn',    'EngramGridRnn'),
    'grnn_prec_delta': ('knitwork.models.grnn_prec_delta', 'GridRnnPrecDelta'),
    'grnn_ema_mem':   ('knitwork.models.grnn_ema_mem',   'GridRnnEmaMem'),
    'grnn_delta':     ('knitwork.models.grnn_delta',    'GridDelta'),
    'grnn_delta_wide':('knitwork.models.grnn_delta',    'GridDelta'),
    'grnn_harmonic':  ('knitwork.models.grnn_harmonic', 'HarmonicGridRNN'),
    'grnn_fix':       ('knitwork.models.grnn_fix',      'GridRnnFix'),
    'grnn_fix_v3':    ('knitwork.models.grnn_fix_v3',   'GridRnnFixV3'),
    'grnn_fix_v4':    ('knitwork.models.grnn_fix_v4',   'GridRnnFixV4'),
    'hgrnn_fix_v4':   ('knitwork.models.hgrnn_fix_v4',  'HopfieldGridRnnFixV4'),
    'grnn_fix_v5':    ('knitwork.models.grnn_fix_v5',   'GridRnnFixV5'),
    'hgrnn_fix':      ('knitwork.models.hgrnn_fix',     'HopfieldGridRnnFix'),
}


def build_model(rnn_type: str, rnn_cfg: dict, n_tokens: int, n_actions: int):
    entry = _REGISTRY.get(rnn_type)
    if entry is None:
        raise ValueError(f'Unknown model type: {rnn_type!r}')
    mod_path, cls_name = entry
    cls = getattr(importlib.import_module(mod_path), cls_name)
    return cls(**rnn_cfg, input_size=n_tokens, output_size=n_actions)


# PPO hyperparameters
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
VALUE_COEF   = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
PPO_EPOCHS   = 4


def compute_gae(
    rewards: torch.Tensor,  # [T, B]
    values:  torch.Tensor,  # [T+1, B]  (last = bootstrap)
    dones:   torch.Tensor,  # [T, B]
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE advantages and returns. Both shapes: [T, B]."""
    T = rewards.shape[0]
    advs = torch.zeros_like(rewards)
    gae  = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(T)):
        mask    = 1.0 - dones[t]
        delta   = rewards[t] + GAMMA * values[t + 1] * mask - values[t]
        gae     = delta + GAMMA * GAE_LAMBDA * mask * gae
        advs[t] = gae
    returns = advs + values[:T]
    return advs, returns


def main(config):
    run_name = (
        config.get('name', None)
        or config.get('log', {}).get('name', None)
        or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    config.setdefault('log', {})['name'] = run_name
    print(f'Run name: {run_name}')

    rng    = np.random.default_rng(config['seed'])
    device = get_device(config.get('device', None))
    dtype  = get_dtype(config.get('dtype', None))
    n_envs = config['n_envs']

    # env
    env_cfg_name = config.get('env', 'easy')
    env_raw_cfg  = config['envs'][env_cfg_name]
    env_cfg = TreasureHuntConfig(**env_raw_cfg)
    gen = TreasureHuntEnv(
        cfg=env_cfg,
        n_envs=n_envs,
        seed=int(rng.integers(1_000_000)),
    )
    print(f'Env: TreasureHunt [{env_cfg_name}]  vocab={gen.n_tokens}  actions={gen.V}')

    # model
    rnn_type = config['model']
    rnn_cfg  = config['models'][rnn_type]
    rnn = build_model(rnn_type, rnn_cfg, gen.n_tokens, gen.V)
    rnn = rnn.to(device=device, dtype=dtype)

    # separate linear critic on top of actor hidden state
    actor_hidden = rnn.hidden_size
    critic = nn.Linear(actor_hidden, 1).to(device=device, dtype=dtype)
    print(f'Actor hidden: {actor_hidden}  |  Critic params: {sum(p.numel() for p in critic.parameters())}')

    def extract_h_top(state) -> torch.Tensor:
        # normalize heterogeneous state shapes to [B, actor_hidden]
        h = state[0] if isinstance(state, tuple) else state
        if h.ndim == 3:
            # GRU: [layers, B, H]
            return h[-1][:, :actor_hidden]
        # Grid: [layers, cols, B, H or 2H]
        return h[-1, 0, :, :actor_hidden]

    all_params = list(rnn.parameters()) + list(critic.parameters())

    lr = DynamicLearningRate(name=f'LR', **config['lr'])
    optim = torch.optim.RMSprop(all_params, lr=lr.val, eps=1e-5)
    lr.connect_to_optimiser(optim)

    rollout_len = config['rollout_len']
    n_steps     = int(config['n_steps'])
    step        = 0

    log_stats_schedule = create_scheduler(config['log']['schedule'])
    print_stats_schedule = create_scheduler(config['log']['print_schedule'])

    logger = create_logger(config)
    if logger is not None:
        config.setdefault('log', {})['name'] = run_name

    stats       = Tracker(lr=2e-4)
    fps_counter = FpsCounter()

    rnn_state = None

    while step < n_steps:
        # rollout collection
        buf_obs       = []  # [T, B, 1]
        buf_actions   = []  # [T, B]
        buf_log_probs = []  # [T, B]
        buf_rewards   = []  # [T, B]
        buf_dones     = []  # [T, B]
        buf_values    = []  # [T, B]

        h_init = rnn_state  # hidden state at rollout start (reused in PPO epochs)

        with torch.no_grad():
            for _ in range(rollout_len):
                obs = gen.observe()
                reset_mask = to_torch(obs['reset_mask'], device=device)
                tokens     = to_torch(obs['tokens'],     device=device)

                rnn_state = rnn.reset_state(rnn_state, reset_mask)
                logits, rnn_state = rnn(tokens.view(-1, 1), rnn_state)

                value = critic(extract_h_top(rnn_state)).squeeze(-1)  # [B]

                dist      = Categorical(F.softmax(logits, dim=-1))
                actions   = dist.sample()             # [B]
                log_probs = dist.log_prob(actions)    # [B]

                rewards_np, dones_np = gen.step(to_numpy(actions))
                rewards = to_torch(rewards_np, device=device).to(dtype)
                dones   = to_torch(dones_np,   device=device).to(dtype)

                buf_obs.append(tokens)
                buf_actions.append(actions)
                buf_log_probs.append(log_probs)
                buf_rewards.append(rewards)
                buf_dones.append(dones)
                buf_values.append(value)

                step += n_envs

            # bootstrap value for last state
            obs = gen.observe()
            reset_mask = to_torch(obs['reset_mask'], device=device)
            tokens_last = to_torch(obs['tokens'], device=device)
            rnn_state = rnn.reset_state(rnn_state, reset_mask)
            _, rnn_state_last = rnn(tokens_last.view(-1, 1), rnn_state)
            value_last = critic(extract_h_top(rnn_state_last)).squeeze(-1).detach()

        obs_t     = torch.stack(buf_obs,       dim=0)  # [T, B, 1]
        actions_t = torch.stack(buf_actions,   dim=0)  # [T, B]
        old_lp_t  = torch.stack(buf_log_probs, dim=0)  # [T, B]
        rewards_t = torch.stack(buf_rewards,   dim=0)  # [T, B]
        dones_t   = torch.stack(buf_dones,     dim=0)  # [T, B]
        values_t  = torch.stack(buf_values,    dim=0)  # [T, B]

        values_with_boot = torch.cat([values_t, value_last.unsqueeze(0)], dim=0)  # [T+1, B]

        advs, returns = compute_gae(rewards_t, values_with_boot, dones_t)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # PPO update
        total_policy_loss = total_value_loss = total_entropy = 0.0
        n_updates = 0

        for _ in range(PPO_EPOCHS):
            h = h_init
            epoch_policy_loss = epoch_value_loss = epoch_entropy = 0.0

            for t in range(rollout_len):
                reset_mask_t = dones_t[t - 1] if t > 0 else torch.zeros(n_envs, device=device)
                h = rnn.reset_state(h, reset_mask_t)

                logits, h = rnn(obs_t[t].view(-1, 1), h)

                value_new = critic(extract_h_top(h)).squeeze(-1)
                dist_new  = Categorical(F.softmax(logits, dim=-1))
                new_lp    = dist_new.log_prob(actions_t[t])
                entropy   = dist_new.entropy().mean()

                ratio = (new_lp - old_lp_t[t]).exp()
                adv_t = advs[t]
                p_loss = -torch.min(
                    ratio * adv_t,
                    ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t,
                ).mean()
                v_loss = F.mse_loss(value_new, returns[t])

                loss = p_loss + VALUE_COEF * v_loss - ENTROPY_COEF * entropy
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, MAX_GRAD_NORM)
                optim.step()

                epoch_policy_loss += p_loss.item()
                epoch_value_loss  += v_loss.item()
                epoch_entropy     += entropy.item()
                n_updates += 1

                h = rnn.detach_state(h)

            total_policy_loss += epoch_policy_loss
            total_value_loss  += epoch_value_loss
            total_entropy     += epoch_entropy

        rnn_state = rnn.detach_state(rnn_state)

        lr.step()

        stats.put({
            'PolicyLoss' : total_policy_loss / max(n_updates, 1),
            'ValueLoss'  : total_value_loss  / max(n_updates, 1),
            'Entropy'    : total_entropy     / max(n_updates, 1),
            'MeanReward' : to_numpy(rewards_t.mean()),
            'OpenedFrac' : float(np.mean([gen._opened[i].mean() for i in range(gen.n_envs)])),
            'LR'         : lr.val,
        })

        if print_stats_schedule.tick(n_envs * rollout_len):
            m = stats.get()
            fps = fps_counter.fps(n_iters=step, start=True)
            print(
                f'[{format_readable_num(step)} / {format_readable_num(n_steps, frac=0)}]'
                f' {format_readable_num(fps, frac=0)} fps |'
                f' LR:{int(100*m["LR"]/lr.base_val)}% |'
                f' PL:{m["PolicyLoss"]:.3f}'
                f' VL:{m["ValueLoss"]:.3f}'
                f' H:{m["Entropy"]:.2f}'
                f' R:{m["MeanReward"]:.3f}'
                f' Open:{m["OpenedFrac"]:.3f}'
            )

        if log_stats_schedule.tick(n_envs * rollout_len) and logger is not None:
            fps = fps_counter.fps(n_iters=step, start=True)
            metrics = {'global_step': step, 'fps': fps} | stats.get()
            metrics['env'] = gen.get_stats()
            logger.track(flatten_dict(metrics))

    fps = fps_counter.fps(n_iters=step)
    print(format_readable_num(fps))


if __name__ == '__main__':
    run_experiment(runner=main)
