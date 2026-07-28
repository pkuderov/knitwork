"""PPO experiments on MIKASA/POPGym with the unified core-model API."""
from __future__ import annotations

from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from knitwork.common.entrypoint import run_experiment
from knitwork.common.logging_alt import start_logger
from knitwork.common.numpy import get_seed
from knitwork.common.scheduler import create_scheduler
from knitwork.common.torch import DynamicLearningRate, to_loggable_metrics, to_numpy, to_torch
from knitwork.common.utils import (
    count_learnable_params,
    dont_throw,
    format_readable_num,
    get_device,
    get_dtype,
)
from knitwork.env.mikasa_wrapper import MikasakWrapper
from knitwork.models.utils import build_model


GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4


def compute_gae(rewards, values, dones, terminated):
    advs = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        continue_mask = 1.0 - dones[t]
        bootstrap_mask = 1.0 - terminated[t]
        delta = rewards[t] + GAMMA * values[t + 1] * bootstrap_mask - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * continue_mask * gae
        advs[t] = gae
    return advs, advs + values[:-1]


def main(config):
    torch.set_float32_matmul_precision('high')

    env_id = config.get('env') or config['env-ids'][config['env-id']]
    config['env'] = env_id
    default_name = f'{config["model"]}_{env_id.removeprefix("popgym-").removesuffix("-v0")}'
    name_sfx = config.get('name') or config['log'].get('name') or ''

    rng = np.random.default_rng(config['seed'])
    device = get_device(config.get('device'))
    dtype = get_dtype(config.get('dtype'))
    n_envs = config['n_envs']
    env = MikasakWrapper(
        env_id=env_id,
        n_envs=n_envs,
        seed=get_seed(rng),
        async_envs=config.get('async_envs', False),
    )
    print(
        f'Env: {env_id} | obs_type={env.obs_type} | obs_dim={env.obs_dim}'
        f' | n_tokens={env.n_tokens} | n_actions={env.n_actions}'
    )

    config['model_cfg'] = config['model'].replace('.', '_')
    config['model'] = config['model'].split('.', 1)[0]
    is_discrete = env.obs_type == 'discrete'
    wrapper_type = 'rl_token' if is_discrete else 'rl_vector'
    input_size = env.n_tokens if is_discrete else env.obs_dim
    model = build_model(
        wrapper_type=wrapper_type,
        wrapper_cfg=dict(
            input_size=input_size,
            output_size=env.n_actions,
            dtype=dtype,
            device=device,
        ),
        rnn_type=config['model'],
        rnn_cfg=config[config['model_cfg']],
    )
    model = model.to(device=device, dtype=dtype)
    if config.get('compile', False):
        model = torch.compile(model)
    rnn = model.rnn

    run_name = f'{default_name}_{count_learnable_params(model, as_str=True)} {name_sfx}'
    config['log']['name'] = run_name
    print(f'Run name: {run_name}')

    lr = DynamicLearningRate(name='LR', **config['lr'])
    optim = torch.optim.RMSprop(model.parameters(), lr=lr.val, eps=1e-5)
    lr.connect_to_optimiser(optim)

    rollout_len = config['rollout_len']
    n_steps = int(config['n_steps'])
    entropy_coef = float(config.get('entropy_coef', 0.005))
    communication_cfg = config.get('communication', {})
    comm_loss_weight = float(communication_cfg.get('loss_weight', 0.0))
    comm_entropy_weight = float(communication_cfg.get('entropy_weight', 0.0))

    has_grid = hasattr(rnn, 'n_layers') and hasattr(rnn, 'n_columns')
    inspect_scheduler = create_scheduler(config.get('inspect_schedule'))
    vis_scheduler = create_scheduler(config.get('vis_inspect_schedule'))
    if not vis_scheduler.is_infinite and has_grid:
        from knitwork.visualization.attn_flow import AttnFlowVisualizerNew
        from knitwork.visualization.cka import CKAVisualizerNew
        attn_vis = AttnFlowVisualizerNew(
            n_layers=rnn.n_layers,
            n_columns=rnn.n_columns,
            lr=0.01,
        )
        cka_vis = CKAVisualizerNew(
            n_layers=rnn.n_layers,
            n_columns=rnn.n_columns,
            lr=0.01,
        )

    def inject_visualizations(step, *, scalars, figures):
        if not has_grid or vis_scheduler.is_infinite:
            return
        figures |= attn_vis.get_figures()
        figures |= cka_vis.get_figures()

    logger = start_logger(
        config,
        tracker=config['trackers'],
        suppress_printing=True,
        callbacks=[
            partial(print_short_summary, max_steps=n_steps, lr=lr),
            inject_visualizations,
        ],
    )

    obs_shape = (rollout_len, n_envs, 1) if is_discrete else (rollout_len, n_envs, env.obs_dim)
    obs_dtype = torch.int64 if is_discrete else dtype
    obs_buf = torch.zeros(obs_shape, dtype=obs_dtype, device=device)
    act_buf = torch.zeros(rollout_len, n_envs, dtype=torch.int64, device=device)
    lp_buf = torch.zeros(rollout_len, n_envs, dtype=dtype, device=device)
    rew_buf = torch.zeros(rollout_len, n_envs, dtype=dtype, device=device)
    done_buf = torch.zeros(rollout_len, n_envs, dtype=dtype, device=device)
    term_buf = torch.zeros(rollout_len, n_envs, dtype=dtype, device=device)
    reset_buf = torch.zeros(rollout_len, n_envs, dtype=torch.bool, device=device)
    val_buf = torch.zeros(rollout_len, n_envs, dtype=dtype, device=device)

    step = 0
    state = None
    try:
        while step < n_steps:
            state_init = state
            inspect_due = False

            with torch.no_grad():
                for t in range(rollout_len):
                    raw = env.observe()
                    reset_mask = to_torch(raw['reset_mask'], device=device)
                    obs = to_torch(raw['obs'], device=device)
                    obs = obs.view(-1, 1) if is_discrete else obs.to(dtype)

                    state = model.reset_state(state, reset_mask)
                    capture_vis = vis_scheduler.tick(n_envs)
                    inspect_due |= inspect_scheduler.tick(n_envs)
                    logits, value, state, info = model(obs, state, capture=capture_vis)

                    if capture_vis and has_grid:
                        cka_vis.update(state['h'])
                        if 'attn_weights' in info:
                            attn_vis.update(info['attn_weights'])

                    dist = Categorical(logits=logits)
                    actions = dist.sample()
                    rewards_np, dones_np, terminated_np = env.step(to_numpy(actions))

                    obs_buf[t] = obs
                    act_buf[t] = actions
                    lp_buf[t] = dist.log_prob(actions)
                    rew_buf[t] = to_torch(rewards_np, device=device).to(dtype)
                    done_buf[t] = to_torch(dones_np, device=device).to(dtype)
                    term_buf[t] = to_torch(terminated_np, device=device).to(dtype)
                    reset_buf[t] = reset_mask
                    val_buf[t] = value
                    step += n_envs

                raw = env.observe()
                reset_mask = to_torch(raw['reset_mask'], device=device)
                obs = to_torch(raw['obs'], device=device)
                obs = obs.view(-1, 1) if is_discrete else obs.to(dtype)
                bootstrap_state = model.reset_state(state, reset_mask)
                _, value_last, _, _ = model(obs, bootstrap_state, capture=False)

            values = torch.cat([val_buf, value_last.unsqueeze(0)], dim=0)
            advs, returns = compute_gae(rew_buf, values, done_buf, term_buf)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            policy_loss = value_loss = entropy_mean = 0.0
            comm_loss_mean = comm_entropy_mean = 0.0
            n_updates = 0
            for _ in range(PPO_EPOCHS):
                replay_state = state_init
                for t in range(rollout_len):
                    replay_state = model.reset_state(replay_state, reset_buf[t])
                    logits, value, replay_state, info = model(
                        obs_buf[t],
                        replay_state,
                        capture=False,
                    )
                    dist = Categorical(logits=logits)
                    entropy = dist.entropy().mean()
                    ratio = (dist.log_prob(act_buf[t]) - lp_buf[t]).exp()
                    policy = -torch.min(
                        ratio * advs[t],
                        ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * advs[t],
                    ).mean()
                    value_error = F.mse_loss(value, returns[t])
                    loss = policy + VALUE_COEF * value_error - entropy_coef * entropy

                    if 'comm_loss' in info:
                        comm_loss = torch.stack(info['comm_loss']).mean()
                        comm_entropy = torch.stack(info['comm_entropy']).mean()
                        loss = loss + comm_loss_weight * comm_loss - comm_entropy_weight * comm_entropy
                        comm_loss_mean += comm_loss.detach().item()
                        comm_entropy_mean += comm_entropy.detach().item()

                    optim.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    if torch.isfinite(grad_norm):
                        optim.step()
                    else:
                        print('Nan/Inf grad — step skipped')

                    policy_loss += policy.detach().item()
                    value_loss += value_error.detach().item()
                    entropy_mean += entropy.detach().item()
                    n_updates += 1
                    replay_state = model.detach_state(replay_state)

            state = model.detach_state(state)
            lr.step()

            metrics = {
                'PolicyLoss': policy_loss / n_updates,
                'ValueLoss': value_loss / n_updates,
                'Entropy': entropy_mean / n_updates,
                'MeanReward': rew_buf.mean(),
                '|Grad|': grad_norm,
                'LR': lr.val,
                'Upd': n_updates,
            }
            if comm_loss_mean:
                metrics['L_comm'] = comm_loss_mean / n_updates
                metrics['H_comm'] = comm_entropy_mean / n_updates
            logger.accumulate(to_loggable_metrics(metrics), key='slow')
            logger.accumulate(env.get_stats(), prefix='env', key='fast')

            if has_grid and inspect_due:
                log_col_similarity(rnn, state, logger)
                log_attn_beta(rnn, logger)

            logger.log(step, flush=True)
    finally:
        env.close()

    logger.log(step, flush=True, force=True)
    logger.finish()


@torch.no_grad()
@dont_throw('col_sim')
def log_col_similarity(rnn, state, logger):
    h = state['h']
    acts = h[-1].mean(dim=1)
    acts = acts / acts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = acts @ acts.T
    sim = sim[torch.ones_like(sim, dtype=torch.bool).triu(diagonal=1)]
    logger.accumulate(
        to_loggable_metrics({'max': sim.max(), 'avg': sim.mean()}),
        prefix='col_sim',
        key='fast',
    )


@torch.no_grad()
@dont_throw('attn beta')
def log_attn_beta(rnn, logger):
    if not hasattr(rnn, 'attn'):
        return
    metrics = {
        f'L{li}': attn.pi_logtemp.exp().mean()
        for li, attn in enumerate(rnn.attn)
    }
    logger.accumulate(to_loggable_metrics(metrics), prefix='attn_beta', key='fast')


def print_short_summary(step, *, scalars, figures, max_steps, lr):
    if 'PolicyLoss' not in scalars:
        return
    env_return = scalars.get('env/ep_return')
    env_sfx = f' EpRet:{env_return:.3f}' if env_return is not None else ''
    print(
        f'[{format_readable_num(step)}/{format_readable_num(max_steps, frac=0)}]'
        f' {format_readable_num(scalars["perf/fps"], frac=0)}fps |'
        f' LR:{int(100 * scalars["LR"] / lr.base_val)}%'
        f' PL:{scalars["PolicyLoss"]:.3f}'
        f' VL:{scalars["ValueLoss"]:.3f}'
        f' H:{scalars["Entropy"]:.2f}'
        f' R:{scalars["MeanReward"]:.3f}'
        f'{env_sfx}'
    )


if __name__ == '__main__':
    run_experiment(runner=main)
