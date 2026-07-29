"""PPO experiments on MIKASA/POPGym with the unified core-model API."""
from __future__ import annotations

from functools import partial

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from knitwork.common.entrypoint import run_experiment
from knitwork.common.logging_alt import start_logger
from knitwork.common.numpy import get_seed
from knitwork.common.scheduler import create_scheduler
from knitwork.common.torch import DynamicLearningRate, to_loggable_metrics
from knitwork.common.utils import (
    count_learnable_params,
    dont_throw,
    format_readable_num,
    get_device,
    get_dtype,
)
from knitwork.models.utils import build_model
from knitwork.rl.on_policy import EpisodeStats, flatten_obs, prep_obs, sample_batch, train_batch

# Side-effect import: registers all POPGym environment IDs.
import popgym  # noqa: F401

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
    VecCls = AsyncVectorEnv if config.get('async_envs', False) else SyncVectorEnv
    env = VecCls(
        [lambda: gym.make(env_id) for _ in range(n_envs)],
    )

    obs_space = env.single_observation_space
    act_space = env.single_action_space
    is_discrete = isinstance(obs_space, spaces.Discrete)
    n_tokens = int(obs_space.n) if is_discrete else 0
    if not isinstance(act_space, spaces.Discrete):
        raise ValueError(f'Only Discrete action spaces supported; got {act_space}')
    n_actions = int(act_space.n)

    obs, _ = env.reset(seed=get_seed(rng))
    obs = prep_obs(obs, obs_space, device, is_discrete, dtype)
    obs_dim = obs.shape[1]
    done = torch.zeros(n_envs, dtype=torch.bool, device=device)
    episode_stats = EpisodeStats(n_envs)
    print(
        f'Env: {env_id} | obs_type={"discrete" if is_discrete else "continuous"}'
        f' | obs_dim={obs_dim} | n_tokens={n_tokens} | n_actions={n_actions}'
    )

    config['model_cfg'] = config['model'].replace('.', '_')
    config['model'] = config['model'].split('.', 1)[0]
    wrapper_type = 'rl_token' if is_discrete else 'rl_vector'
    input_size = n_tokens if is_discrete else obs_dim
    model = build_model(
        wrapper_type=wrapper_type,
        wrapper_cfg=dict(
            input_size=input_size,
            output_size=n_actions,
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
    rl_cfg = config['rl']
    communication_cfg = config.get('communication', {})
    comm_loss_weight = float(communication_cfg.get('loss_weight', 0.0))
    comm_entropy_weight = float(communication_cfg.get('entropy_weight', 0.0))

    has_grid = hasattr(rnn, 'n_layers') and hasattr(rnn, 'n_columns')
    inspect_scheduler = create_scheduler(config.get('inspect_schedule'))
    vis_scheduler = create_scheduler(config.get('vis_inspect_schedule'))
    if not vis_scheduler.is_infinite and has_grid:
        from knitwork.visualization.attn_flow import AttnFlowVisualizerNew
        from knitwork.visualization.cka import CKAVisualizerNew
        attn_vis = AttnFlowVisualizerNew(n_layers=rnn.n_layers, n_columns=rnn.n_columns, lr=0.01)
        cka_vis = CKAVisualizerNew(n_layers=rnn.n_layers, n_columns=rnn.n_columns, lr=0.01)
        vis_draw_scheduler = create_scheduler(4)

    def inject_visualizations(step, *, scalars, figures):
        if not has_grid or vis_scheduler.is_infinite:
            return
        if vis_draw_scheduler.tick():
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

    step = 0
    state = model.rnn.reset_state(None, bsz=n_envs)
    while step < n_steps:
        inspect_due = False
        def capture_fn():
            nonlocal inspect_due
            inspect_due |= inspect_scheduler.tick(n_envs)
            return vis_scheduler.tick(n_envs)

        def on_sample_step(sampled_state, info, capture):
            if not capture or not has_grid:
                return
            cka_vis.update(sampled_state['h'])
            if 'attn_weights' in info:
                attn_vis.update(info['attn_weights'])

        batch, state, obs, done = sample_batch(
            env,
            model,
            state,
            obs,
            done,
            obs_space=obs_space,
            rollout_len=rollout_len,
            dtype=dtype,
            is_discrete=is_discrete,
            capture_fn=capture_fn,
            on_step=on_sample_step,
        )
        train_metrics = train_batch(
            model,
            batch,
            optim,
            ppo_epochs=rl_cfg['ppo_epochs'],
            clip_eps=rl_cfg['clip_eps'],
            value_coef=rl_cfg['value_coef'],
            entropy_coef=rl_cfg['entropy_coef'],
            max_grad_norm=rl_cfg['max_grad_norm'],
            gamma=rl_cfg['gamma'],
            gae_lambda=rl_cfg['gae_lambda'],
            comm_loss_weight=comm_loss_weight,
            comm_entropy_weight=comm_entropy_weight,
        )
        step += n_envs * rollout_len
        episode_stats.update(batch)

        state = model.detach_state(state)
        lr.step()

        metrics = train_metrics | {
            'LR': lr.val,
        }
        logger.accumulate(metrics, key='slow')
        logger.accumulate(episode_stats.get(), prefix='env', key='fast')

        if has_grid and inspect_due:
            log_col_similarity(rnn, state, logger)
            log_attn_beta(rnn, logger)

        logger.log(step, flush=True)

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
    if 'L_pi' not in scalars:
        return
    env_return = scalars.get('env/EpRet')
    env_sfx = f' EpRet:{env_return:.3f}' if env_return is not None else ''
    print(
        f'[{format_readable_num(step)}/{format_readable_num(max_steps, frac=0)}]'
        f' {format_readable_num(scalars["perf/fps"], frac=0)}fps |'
        f' LR:{int(100 * scalars["LR"] / lr.base_val)}%'
        f' PL:{scalars["L_pi"]:.3f}'
        f' VL:{scalars["L_v"]:.3f}'
        f' H:{scalars["H"]:.2f}'
        f' R:{scalars["Rew"]:.3f}'
        f'{env_sfx}'
    )


if __name__ == '__main__':
    run_experiment(runner=main)
