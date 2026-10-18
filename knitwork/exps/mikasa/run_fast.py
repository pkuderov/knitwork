"""Recurrent PPO on GPU-vectorized POPGym card-memory tasks (see knitwork/env/popgym_torch.py)."""
from __future__ import annotations

import copy
import time

import torch

from knitwork.common.entrypoint import run_experiment
from knitwork.common.logging_alt import start_logger
from knitwork.common.torch import to_loggable_metrics
from knitwork.common.utils import count_learnable_params, get_device
from knitwork.env.popgym_batched import PopgymBatchEnv
from knitwork.env.popgym_torch import TASKS, CardMemoryEnv
from knitwork.models.utils import build_model
from knitwork.rl.ppo_gpu import EpisodeTracker, GraphedPolicy, collect, evaluate, ppo_update


def main(config):
    config = copy.deepcopy(config)
    torch.manual_seed(config['seed'])
    torch.set_float32_matmul_precision('high')
    device = get_device(config.get('device'))
    # resource caps: keep the workstation responsive when several runs share it
    torch.set_num_threads(config.get('cpu_threads', 2))
    if device.type == 'cuda' and config.get('gpu_mem_frac'):
        torch.cuda.set_per_process_memory_fraction(config['gpu_mem_frac'], device.index or 0)
    rl = config['rl']

    # GPU re-implementation when available, else original POPGym stepped on CPU
    Env = CardMemoryEnv if config['env'] in TASKS else PopgymBatchEnv
    env = Env(config['env'], config['n_envs'], device=device, seed=config['seed'], **config.get('env_cfg', {}))
    eval_env = Env(config['env'], config['eval']['n_envs'], device=device, seed=10_000 + config['seed'], **config.get('env_cfg', {}))

    model_key = config['model']
    model = build_model(
        wrapper_type='rl_vector',
        wrapper_cfg=dict(input_size=env.obs_dim, output_size=env.n_actions, dtype=torch.float32, device=device)
        | config.get('rl_wrapper', {}),
        rnn_type=model_key.split('.', 1)[0],
        rnn_cfg=config[model_key.replace('.', '_')],
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=rl['lr'], eps=1e-5, fused=device.type == 'cuda')
    n_steps = int(config['n_steps'])
    slots_per_update = config['n_envs'] * config['rollout_len']
    n_updates = n_steps // slots_per_update

    params = count_learnable_params(model, as_str=True)
    name = config.get('name') or ''
    config['log']['name'] = f'{model_key}_{config["env"]}_{params} {name}'.strip()
    config['protocol'] = dict(version='popgym_torch_ppo_v1', parameters=count_learnable_params(model))
    logger = start_logger(config, suppress_printing=True)
    print(f'Run: {config["log"]["name"]} | obs_dim={env.obs_dim} actions={env.n_actions} ep_len={env.ep_len}')

    try:
        obs = env.reset()
        state = model.rnn.reset_state(None, bsz=config['n_envs'])
        forward = model
        if config.get('cuda_graphs'):
            forward = GraphedPolicy(
                model, state, obs, rollout_len=config['rollout_len'],
                minibatch=config['n_envs'] // rl['minibatches'],
            )
        episodes = EpisodeTracker(config['n_envs'], device)
        step, t0 = 0, time.time()
        for update in range(n_updates):
            if rl.get('lr_anneal'):
                for group in optim.param_groups:
                    group['lr'] = rl['lr'] * (1 - update / n_updates)
            batch, state, obs = collect(env, forward, state, obs, config['rollout_len'])
            metrics = ppo_update(
                forward, batch, optim,
                epochs=rl['epochs'], minibatches=rl['minibatches'], clip_eps=rl['clip_eps'],
                value_coef=rl['value_coef'], entropy_coef=rl['entropy_coef'],
                max_grad_norm=rl['max_grad_norm'], gamma=rl['gamma'], gae_lambda=rl['gae_lambda'],
                target_kl=rl.get('target_kl'), supervised=rl.get('supervised', False),
                reward_mult=rl.get('reward_mult', 1.0),
            )
            if rl.get('refresh_state'):
                state = refresh(forward, batch, config['n_envs'])
            step += slots_per_update

            rewarded = batch['targets'] >= 0
            metrics['Acc'] = (batch['actions'] == batch['targets'])[rewarded].float().mean()
            finished = episodes.update(batch['rewards'], batch['dones'])
            if finished.numel():
                metrics['env/EpRet'] = finished.mean()
            metrics['LR'] = optim.param_groups[0]['lr']
            logger.log(step, to_loggable_metrics(metrics), flush=False)

            if update % config['eval']['every'] == 0 or update == n_updates - 1:
                logger.log(step, {
                    'eval/EpRetGreedy': evaluate(eval_env, model, greedy=True),
                    'eval/EpRet': evaluate(eval_env, model, greedy=False),
                    'perf/sps': step / (time.time() - t0),
                }, flush=False)
            logger.log(step, flush=True, force=update == n_updates - 1)
            if update % config['print_every'] == 0:
                print(
                    f'[{step / 1e6:.2f}M] {step / (time.time() - t0):.0f} sps'
                    f' Acc {metrics["Acc"].item():.3f} H {metrics["H"].item():.3f}'
                    f' KL {metrics["ApproxKL"].item():.4f} EV {metrics["ExplVar"].item():.2f}'
                    + (f' EpRet {metrics["env/EpRet"].item():.3f}' if 'env/EpRet' in metrics else ''),
                    flush=True,
                )
    finally:
        logger.finish()


@torch.no_grad()
def refresh(model, batch, n_envs):
    """Recompute the carried state with the updated weights from the rollout boundary state."""
    state = batch['state_init']
    for obs, done in zip(batch['obs'], batch['dones']):
        _, _, state, _ = model(obs, state)
        state = model.reset_state(state, done)
    return model.detach_state(state)


if __name__ == '__main__':
    run_experiment(runner=main)
