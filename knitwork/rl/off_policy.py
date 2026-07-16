from __future__ import annotations

import copy
from collections import defaultdict
from types import SimpleNamespace
from typing import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from knitwork.common.dynamic_param import DynamicParameter
from knitwork.common.numpy import stochastic_round
from knitwork.common.torch import DynamicLearningRate, chain, fw, make_layers, to_loggable_metrics, to_softmax_distr
from knitwork.rl.nn_modules import AdaptiveTemperature, MlpEncoder


class Critic(nn.Module):
    def __init__(
            self, *,
            input_size: int, act_size: int, 
            body: Sequence[int] = (), fn_act=nn.SiLU,
            rng: torch.Generator = None,
    ):
        super().__init__()
        _, self.val = make_layers(
            name='Q-value', input_size=input_size, layers=chain(body, act_size), 
            activation=fn_act, out_logits=True, is_output=True, rng=rng,
        )

    def forward(self, x):
        v = fw(self.val, x)
        return v


class Model(nn.Module):
    def __init__(
            self, *,
            obs_size: int, act_size: int,
            encoder: dict, critic: dict, temperature: dict,
            fn_act=nn.SiLU,
            rng: torch.Generator = None,
    ):
        super().__init__()

        self.obs_size = obs_size
        self.action_size = act_size

        self.encoder = MlpEncoder(obs_size=obs_size, rng=rng, fn_act=fn_act, **encoder)
        self.encoder_target = copy.deepcopy(self.encoder)
        enc_size = self.encoder.enc_size

        self.Q = Critic(input_size=enc_size, act_size=act_size, rng=rng, fn_act=fn_act, **critic)
        self.Q_target = copy.deepcopy(self.Q)

        self.log_temp = AdaptiveTemperature(input_size=enc_size, rng=rng, fn_act=fn_act, **temperature)

    def trainable_parameters(self, key):
        if key == 'q':
            return self.critic_parameters()
        elif key == 'temp':
            return self.temperature_parameters()
        else:
            raise ValueError(f'Unknown active_parameters key: {key}')

    def critic_parameters(self):
        yield from self.encoder.parameters()
        yield from self.Q.parameters()

    def temperature_parameters(self):
        return self.log_temp.parameters()

    def forward(self, x):
        pass


class Agent:
    def __init__(
            self, *,
            model: dict,

            # learning
            batch_size: int,
            lr_q: dict, lr_temp: dict, lr_target: float = 0.005,
            utd: float = 1.0,
            adamw_betas: tuple[float, float] = (0.9, 0.999),
            max_grad_norm: float = None,

            # RL/Losses
            discount_gamma: float = 0.99,
            target_entropy: dict,
            soft_q: int = 0,

            # general
            device=None,
            seed: int = None,
            dtype=torch.float32,
    ):
        self.device = device
        self.dtype = dtype
        self.seed = seed
        # for small on-cpu sampling
        self.np_cpu_rng = np.random.default_rng(seed)

        cpu_rng = torch.Generator()
        if seed is not None:
            cpu_rng.manual_seed(seed)

        self.model = Model(**model, rng=cpu_rng).to(device, dtype=dtype)
        self.obs_size = self.model.obs_size
        self.action_size = self.model.action_size

        self.lr, self.optimizer = {}, {}
        lr_optimizer_list = [('q', lr_q), ('temp', lr_temp)]
        for key, lr in lr_optimizer_list:
            self.lr[key] = DynamicLearningRate(name=f'LR_{key}', **lr)
            self.optimizer[key] = torch.optim.AdamW(
                self.model.trainable_parameters(key), lr=self.lr[key].val, betas=adamw_betas
            )
            self.lr[key].connect_to_optimiser(self.optimizer[key])

        self.lr_target = lr_target

        self.utd = utd
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size

        self.discount_gamma = discount_gamma
        self.gamma_invariance_scale = 1.0 - discount_gamma

        self.target_entropy = DynamicParameter(name="H_t scale", **target_entropy)
        self.target_entropy_tensor = torch.tensor(self.target_entropy.val, device=self.device, dtype=self.dtype)

        v_computation = [
            'V(s) = Q_tar(s, argmax Q(s, a))',
            'V(s) = temp * LSE(Q_tar/temp)', 
            'V(s) = temp * LSE(minimum(Q, Q_tar)/temp)',
            'V(s) = E_pi[Q_tar(s, a) - temp * log pi(a|s)], wher pi = softmax(Q/temp)'
        ]
        is_soft_q = ['NO', 'YES'][soft_q > 0]
        self.soft_q = soft_q
        print(f'Soft Q ({is_soft_q}). {v_computation[self.soft_q]}')

    @torch.no_grad()
    def act(self, obs, temp_rescale=1.0, with_info=False):
        e = self.model.encoder(obs)
        q = self.model.Q(e)
        softmax_temp = self.model.log_temp(e).exp()

        if temp_rescale > 0.0 and temp_rescale != 1.0:
            softmax_temp = softmax_temp * temp_rescale

        pi = to_softmax_distr(q, softmax_temp)
        a = pi.sample() if temp_rescale > 0.0 else q.argmax(dim=-1)

        if not with_info:
            return a

        info = {
            'a': a,
            'Q': q,
            'H': pi.entropy(),
        }
        return a, info

    def update(self, batch):
        data = get_dqn_loss(
            o_t=batch.o_t, a_t=batch.a_t, r_t=batch.r_t, o_tn=batch.o_tn,
            encoder=self.model.encoder, encoder_tar=self.model.encoder_target,
            Q=self.model.Q, Q_tar=self.model.Q_target, log_temp=self.model.log_temp,

            entropy_tar=self.target_entropy_tensor,
            discount_gamma=self.discount_gamma, gamma_invariance_scale=self.gamma_invariance_scale, 
            soft_q=self.soft_q
        )

        for opt in self.optimizer.values():
            opt.zero_grad()

        loss_q, loss_temp = data['L_q'], data['L_t']
        loss = loss_q + loss_temp
        loss.backward()

        grad_norm_q = self.maybe_clip_grad_norm(self.model.trainable_parameters('q'))
        grad_norm_temp = self.maybe_clip_grad_norm(self.model.trainable_parameters('temp'))

        if not torch.stack([grad_norm_q, grad_norm_temp], -1).isfinite().all():
            print(f'Non-finite loss: {grad_norm_q=}, {grad_norm_temp=}')
            return {}

        self.optimizer['q'].step()
        self.optimizer['temp'].step()
        
        self.lr['q'].step()
        self.lr['temp'].step()
        if self.target_entropy.step():
            self.target_entropy_tensor.fill_(self.target_entropy.val)

        self.update_target(base=self.model.encoder, target=self.model.encoder_target)
        self.update_target(base=self.model.Q, target=self.model.Q_target)

        # save stats for logging
        stats_keys = [
            'grad_norm_q', 'grad_norm_t', 'lr_q', 'lr_t'
        ]
        stats_vals = [
            grad_norm_q, grad_norm_temp, self.lr['q'].val, self.lr['temp'].val
        ]
        stats = dict(zip(stats_keys, stats_vals))

        # prepend all precomputed losses, values and metrics, 
        stats = data | stats
        # convert to loggable format (averaged, detached, on cpu)
        return to_loggable_metrics(stats)

    def update_many(self, rb: EpisodicReplayBuffer, n_new_data: int):
        bsz = self.batch_size
        if rb.size < 20 * bsz:
            return []

        fr = min(rb.size, rb.enough_size) / rb.enough_size
        fr = fr ** 0.5
        utd = self.utd
        n_updates = stochastic_round(utd * n_new_data * fr / bsz, rng=self.np_cpu_rng)

        return [self.update(rb.sample(bsz)) for _ in range(n_updates)]


    @torch.no_grad()
    def update_target(self, base, target):
        lr = self.lr_target
        with torch.no_grad():
            for p, p_targ in zip(base.parameters(), target.parameters()):
                p_targ.data.lerp_(p.data, lr)

    def maybe_clip_grad_norm(self, parameters):
        max_norm = self.max_grad_norm if self.max_grad_norm is not None else 1e+6
        return nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


@torch.compile(mode="reduce-overhead")
def get_dqn_loss(
        o_t, a_t, r_t, o_tn,
        encoder, encoder_tar, 
        Q, Q_tar, log_temp,

        entropy_tar, discount_gamma, gamma_invariance_scale, soft_q
):
    # critic loss
    with torch.no_grad():
        e_tn = encoder_tar(o_tn)
        q_tar_tn = Q_tar(e_tn)
        eff_temp_tn = gamma_invariance_scale * log_temp(e_tn).exp()
        eff_r_t = gamma_invariance_scale * r_t

        if soft_q == 0:
            # DDQN
            a_tn = Q(e_tn).argmax(dim=-1, keepdim=True)
            v_tn = q_tar_tn.gather(-1, a_tn)
        elif soft_q == 1:
            # V(s) = temp * LSE(Q_tar/temp), it's regular Soft-Q
            v_tn = eff_temp_tn * torch.logsumexp(q_tar_tn / eff_temp_tn, dim=-1, keepdim=True)
        elif soft_q == 2:
            # V(s) = temp * LSE(minimum(Q, Q_tar)/temp), it mimics SAC regularization
            q_tn = Q(e_tn)
            v_tn = eff_temp_tn * torch.logsumexp(
                torch.minimum(q_tn, q_tar_tn) / eff_temp_tn, dim=-1, keepdim=True
            )
        elif soft_q == 3:
            # V(s) = E_pi[Q_tar(s, a) - temp * log pi(a|s)], where pi = softmax(Q/temp), 
            # it's DDQN analogue for Soft-Q; 
            # NB: expectation is computed explicitly (not sampled as in SAC)
            # NB2: in case of multi-dim actions, entropy is averaged
            q_tn = Q(e_tn)
            pi_tn = to_softmax_distr(q_tn, softmax_temp=eff_temp_tn)
            v_tn = torch.sum(q_tar_tn * pi_tn.probs, dim=-1, keepdim=True) + eff_temp_tn * pi_tn.entropy().unsqueeze(-1)
        else:
            raise ValueError(f'{soft_q}')

        # zero out terminal values if needed: 
        # v_tn = torch.where(terminated, torch.zeros_like(v_tn), v_tn)
        target_q_t = eff_r_t + discount_gamma * v_tn

    e_t = encoder(o_t)
    q_t = Q(e_t)
    q_a_t = q_t.gather(-1, a_t)

    log_temp_t = log_temp(e_t.detach())
    with torch.no_grad():
        # temp_tar = log_temp_tar(e_t).exp()
        temp_t = log_temp_t.exp()
        pi_t = to_softmax_distr(q_t, softmax_temp=temp_t)
        # NB: in case of multi-dim actions, entropy is averaged
        h_t = pi_t.entropy()

    loss_q = nn.functional.huber_loss(q_a_t, target_q_t)

    loss_temp = log_temp_t * (h_t.unsqueeze(-1) - entropy_tar)
    loss_temp = loss_temp.mean()

    # save stats for logging
    with torch.no_grad():
        act = a_t.squeeze(-1).to(q_t.dtype)
        nz_mask = act > 0.0
        nz_cnt = torch.count_nonzero(nz_mask)
        nz_act_ratio = nz_cnt / act.numel()

        p_greedy = pi_t.probs.gather(-1, pi_t.mode.unsqueeze(-1)).mean()
        p_non_greedy = 1.0 - p_greedy

        safe_denom = torch.clamp(nz_cnt, min=1)
        nz_act = torch.sum(torch.where(nz_mask, act, 0.0)) / safe_denom
        nz_entropy = torch.sum(torch.where(nz_mask, h_t, 0.0)) / safe_denom

    # save stats for logging
    stats_keys = [
        'L_q', 'Q', 'Q_tar', 'r',
        'act', 'act+%', 'act+',
        'H', 'H+', 'H_t', 'P(explore)',
        'L_t', 'temp',
    ]
    stats_vals = [
        loss_q, q_a_t, target_q_t, r_t,
        act, nz_act_ratio, nz_act,
        h_t, nz_entropy, entropy_tar, p_non_greedy,
        loss_temp, temp_t,
    ]
    return dict(zip(stats_keys, stats_vals))


def sample_episode(env, agent, run_data):
    rb = run_data.replay_buffer
    logger = run_data.logger

    obs, _ = env.reset()
    rb.put(obs=obs)
    done = False

    while not done:
        a = agent.act(obs)
        obs_tn, reward, term_tn, trunc_tn, info = env.step(a)
        rb.put(obs=obs_tn, act=a, rew=reward)

        done = trunc_tn.flatten()[0] or term_tn.flatten()[0]
        obs = obs_tn

    n_exp_steps, n_rb_steps = rb.commit_episode()
    logger.accumulate(info['episode_extra_stats'], prefix='train')

    run_data.ep += 1
    run_data.step += n_exp_steps
    run_data.n_last_collected_steps += n_rb_steps


def train_iter(agent, run_data):
    stats_hist = agent.update_many(
        rb=run_data.replay_buffer,
        n_new_data=run_data.n_last_collected_steps,
    )
    if len(stats_hist) > 0:
        run_data.n_last_collected_steps = 0

    for stats in stats_hist:
        run_data.logger.accumulate(stats, prefix="agent")

    n_total_updates = run_data.replay_buffer.n_sampled
    run_data.logger.set_summary({'update': n_total_updates}, prefix="agent")


def train_epoch(env, agent, run_data, config):
    logger = run_data.logger
    n_episodes = config.run.n_episodes
    emb_schedule = config.embedding.schedule

    for i in range(n_episodes):
        sample_episode(env, agent, run_data)
        train_iter(agent, run_data)
        if emb_schedule is not None:
            if run_data.step // emb_schedule > run_data.last_emb_save_step // emb_schedule:
                _maybe_save_embedder(agent, run_data.step, config.log.name)
                run_data.last_emb_save_step = run_data.step
        
        logger.log(run_data.step, flush=True)

    logger.log(run_data.step, flush=True, force=True, prefix='train')


def eval_epoch(env, agent, run_data, config, prefix='eval'):
    logger = run_data.logger
    eval_config = config.eval
    uenv = env.unwrapped
    temp_rescale = 1/4 if config.eval.soft_greedy else 0.0

    stats = defaultdict(list)
    cluster_step_data = defaultdict(list) 
    
    ids_np = np.array(eval_config.pairs)
    ids = to_torch(ids_np, device=agent.device, copy=True)

    # reset state and set eval time limits
    train_time_limits = uenv.set_time_limits(save_state=True, full_history=True)
    obs, _ = env.reset()
    done = False

    while not done:
        a, pi_info = agent.act(obs, temp_rescale=temp_rescale, with_info=True)
        obs_tn, reward, term_tn, trunc_tn, info = env.step(a)

        torch_stats = pi_info
        np_stats = {
            'demand': uenv.ep_state.last_demand,
            'inv': uenv.ep_state.inventory,
            'inv_tr': uenv.ep_state.inventory_in_transit_sum,
        }
        step_stats = {k: v.flatten()[ids].numpy(force=True) for k, v in torch_stats.items()}
        step_stats |= {k: v.flatten()[ids_np] for k, v in np_stats.items()}
        
        # == cluster stats
        if eval_config.item_clusters is not None:
            all_np_stats = {
                'demand': uenv.ep_state.last_demand.flatten(),   # (S*I,)
                'inv': uenv.ep_state.inventory.flatten(),
                'inv_tr': uenv.ep_state.inventory_in_transit_sum.flatten(),
                'a': to_numpy(a).flatten(),
            }
            cluster_step_data['_all'].append(all_np_stats)
        # ==
        
        for k, v in step_stats.items():
            stats[k].append(v)

        done = trunc_tn.flatten()[0] or term_tn.flatten()[0]
        obs = obs_tn

    logger.accumulate(info['episode_extra_stats'], prefix=prefix)

    # 2d arrays (T, ids) -> (ids, T) -> split dicts by id {id: {...}}
    stats = {k: np.array(v).T for k, v in stats.items()}
    stats = {
        str(id): {k: v[i] for k, v in stats.items()}
        for i, id in enumerate(ids_np)
    }

    import matplotlib.pyplot as plt

    def _plt(ys, fig_num, is_bar_plot):
        assert ys.ndim == 1
        fig = plt.figure(num=fig_num+1, clear=True)
        ax = fig.subplots()
        if is_bar_plot:
            ax.bar(np.arange(ys.size), ys)
        else:
            ax.plot(ys)
        return fig

    bar_plot = {'a', 'demand'}
    stats = {
        id: {
            k: _plt(v, fig_num=j*len(id_stats) + i, is_bar_plot=k in bar_plot)
            for i, (k, v) in enumerate(id_stats.items())
        }
        for j, (id, id_stats) in enumerate(stats.items())
    }

    logger.log(run_data.step, metrics=stats, prefix=prefix, flush=True, force=True)

    # reset state again
    obs, _ = env.reset()
    env.unwrapped.set_time_limits(save_state=False, **train_time_limits)

def run_experiment(config, env, env_test, agent, nz_mask):
    rb_rng = torch.Generator().manual_seed(config.seed)
    n_epochs = config.run.n_epochs

    logger = start_logger(config)

    run_data = SimpleNamespace(
        step=0, ep=0, update=0,
        replay_buffer=None,
        logger=logger,
        n_last_collected_steps=0,
        last_emb_save_step=0,   # item_emb
    )
    run_data.replay_buffer = get_episodic_replay_buffer(
        max_size=config.run.rb_size, rng=rb_rng, nz_mask=nz_mask, device=agent.device
    )

    obs_t, _, = env.reset(seed=config.seed)

    eval_epoch(env=env, agent=agent, run_data=run_data, config=config, prefix='eval/train')
    eval_epoch(env=env_test, agent=agent, run_data=run_data, config=config, prefix='eval/test')

    for epoch in range(1, n_epochs + 1):
        train_epoch(env=env, agent=agent, run_data=run_data, config=config)
        eval_epoch(env=env, agent=agent, run_data=run_data, config=config, prefix='eval/train')
        eval_epoch(env=env_test, agent=agent, run_data=run_data, config=config, prefix='eval/test')


def train(config):
    rng = np.random.default_rng(config['seed'])
    device = get_device(config.get('device', None))
    dtype = get_dtype(config.get('dtype', None))
    print(f'Using device: {device}  |   dtype: {dtype}')

    env_id = config['env'] = config['env-ids'][config['env-id']]
    if 'overrides' in config:
        override_config(config, config['overrides'].get(env_id, {}))

    if device.type == 'cuda':
        kk = 2
        config['agents']['base']['batch_size'] *= kk
        config['agents']['base']['utd'] *= kk
    n_envs, env_bsz = config['n_envs'], config['batch_size']

    env = envpool.make_gymnasium(
        env_id, num_envs=n_envs, seed=get_seed(rng),
        batch_size=env_bsz, num_threads=config.get('num_threads', env_bsz),
    )
    # start early
    env.async_reset()
    obs_size, act_size = np.prod(env.single_observation_space.shape), np.prod(env.single_action_space.shape)
    print(f'ENV: {env_id} | Obs: {obs_size}  Act: {act_size}')
    obs_space, act_space = env.observation_space, env.action_space
    print(f'Obs space: {obs_space}  | Act space: {act_space}')
    reward_scale = config['reward_scale']

    agent_cfg = config['agents']['base'] | config['agents'][config['agent']]
    model = config['model']
    model['obs_size'] = obs_size
    model['act_size'] = act_size
    
    model['actor']['act_bias'] = (act_space.high + act_space.low) / 2.0
    model['actor']['act_scale'] = (act_space.high - act_space.low) / 2.0
    agent = Agent(model=model, **agent_cfg, device=device, seed=get_seed(rng), dtype=dtype)
    agent.replay_buffer.init_state(n_envs=n_envs, env_bsz=env_bsz)

    config['log']['name'] = isnone(config['log'].get('name', None), agent.short_name())
    logger=start_logger(config, tracker=config['trackers'])

    rl_perf_tracker = EpStatsTracker(n_envs=n_envs)
    n_steps = config['n_steps']
    # update each "full cycle", i.e. when all envs roughly made a single step 
    # for better thread pool utilization
    update_scheduler = Scheduler(n_envs)

    # eval_epoch(env=env, agent=agent, run_data=run_data, config=config, prefix='eval/train')
    # eval_epoch(env=env_test, agent=agent, run_data=run_data, config=config, prefix='eval/test')

    step = 0
    while step < n_steps:
        obs, reward, term, trunc, info = env.recv()
        ids = info['env_id']
        step += env_bsz

        flags = to_flags_np(term, trunc)
        m_done = np.logical_or(term, trunc)

        _obs = agent.to_torch(obs)
        _reward = agent.to_torch(reward * reward_scale).view(-1, 1)
        _flags = agent.to_torch(flags, keep_type=True).view(-1, 1)

        _a = agent.act(_obs)
        a_env = agent.model.pi.to_env_action(_a)
        env.send(a_env, ids)

        _m_reset = agent.replay_buffer.put(ixs=ids, rew=_reward, flags=_flags, obs_next=_obs, act_next=_a)
        rl_perf_tracker.step(ids, rew=reward, done=m_done, reset=to_numpy(_m_reset, copy=False))
        if rl_perf_tracker.is_ready:
            # logger.set_summary(rl_perf_tracker.flush(), prefix='train', key='rl')
            for rl_stats in rl_perf_tracker.flush(raw=True):
                logger.accumulate(rl_stats, prefix='train', key='rl')

        if update_scheduler.tick(env_bsz):
            stats_hist = agent.update_many(n_envs)
            for stats in stats_hist:
                logger.accumulate(stats, prefix="agent", key="slow")
            logger.set_summary(
                {
                    'n_updates': agent.replay_buffer.n_sampled,
                    'n_collected': agent.replay_buffer.n_total_added,
                    'n_ep': rl_perf_tracker.n_ep
                },
                prefix="agent", key="slow"
            )

        logger.log(step, flush=True)

    logger.log(step, flush=True, force=True)
    env.close()
    logger.finish()
