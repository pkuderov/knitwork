import contextlib
import copy
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv
import numpy as np
import torch
from torch import nn

from knitwork.common.config import load_config
from knitwork.env.mikasa_observations import PreviousActionReward
from knitwork.exps.mikasa.run import evaluate_policy, main, make_env
from knitwork.models.utils import build_model
from knitwork.rl.on_policy import (
    EpisodeStats, RolloutBatch, compute_gae, prep_obs, refresh_rollout_state,
    sample_batch, train_batch, validate_recurrent_policy,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'knitwork/exps/mikasa/config/lru_mid'


class CounterEnv(gym.Env):
    observation_space = spaces.Box(0, 100, shape=(1,), dtype=np.float32)
    action_space = spaces.Discrete(2)

    def __init__(self, truncation=False):
        self.truncation = truncation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_ix = 0
        return np.array([1], dtype=np.float32), {}

    def step(self, action):
        self.step_ix += 1
        ended = self.step_ix == 2
        return np.array([self.step_ix + 1], dtype=np.float32), float(self.step_ix), ended and not self.truncation, ended and self.truncation, {}


class SumPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, obs, state, **_):
        state = state + self.scale * obs[:, 0]
        return torch.stack([state, -state], -1), state, state, {}

    def reset_state(self, state, reset):
        return state * ~reset

    def detach_state(self, state):
        return state.detach()


class MikasaPpoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gae_bootstraps_truncation_and_cuts_episode_and_reset_slots(self):
        rewards = torch.tensor([[1.], [2.], [0.], [3.]])
        values = torch.tensor([[.5], [.6], [4.], [.8], [99.]])
        term = torch.tensor([[False], [False], [False], [True]])
        trunc = torch.tensor([[False], [True], [False], [False]])
        valid = torch.tensor([[True], [True], [False], [True]])
        advs, returns = compute_gae(rewards, values, term, trunc, valid, gamma=.9, lambda_=1.)
        torch.testing.assert_close(advs, torch.tensor([[5.54], [5.], [0.], [2.2]]))
        torch.testing.assert_close(returns[0], torch.tensor([6.04]))
        _, terminal_returns = compute_gae(rewards, values, term | trunc, torch.zeros_like(trunc), valid, gamma=.9, lambda_=1.)
        torch.testing.assert_close(terminal_returns[0], torch.tensor([2.8]))

    def test_next_step_sampling_reset_order_bootstrap_replay_and_episode_stats(self):
        env = SyncVectorEnv([lambda: CounterEnv(truncation=True)], autoreset_mode=AutoresetMode.NEXT_STEP)
        try:
            obs, _ = env.reset(seed=1)
            model = SumPolicy()
            batch, state, obs, done = sample_batch(env, model, torch.zeros(1), torch.from_numpy(obs), torch.zeros(1,dtype=torch.bool), obs_space=env.single_observation_space,rollout_len=4,dtype=torch.float32,is_discrete=False)
            torch.testing.assert_close(batch.values[:, 0], torch.tensor([1., 3., 6., 1., 3.]))
            self.assertEqual(batch.reset[:, 0].tolist(), [False, False, True, False])
            stats = EpisodeStats(1)
            stats.update(batch)
            self.assertEqual(stats.get(), {'EpRet':3., 'EpLen':2.})
            replay = refresh_rollout_state(model, batch)
            torch.testing.assert_close(replay,state)
            model.scale.data.fill_(2.)
            torch.testing.assert_close(refresh_rollout_state(model, batch), 2 * state)
        finally:
            env.close()
        env = SyncVectorEnv([CounterEnv],autoreset_mode=AutoresetMode.SAME_STEP)
        try:
            with self.assertRaisesRegex(ValueError,'NEXT_STEP'):
                sample_batch(env,SumPolicy(),torch.zeros(1),torch.zeros(1,1),torch.zeros(1,dtype=torch.bool),obs_space=env.single_observation_space,rollout_len=1,dtype=torch.float32,is_discrete=False)
        finally:
            env.close()

    def test_previous_action_reward_is_observed_and_cleared_on_reset(self):
        env = PreviousActionReward(CounterEnv())
        obs, _ = env.reset(seed=1)
        np.testing.assert_array_equal(obs, [1,0,0,0])
        obs, reward, _, _, _ = env.step(1)
        np.testing.assert_array_equal(obs, [2,0,1,reward])
        obs, _ = env.reset()
        np.testing.assert_array_equal(obs, [1,0,0,0])
        env.close()

    def test_battleship_flat_actions_and_tuple_observations(self):
        env = make_env('popgym-BattleshipEasy-v0', {'discrete_actions':True, 'previous_action_reward':True})
        try:
            self.assertEqual(env.action_space.n,64)
            obs, _ = env.reset(seed=2)
            self.assertEqual(obs.shape,(67,))
            obs, *_ = env.step(63)
            self.assertEqual(obs[2:66].argmax(),63)
        finally:
            env.close()
        env = SyncVectorEnv([lambda:make_env('popgym-AutoencodeEasy-v0', {}) for _ in range(3)],autoreset_mode=AutoresetMode.NEXT_STEP)
        try:
            obs, _ = env.reset(seed=3)
            prepared = prep_obs(obs,env.single_observation_space,'cpu',False,torch.float32)
            self.assertEqual(prepared.shape,(3,2))
        finally:
            env.close()

    def test_all_models_support_token_and_vector_rl_and_ppo_replay(self):
        for path in CONFIG.glob('*.yaml'):
            cfg = load_config(path)
            options = copy.deepcopy(cfg[cfg['model'].replace('.','_')])
            if 'hidden_size' in options:
                options['hidden_size'] = 12
            if 'module_size' in options:
                options['module_size'] = 4
            options['n_layers'] = 2
            for wrapper, size in [('rl_token',4),('rl_vector',9)]:
                with self.subTest(model=cfg['model'], wrapper=wrapper), contextlib.redirect_stdout(io.StringIO()):
                    model = build_model(wrapper, dict(input_size=size,output_size=4,dtype=torch.float32,device='cpu',feature_norm=True,policy_gain=.01),cfg['model'].split('.')[0],options)
                    state = model.rnn.reset_state(None,bsz=3)
                    obs = torch.tensor([[0],[1],[3]]) if wrapper=='rl_token' else torch.randn(3,size)
                    logits,value,state,_ = model(obs,state)
                    self.assertEqual(logits.shape,(3,4))
                    self.assertEqual(value.shape,(3,))
                    (logits.square().mean()+value.square().mean()).backward()
                    self.assertTrue(all(p.grad is None or p.grad.isfinite().all() for p in model.parameters()))
            with contextlib.redirect_stdout(io.StringIO()):
                model = build_model('rl_vector',dict(input_size=9,output_size=4,dtype=torch.float32,device='cpu',feature_norm=True,policy_gain=.01),cfg['model'].split('.')[0],options)
                env = SyncVectorEnv([lambda:make_env('popgym-RepeatFirstEasy-v0',{'previous_action_reward':True}) for _ in range(3)],autoreset_mode=AutoresetMode.NEXT_STEP)
                try:
                    raw, _ = env.reset(seed=4)
                    obs = prep_obs(raw,env.single_observation_space,'cpu',False,torch.float32)
                    batch, *_ = sample_batch(env,model,model.rnn.reset_state(None,bsz=3),obs,torch.zeros(3,dtype=torch.bool),obs_space=env.single_observation_space,rollout_len=54,dtype=torch.float32,is_discrete=False)
                    metrics = train_batch(model,batch,torch.optim.SGD(model.parameters(),lr=0),ppo_epochs=2,clip_eps=.1,value_coef=.5,entropy_coef=.005,max_grad_norm=.5,gamma=.995,gae_lambda=.95,target_kl=.02)
                    self.assertEqual(metrics['Upd'],2)
                    self.assertEqual(metrics['Skipped'],0)
                    self.assertLess(abs(metrics['ApproxKL']),1e-6)
                    self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
                finally:
                    env.close()

    def test_stochastic_recurrence_is_rejected_and_empty_batch_is_safe(self):
        with self.assertRaisesRegex(ValueError,'dropout'):
            validate_recurrent_policy(nn.Dropout(.1))
        model = SumPolicy()
        batch = RolloutBatch(obs=torch.zeros(1,1,1),actions=torch.zeros(1,1,dtype=torch.long),log_probs=torch.zeros(1,1),rewards=torch.zeros(1,1),values=torch.zeros(2,1),term=torch.zeros(1,1,dtype=torch.bool),trunc=torch.zeros(1,1,dtype=torch.bool),reset=torch.ones(1,1,dtype=torch.bool),state_init=torch.zeros(1),prev_batch_done=torch.zeros(1,dtype=torch.bool))
        metrics = train_batch(model,batch,torch.optim.SGD(model.parameters(),lr=.1),ppo_epochs=4,clip_eps=.1,value_coef=.5,entropy_coef=.005,max_grad_norm=.5,gamma=.995,gae_lambda=.95)
        self.assertEqual(metrics['Upd'],0)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))

    def test_kl_stop_prevents_an_extra_update(self):
        env = SyncVectorEnv([CounterEnv],autoreset_mode=AutoresetMode.NEXT_STEP)
        try:
            raw, _ = env.reset(seed=3)
            model = SumPolicy()
            batch, *_ = sample_batch(env,model,torch.zeros(1),torch.from_numpy(raw),torch.zeros(1,dtype=torch.bool),obs_space=env.single_observation_space,rollout_len=3,dtype=torch.float32,is_discrete=False)
            batch.log_probs -= 2
            metrics = train_batch(model,batch,torch.optim.SGD(model.parameters(),lr=.1),ppo_epochs=4,clip_eps=.1,value_coef=.5,entropy_coef=.005,max_grad_norm=.5,gamma=.995,gae_lambda=.95,target_kl=.02)
            self.assertEqual(metrics['Upd'],0)
            self.assertEqual(metrics['KLStop'],1)
            self.assertEqual(model.scale.item(),1.)
        finally:
            env.close()

    def test_value_loss_weights_individual_valid_transitions(self):
        model = SumPolicy()
        obs = torch.ones(2,2,1)
        logits = torch.tensor([[[1.,-1.],[1.,-1.]],[[2.,-2.],[2.,-2.]]])
        actions = torch.zeros(2,2,dtype=torch.long)
        batch = RolloutBatch(
            obs=obs,actions=actions,log_probs=torch.distributions.Categorical(logits=logits).log_prob(actions),
            rewards=torch.zeros(2,2),values=torch.zeros(3,2),term=torch.zeros(2,2,dtype=torch.bool),
            trunc=torch.zeros(2,2,dtype=torch.bool),reset=torch.tensor([[False,False],[True,False]]),
            state_init=torch.zeros(2),prev_batch_done=torch.zeros(2,dtype=torch.bool),
        )
        metrics = train_batch(model,batch,torch.optim.SGD(model.parameters(),lr=0),ppo_epochs=1,clip_eps=.1,value_coef=.5,entropy_coef=.005,max_grad_norm=.5,gamma=.995,gae_lambda=.95)
        self.assertAlmostEqual(metrics['L_v'],2.)
        self.assertAlmostEqual(metrics['ValidFraction'],.75)

    def test_fixed_seed_evaluation_preserves_rng_mode_and_is_repeatable(self):
        cfg = load_config(CONFIG/'grnn_lru.yaml')
        options = cfg['grnn_lru_L2C4'] | {'hidden_size':12}
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model('rl_vector',dict(input_size=9,output_size=4,dtype=torch.float32,device='cpu'), 'grnn_lru', options)
        before = torch.get_rng_state().clone()
        kw = dict(n_envs=2,episodes=4,seed=50000,max_vector_steps=120)
        first = evaluate_policy(model,'popgym-RepeatFirstEasy-v0',{'previous_action_reward':True},**kw)
        torch.testing.assert_close(before,torch.get_rng_state())
        self.assertTrue(model.training)
        self.assertEqual(first,evaluate_policy(model,'popgym-RepeatFirstEasy-v0',{'previous_action_reward':True},**kw))

    def test_configs_budget_and_runner_seed_repeatability(self):
        for path in CONFIG.rglob('*.yaml'):
            cfg = load_config(path)
            self.assertEqual(cfg['n_envs'] * cfg['rollout_len'],4096)
            self.assertEqual(cfg['n_steps'] % 4096,0)
            self.assertEqual(cfg['communication'],{'loss_weight':0.,'entropy_weight':0.})
        cfg = load_config(CONFIG/'grnn_lru.yaml')
        cfg.update(n_envs=2,rollout_len=4,n_steps=16,compile=False,diagnostics_schedule=None)
        cfg['grnn_lru_L2C4']['hidden_size']=12
        cfg['log']['logger']=None
        cfg['eval']['enabled']=False
        original = copy.deepcopy(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            main(cfg)
            first = torch.get_rng_state().clone()
            main(cfg)
        torch.testing.assert_close(first,torch.get_rng_state())
        self.assertEqual(cfg,original)


if __name__ == '__main__':
    unittest.main()
