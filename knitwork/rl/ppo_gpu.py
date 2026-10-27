"""Recurrent PPO for batched on-device environments (SAME_STEP autoreset, no truncation)."""
from __future__ import annotations

import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils._pytree import tree_flatten, tree_unflatten


def map_state(state, fn):
    if isinstance(state, torch.Tensor):
        return fn(state)
    if isinstance(state, dict):
        return {k: map_state(v, fn) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(map_state(v, fn) for v in state)
    return state


def index_state(state, idx, n_envs):
    """Select envs idx from a state whose tensors have exactly one n_envs-sized dim."""
    def select(x):
        dims = [d for d, size in enumerate(x.shape) if size == n_envs]
        assert len(dims) == 1, f'ambiguous batch dim in state of shape {tuple(x.shape)}'
        return x.index_select(dims[0], idx)
    return map_state(state, select)


@torch.no_grad()
def collect(env, model, state, obs, rollout_len):
    """One rollout; returns tensors [T, B, ...], the next obs/state and bootstrap value."""
    state_init = state
    buf = {k: [] for k in ('obs', 'actions', 'log_probs', 'values', 'rewards', 'dones', 'targets')}
    for _ in range(rollout_len):
        logits, value, state, _ = model(obs, state)
        actions = Categorical(logits=logits).sample()
        buf['targets'].append(env.target())
        next_obs, reward, done = env.step(actions)
        buf['obs'].append(obs)
        buf['actions'].append(actions)
        buf['log_probs'].append(Categorical(logits=logits).log_prob(actions))
        buf['values'].append(value)
        buf['rewards'].append(reward)
        buf['dones'].append(done)
        state = model.reset_state(state, done)
        obs = next_obs
    _, last_value, _, _ = model(obs, state)
    batch = {k: torch.stack(v) for k, v in buf.items()}
    batch['state_init'] = state_init
    batch['last_value'] = last_value
    return batch, model.detach_state(state), obs


@torch.no_grad()
def compute_gae(rewards, values, dones, last_value, *, gamma, lambda_):
    advs = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(rewards.shape[0])):
        nonterminal = (~dones[t]).to(rewards.dtype)
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lambda_ * nonterminal * gae
        advs[t] = gae
        next_value = values[t]
    return advs, advs + values


def replay(model, obs, dones, state):
    """Re-run the recurrent policy over a stored rollout: logits [T, b, A], values [T, b],
    aux predictions (or head features) [T, b, F] and normalised routing entropy [T] (zeros without dynamic routing)."""
    logits, values, auxs, ents = [], [], [], []
    for t in range(obs.shape[0]):
        lg, v, state, info = model(obs[t], state)
        state = model.reset_state(state, dones[t])
        logits.append(lg)
        values.append(v)
        auxs.append(info['aux'])
        ent = info.get('comm_entropy')
        ents.append(torch.stack(ent).mean() if ent else v.new_zeros(()))
    return torch.stack(logits), torch.stack(values), torch.stack(auxs), torch.stack(ents)


def aux_memory_loss(aux, lags, obs, dones):
    """MSE of the prediction of obs[t - k] made at step t, for each lag k, within the same episode."""
    T, b, _ = obs.shape
    pred = aux.view(T, b, len(lags), -1)  # [T, b, K, obs_dim]
    # episode id per step: increments after each done
    ep = torch.cumsum(torch.cat([dones.new_zeros(1, b), dones[:-1]]).long(), 0)  # [T, b]
    losses = []
    for i, k in enumerate(lags):
        same = (ep[k:] == ep[:-k]).unsqueeze(-1).to(obs.dtype)  # [T-k, b, 1]
        err = (pred[k:, :, i] - obs[:-k]) ** 2 * same
        losses.append(err.sum() / (same.sum() * obs.shape[-1]).clamp_min(1))
    return torch.stack(losses).mean()


def ppo_update(
        model, batch, optim, *, epochs, minibatches, clip_eps, value_coef, entropy_coef,
        max_grad_norm, gamma, gae_lambda, target_kl=None, supervised=False, reward_mult=1.0,
        aux_coef=0.0, route_entropy_coef=0.0,
):
    rewards = batch['rewards'] * reward_mult
    advs, returns = compute_gae(
        rewards, batch['values'], batch['dones'], batch['last_value'], gamma=gamma, lambda_=gae_lambda,
    )
    T, B = rewards.shape
    stats = {k: [] for k in ('L_pi', 'L_v', 'H', 'ApproxKL', 'ClipFrac', '|Grad|', 'L_sup', 'L_aux', 'H_route')}
    net = model.model if isinstance(model, GraphedPolicy) else model
    stopped = False
    for _ in range(epochs):
        perm = torch.randperm(B, device=rewards.device)
        for idx in perm.chunk(minibatches):
            state = index_state(batch['state_init'], idx, B)
            obs, dones = batch['obs'][:, idx], batch['dones'][:, idx]
            if isinstance(model, GraphedPolicy):
                logits, values, aux, route_ent = model.replay_minibatch(obs, dones, state)
            else:
                logits, values, aux, route_ent = replay(model, obs, dones, state)
            dist = Categorical(logits=logits)
            actions = batch['actions'][:, idx]
            log_ratio = dist.log_prob(actions) - batch['log_probs'][:, idx]
            ratio = log_ratio.exp()
            adv = advs[:, idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv).mean()
            value_loss = F.mse_loss(values, returns[:, idx])
            entropy = dist.entropy().mean()
            if supervised:
                # oracle diagnostic: same data stream and model, cross-entropy on rewarded targets
                targets = batch['targets'][:, idx]
                # CountRecall counts can exceed the action range; such targets are unreachable
                rewarded = (targets >= 0) & (targets < logits.shape[-1])
                sup_loss = F.cross_entropy(logits[rewarded], targets[rewarded])
                loss = sup_loss + value_coef * value_loss
                stats['L_sup'].append(sup_loss.detach())
            else:
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            if aux_coef:
                aux_loss = aux_memory_loss(aux, net.aux_lags, obs, dones)
                loss = loss + aux_coef * aux_loss
                stats['L_aux'].append(aux_loss.detach())
            if route_entropy_coef:
                # > 0 rewards softer routing, < 0 sharper
                loss = loss - route_entropy_coef * route_ent.mean()
            stats['H_route'].append(route_ent.mean().detach())

            optim.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(optim.param_groups[0]['params'], max_grad_norm)
            optim.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
                stats['L_pi'].append(policy_loss.detach())
                stats['L_v'].append(value_loss.detach())
                stats['H'].append(entropy.detach())
                stats['ApproxKL'].append(approx_kl)
                stats['ClipFrac'].append(((ratio - 1).abs() > clip_eps).float().mean())
                stats['|Grad|'].append(grad_norm)
        if target_kl is not None and not supervised and stats['ApproxKL'][-1] > target_kl:
            stopped = True
            break
    out = {k: torch.stack(v).mean() for k, v in stats.items() if v}
    out['KLStop'] = float(stopped)
    out['ExplVar'] = 1 - (returns - batch['values']).var() / returns.var().clamp_min(1e-8)
    return out


class EpisodeTracker:
    """Returns of episodes that started after the (desynchronized) initial reset."""

    def __init__(self, n_envs, device):
        self.ret = torch.zeros(n_envs, device=device)
        self.fresh = torch.zeros(n_envs, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, rewards, dones):
        finished = []
        for t in range(rewards.shape[0]):
            self.ret += rewards[t]
            ended = dones[t] & self.fresh
            finished.append(self.ret[ended])
            self.ret = self.ret.masked_fill(dones[t], 0.0)
            self.fresh |= dones[t]
        return torch.cat(finished)


@torch.no_grad()
def evaluate(env, model, *, greedy):
    """Mean return of the first episode of every env (capped at env.ep_len steps)."""
    was_training = model.training
    model.eval()
    obs = env.reset(desync=False)
    state = model.rnn.reset_state(None, bsz=env.n_envs)
    total = torch.zeros(env.n_envs, device=obs.device)
    running = torch.ones(env.n_envs, dtype=torch.bool, device=obs.device)
    for _ in range(env.ep_len):
        logits, _, state, _ = model(obs, state)
        actions = logits.argmax(-1) if greedy else Categorical(logits=logits).sample()
        obs, reward, done = env.step(actions)
        total += reward * running
        running &= ~done
        state = model.reset_state(state, done)
    model.train(was_training)
    return total.mean().item()

class _Replay(torch.nn.Module):
    def __init__(self, model, spec):
        super().__init__()
        self.model, self.spec = model, spec

    def forward(self, obs, dones, *leaves):
        return replay(self.model, obs, dones, tree_unflatten(list(leaves), self.spec))


class GraphedPolicy:
    """CUDA-graph captured sampling step and minibatch replay (fwd + bwd); shapes are static."""

    def __init__(self, model, state, obs, *, rollout_len, minibatch):
        self.model = model
        leaves, self.spec = tree_flatten(state)
        n_envs = obs.shape[0]
        # sampling step: static inputs are overwritten in place before each replay
        self.obs = obs.clone()
        self.leaves = [x.clone() for x in leaves]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                model(self.obs, tree_unflatten(self.leaves, self.spec))
        torch.cuda.current_stream().wait_stream(stream)
        self.step_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.step_graph), torch.no_grad():
            logits, value, new_state, _ = model(self.obs, tree_unflatten(self.leaves, self.spec))
            self.out = (logits, value, tree_flatten(new_state)[0])

        idx = torch.arange(minibatch, device=obs.device)
        sample = (
            obs.new_zeros(rollout_len, minibatch, obs.shape[1]),
            torch.zeros(rollout_len, minibatch, dtype=torch.bool, device=obs.device),
            *tree_flatten(index_state(state, idx, n_envs))[0],
        )
        self.replay = torch.cuda.make_graphed_callables(_Replay(model, self.spec), sample)
        model.zero_grad(set_to_none=True)

    def __call__(self, obs, state, **_):
        self.obs.copy_(obs)
        for dst, src in zip(self.leaves, tree_flatten(state)[0]):
            dst.copy_(src)
        self.step_graph.replay()
        logits, value, leaves = self.out
        state = tree_unflatten([x.clone() for x in leaves], self.spec)
        return logits.clone(), value.clone(), state, {}

    def reset_state(self, state, mask):
        return self.model.reset_state(state, mask)

    def detach_state(self, state):
        return self.model.detach_state(state)

    def replay_minibatch(self, obs, dones, state):
        return self.replay(obs, dones, *tree_flatten(state)[0])
