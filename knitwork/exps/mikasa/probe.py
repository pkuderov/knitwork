"""Supervised memory probe on the env's own observations: predict obs[t - delay] from obs[:t+1]; no RL loss.

uv run python -m knitwork.exps.mikasa.probe <rl config> [--model=rnn.L2] [--delay=64] [--updates=3000]
"""
import argparse
import json

import gymnasium as gym
import numpy as np
import popgym  # noqa: F401
import torch
import yaml
from torch.nn import functional as F

from knitwork.models.utils import build_model


def collect(env_id, episodes, seed):
    env, rng, out = gym.make(env_id), np.random.default_rng(seed), []
    for i in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        seq, done = [int(obs)], False
        while not done:
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
            seq.append(int(obs))
        out.append(seq)
    length = min(map(len, out))
    return torch.tensor([s[:length] for s in out])  # [N, T]


def objective(model, tokens, delay):
    state = model.init_state(tokens.shape[0])
    emb = model.embedding(tokens)  # [B, T, H]
    feats = []
    for t in range(tokens.shape[1]):
        out, state, _ = model.rnn(emb[:, t][None], state)
        feats.append(out)
    logits = model.head(torch.stack(feats, 1)[:, delay:])  # [B, T-delay, V]
    target = tokens[:, :-delay]
    return F.cross_entropy(logits.flatten(0, 1), target.flatten()), (logits.argmax(-1) == target).float().mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('--model'); ap.add_argument('--delay', type=int, default=64)
    ap.add_argument('--updates', type=int, default=3000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); name = a.model or cfg['model']
    torch.manual_seed(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    train, val = collect(cfg['env'], 4000, a.seed), collect(cfg['env'], 500, a.seed + 1)
    model = build_model(wrapper_type='token', rnn_type=name.split('.', 1)[0], rnn_cfg=cfg[name.replace('.', '_')],
                        wrapper_cfg=dict(input_size=4, output_size=4, dtype=torch.float32, device=device)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(a.seed)
    for u in range(1, a.updates + 1):
        for g in opt.param_groups:
            g['lr'] = a.lr * min(1.0, u / 200)
        loss, acc = objective(model, train[torch.as_tensor(rng.choice(len(train), 64))].to(device), a.delay)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if u % 500 == 0 or u == a.updates:
            with torch.no_grad():
                vl, va = objective(model, val.to(device), a.delay)
            print(json.dumps(dict(model=name, delay=a.delay, update=u, train_acc=round(float(acc), 3), val_acc=round(float(va), 3), val_loss=round(float(vl), 3), chance=0.25)), flush=True)


if __name__ == '__main__':
    main()
