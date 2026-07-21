"""Column-contribution ablation for a trained grnn_fix_v4 checkpoint on SDQ.

Loads a checkpoint, runs SDQ batches in inference, and quantifies how much each
column (and column subgroups) contributes to the SDQ prediction:

  A. interaction/activation ranking  -- attention to/from column 0, activation norms
  B. attention masking               -- let only a subgroup interact via attention,
                                        then read out SDQ acc with a linear head
  C. per-column readout              -- a linear head on each single column / subgroup

All metrics + heatmaps are logged to Comet (--log.logger=comet).

Usage:
  uv run inference/analyze_columns_sdq.py \
      --checkpoint runs/checkpoints/<run>/step_40000000.pt \
      --device cuda --n_collect 400 --subgroup_k 3 \
      --log.logger=comet --name "v4 col-ablation 40M"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from knitwork.common.entrypoint import parse_str
from knitwork.common.logging import create_logger
from knitwork.common.utils import CE_ignore_index, to_torch
from knitwork.gens.sdq import StoreDistractQueryGenerator
from knitwork.exps.sdq.run_sdq import build_model
from knitwork.visualization.attn_flow import AttnFlowVisualizer
from knitwork.exps.sdq._viz import log_figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── collection ────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect(rnn, gen, device, n_collect: int):
    """Run n_collect SDQ steps; return per-column top-layer states, targets, and
    time-averaged per-layer attention matrices [L, C, C]."""
    rnn.eval()
    state = None
    feats, tgts = [], []
    L, C = rnn.n_layers, rnn.n_columns
    attn_acc = [np.zeros((C, C)) for _ in range(L)]
    attn_cnt = 0
    for _ in range(n_collect):
        obs = gen.next()
        obs = {k: to_torch(v, device=device) for k, v in obs.items()}
        state = rnn.reset_state(state, obs['reset_mask'])
        x = obs['tokens'].view(-1, 1)
        out = rnn(x, state, return_attn=True)
        y, state = out[0], out[1]
        extras = out[2] if isinstance(out[2], dict) else {}
        h = state                                        # [L, C, B, H]
        feats.append(h[-1].permute(1, 0, 2).float().cpu())   # [B, C, H] top layer
        tgts.append(obs['targets'].cpu())
        aw = extras.get('attn_weights')
        if aw:
            for li, a in enumerate(aw):
                if a is not None:
                    m = a.detach().float().cpu().numpy()
                    while m.ndim > 2:
                        m = m.mean(0)
                    attn_acc[li] += m
            attn_cnt += 1
        state = rnn.detach_state(state)
    feats = torch.cat(feats, dim=0)                      # [N, C, H]
    tgts = torch.cat(tgts, dim=0)                        # [N]
    attn = np.stack([a / max(attn_cnt, 1) for a in attn_acc], axis=0)  # [L, C, C]
    return feats, tgts, attn


# ── linear readout head ───────────────────────────────────────────────────────

def train_readout(feats: torch.Tensor, tgts: torch.Tensor,
                  V: int, device, epochs: int = 300, split: float = 0.7):
    """Train a frozen-backbone linear head on `feats` [N, D]; return held-out SDQ
    accuracy (on active, non-ignore targets = the scored SDQ predictions)."""
    valid = tgts != CE_ignore_index
    feats, tgts = feats[valid], tgts[valid]
    N = feats.shape[0]
    if N < 32:
        return {'Acc': float('nan')}
    n_tr = int(N * split)
    perm = torch.randperm(N)
    tr, ev = perm[:n_tr], perm[n_tr:]
    Xtr, Ytr = feats[tr].to(device), tgts[tr].to(device)
    Xev, Yev = feats[ev].to(device), tgts[ev].to(device)

    head = nn.Linear(feats.shape[1], V).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=5e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(head(Xtr), Ytr).backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        acc = (head(Xev).argmax(dim=-1) == Yev).float().mean()
    return {'Acc': float(acc)}


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_bars(values: dict[str, float], title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(max(4, len(values) * 0.8), 3.2))
    keys = list(values.keys())
    ax.bar(range(len(keys)), [values[k] for k in keys], color='steelblue')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n_collect', type=int, default=400)
    ap.add_argument('--subgroup_k', type=int, default=3,
                    help='size of the subgroup selected by interaction with column 0')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--name', default=None, help='Comet run name')
    # logging overrides (dot-args passed through, e.g. --log.logger=comet)
    args, extra = ap.parse_known_args()
    overrides = {k: parse_str(v) for k, v in
                 (a.lstrip('-').split('=', 1) for a in extra if '=' in a)}

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    rnn_type = ckpt['model_type']
    print(f'Loaded {rnn_type} @ step {ckpt["step"]:,} from {args.checkpoint}')

    gen_cfg = config['gens'][config['gen']]
    gen = StoreDistractQueryGenerator(**gen_cfg, n_envs=config['n_envs'],
                                      seed=0, ignore_index=CE_ignore_index)
    rnn = build_model(rnn_type, config['models'][rnn_type], gen)
    rnn.load_state_dict(ckpt['model_state'])
    rnn = rnn.to(device)
    V, C = gen.V, rnn.n_columns

    # logger (Comet): reuse ckpt config, apply --log.* overrides, default enabled
    log_cfg = dict(config.get('log', {}))
    for k, v in overrides.items():
        if k.startswith('log.'):
            log_cfg[k[4:]] = v
    log_cfg.setdefault('enabled', True)
    log_cfg.setdefault('logger', 'comet')
    log_cfg.setdefault('project', 'grid-rnn-sdq')
    log_cfg['name'] = args.name or log_cfg.get('name') or f'{rnn_type}_col_ablation'
    logger = create_logger({'log': log_cfg} | {'model': rnn_type})
    step = int(ckpt['step'])

    # ── collect unmasked ────────────────────────────────────────────────────────
    feats, tgts, attn = collect(rnn, gen, device, args.n_collect)        # feats [N,C,H]
    print(f'Collected feats={tuple(feats.shape)} attn={attn.shape}')

    # A. interaction with column 0 + activation norms
    to_c0   = {f'C{c}': float(attn[:, c, 0].mean()) for c in range(1, C)}   # c attends to 0
    from_c0 = {f'C{c}': float(attn[:, 0, c].mean()) for c in range(1, C)}   # 0 attends to c
    interaction = {f'C{c}': to_c0[f'C{c}'] + from_c0[f'C{c}'] for c in range(1, C)}
    act_norm = {f'C{c}': float(feats[:, c, :].norm(dim=-1).mean()) for c in range(C)}
    print('interaction with C0:', {k: round(v, 3) for k, v in interaction.items()})
    print('activation norm     :', {k: round(v, 3) for k, v in act_norm.items()})

    av = AttnFlowVisualizer(n_layers=rnn.n_layers, n_columns=C)
    for li in range(rnn.n_layers):
        av._buffers[li].append(attn[li])
    av.log(logger, step=step)
    log_figure(logger, plot_bars(interaction, 'Interaction with C0 (attn to+from)', 'weight'),
               'ablation/interaction_c0', step)
    log_figure(logger, plot_bars(act_norm, 'Per-column activation norm', 'L2 norm'),
               'ablation/activation_norm', step)
    for k, v in interaction.items():
        logger.track(v, name=f'ablation/interaction_c0/{k}', step=step)
    for k, v in act_norm.items():
        logger.track(v, name=f'ablation/act_norm/{k}', step=step)

    # subgroup = column 0 + top-(k-1) by interaction with C0
    ranked = sorted(range(1, C), key=lambda c: interaction[f'C{c}'], reverse=True)
    subgroup = sorted([0] + ranked[:max(args.subgroup_k - 1, 0)])
    print(f'Selected subgroup S = {subgroup}')
    logger.track(float(len(subgroup)), name='ablation/subgroup_size', step=step)

    # C. per-single-column and subgroup linear readout (unmasked features)
    per_col_acc = {}
    for c in range(C):
        m = train_readout(feats[:, c, :], tgts, V, device, epochs=args.epochs)
        per_col_acc[f'C{c}'] = m['Acc']
        logger.track(m['Acc'], name=f'ablation/readout_acc/single/C{c}', step=step)
    all_feats = feats.reshape(feats.shape[0], -1)
    m_all = train_readout(all_feats, tgts, V, device, epochs=args.epochs)
    sub_feats = feats[:, subgroup, :].reshape(feats.shape[0], -1)
    m_sub = train_readout(sub_feats, tgts, V, device, epochs=args.epochs)
    print('readout Acc  all:', round(m_all['Acc'], 3), '| subgroup:', round(m_sub['Acc'], 3))
    logger.track(m_all['Acc'], name='ablation/readout_acc/all_columns', step=step)
    logger.track(m_sub['Acc'], name='ablation/readout_acc/subgroup', step=step)
    log_figure(logger, plot_bars({**per_col_acc, 'ALL': m_all['Acc'], 'SUB': m_sub['Acc']},
                                 'SDQ readout accuracy by column', 'Acc'),
               'ablation/readout_acc', step)

    # B. attention masking: only the subgroup interacts among itself; others isolated
    mask = torch.eye(C, dtype=torch.bool)
    for i in subgroup:
        for j in subgroup:
            mask[i, j] = True
    rnn.attn_col_mask = mask.to(device)
    feats_m, tgts_m, _ = collect(rnn, gen, device, args.n_collect)
    rnn.attn_col_mask = None
    sub_feats_m = feats_m[:, subgroup, :].reshape(feats_m.shape[0], -1)
    m_sub_masked = train_readout(sub_feats_m, tgts_m, V, device, epochs=args.epochs)
    print('readout Acc subgroup (attn-masked):', round(m_sub_masked['Acc'], 3))
    logger.track(m_sub_masked['Acc'], name='ablation/readout_acc/subgroup_masked', step=step)

    logger.end()
    print('Done. Logged to Comet.')


if __name__ == '__main__':
    main()
