"""Column-information & contribution analysis for a trained grnn_fix_v4 checkpoint on SDQ.

Loads a checkpoint, runs SDQ in inference, and quantifies (a) what each column stores and
(b) each column's / small-subgroup's real contribution to the SDQ prediction. Accuracy is
never collapsed to a single scalar: it is reported as a sliding curve.

Methods
  Sliding acc  -- model's own SDQ accuracy as a recall-vs-gap curve (binned by store->query
                  lag) + a rolling window over the stream + mean +/- 95% CI (Wilson).
  A. content   -- linear probes of each column for the causal constituents of the target
                  (stored value, distract accumulator, net count, queried key) -> a
                  [column x variable] probe-accuracy matrix + per-column selectivity.
  B1 logit     -- exact direct-logit attribution: decompose the correct-class logit into
                  per-column terms from the linear head (works for concat and mean heads).
  B2 causal    -- zero/mean ablate a column (via col_keep_mask) and re-measure the model's
                  OWN end-to-end accuracy drop (overall + gap curve).
  B3 subsets   -- exhaustive + greedy subset selection capped at K_max (<=4) columns:
                  best subset per size, minimal sufficient subgroup, and a fair per-column
                  value (mean marginal gain over all <=K_max coalitions).
  B4 redundancy-- pairwise linear CKA between columns (redundancy vs. specialization).

All metrics + heatmaps are logged (default Comet); everything is also printed to stdout.

Usage:
  uv run inference/analyze_columns_sdq.py \
      --checkpoint runs/checkpoints/<run>/step_60000000.pt \
      --device cuda --n_collect 800 --subset_max_k 4 \
      --log.logger=comet --name "v4 col-analysis 60M"
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import nn

from knitwork.common.entrypoint import _load_dotenv, parse_str
from knitwork.common.logging import create_logger
from knitwork.common.utils import CE_ignore_index, to_torch
from knitwork.gens.sdq import StoreDistractQueryGenerator
from knitwork.exps.sdq.run_sdq import build_model, sq_gap_metrics
from knitwork.visualization.attn_flow import AttnFlowVisualizer
from knitwork.visualization.cka import linear_cka
from knitwork.exps.sdq._viz import log_figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# collection

@torch.no_grad()
def collect(rnn, gen, device, n_collect: int):
    """Run n_collect SDQ steps; return a dict of per-query tensors aligned across all steps
    (features, head inputs, targets, sq_gaps, model correctness, causal latents) plus the
    time-averaged per-layer attention matrices [L, C, C]."""
    rnn.eval()
    state = None
    L, C = rnn.n_layers, rnn.n_columns
    feats, ocols, tgts, gaps, corr = [], [], [], [], []
    distract, stored_qk, net_qk, qkey = [], [], [], []
    attn_acc = [np.zeros((C, C)) for _ in range(L)]
    attn_cnt = 0
    for _ in range(n_collect):
        obs = gen.next()
        t = {k: to_torch(v, device=device) for k, v in obs.items() if k != 'sq_gaps'}
        state = rnn.reset_state(state, t['reset_mask'])
        out = rnn(t['tokens'].view(-1, 1), state, return_attn=True)
        y, state = out[0], out[1]
        extras = out[2] if isinstance(out[2], dict) else {}
        h = state                                             # [L, C, B, H]

        targets_np = obs['targets']
        vi = np.flatnonzero(targets_np != gen.ignore_index)   # query env indices (sorted)
        if vi.size:
            feats.append(h[-1].permute(1, 0, 2)[vi].float().cpu())        # [n, C, H] top state
            oc = extras.get('o_cols')
            if oc is not None:
                ocols.append(oc.permute(1, 0, 2)[vi].float().cpu())       # [n, C, H] head input
            yv = y[vi].argmax(dim=-1).cpu().numpy()
            tv = targets_np[vi].astype(np.int64)
            corr.append((yv == tv).astype(np.float32))
            tgts.append(tv)
            gaps.append(obs['sq_gaps'].astype(np.float32))               # aligns with vi order
            k = (obs['tokens'][vi] - gen.ix_query).astype(np.int64)      # queried key
            distract.append(gen.distract_accum[vi].astype(np.int64))
            stored_qk.append(gen.stored[vi, k].astype(np.int64))
            net_qk.append((gen.stored_cnt[vi, k] - gen.queried_cnt[vi, k]).astype(np.int64))
            qkey.append(k)

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

    cat = lambda xs: torch.from_numpy(np.concatenate(xs))
    D = {
        'feats':     torch.cat(feats, dim=0),                 # [N, C, H]
        'o_cols':    torch.cat(ocols, dim=0) if ocols else None,
        'tgts':      cat(tgts),                               # [N]
        'gaps':      cat(gaps),                               # [N]
        'correct':   cat(corr),                               # [N] model's own
        'distract':  cat(distract),
        'stored_qk': cat(stored_qk),
        'net_qk':    cat(net_qk),
        'qkey':      cat(qkey),
    }
    attn = np.stack([a / max(attn_cnt, 1) for a in attn_acc], axis=0)  # [L, C, C]
    return D, attn


# sliding accuracy

def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, center - half, center + half


def gap_binned_acc(correct: np.ndarray, gaps: np.ndarray, edges):
    """Accuracy binned by store->query gap; misses (gap < 0) reported separately."""
    res = {}
    miss = gaps < 0
    if miss.any():
        res['miss'] = (float(correct[miss].mean()), int(miss.sum()))
    g, c = gaps[~miss], correct[~miss]
    lo_edges = [0] + list(edges)
    for lo, hi in zip(lo_edges[:-1], lo_edges[1:]):
        m = (g >= lo) & (g < hi)
        res[f'[{lo},{hi})'] = (float(c[m].mean()) if m.any() else float('nan'), int(m.sum()))
    m = g >= lo_edges[-1]
    res[f'[{lo_edges[-1]},inf)'] = (float(c[m].mean()) if m.any() else float('nan'), int(m.sum()))
    return res


def rolling_acc(correct: np.ndarray, window: int) -> np.ndarray:
    c = np.asarray(correct, dtype=float)
    if len(c) < window or window <= 1:
        return c.cumsum() / (np.arange(len(c)) + 1)
    return np.convolve(c, np.ones(window) / window, mode='valid')


def sliding_report(correct: torch.Tensor, gaps: torch.Tensor, edges, window: int) -> dict:
    c = correct.numpy() if torch.is_tensor(correct) else np.asarray(correct)
    g = gaps.numpy() if torch.is_tensor(gaps) else np.asarray(gaps)
    p, lo, hi = wilson_ci(int(c.sum()), len(c))
    named = {k: (float(v.item()) if torch.is_tensor(v) else float(v))
             for k, v in sq_gap_metrics(torch.tensor(c), torch.tensor(g)).items()}
    return {
        'mean': p, 'ci_lo': lo, 'ci_hi': hi, 'n': len(c),
        'named': named,
        'gap_curve': gap_binned_acc(c, g, edges),
        'rolling': rolling_acc(c, window),
    }


def print_sliding(name: str, rep: dict):
    print(f'\n[{name}]  Acc = {rep["mean"]:.3f}  (95% CI [{rep["ci_lo"]:.3f}, {rep["ci_hi"]:.3f}], n={rep["n"]})')
    print('  named  :', {k: round(v, 3) for k, v in rep['named'].items()})
    print('  by gap :', {k: (round(a, 3), n) for k, (a, n) in rep['gap_curve'].items()})


def log_sliding(logger, name: str, rep: dict, step: int):
    logger.track(rep['mean'], name=f'sliding/{name}/acc', step=step)
    logger.track(rep['ci_hi'] - rep['ci_lo'], name=f'sliding/{name}/ci_width', step=step)
    for k, (a, n) in rep['gap_curve'].items():
        if a == a:  # not nan
            logger.track(a, name=f'sliding/{name}/gap/{k}', step=step)
    log_figure(logger, plot_curve(rep['gap_curve'], f'{name}: recall vs gap', 'gap bin', 'Acc'),
               f'sliding/{name}/gap_curve', step)
    log_figure(logger, plot_line(rep['rolling'], f'{name}: rolling Acc', 'query #', 'Acc'),
               f'sliding/{name}/rolling', step)


# linear readout / probe head

def train_readout(X: torch.Tensor, y: torch.Tensor, n_classes: int, device, *,
                  epochs: int = 300, split: float = 0.7, seed: int = 0):
    """Frozen-backbone linear head on X [N, D] -> integer y; fixed split (seed) so subset
    accuracies are comparable. Returns held-out acc + per-example eval correctness/index."""
    N = X.shape[0]
    if N < 32 or n_classes < 2:
        return {'Acc': float('nan'), 'ev_idx': np.array([], int), 'correct': np.array([])}
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    n_tr = int(N * split)
    tr, ev = perm[:n_tr], perm[n_tr:]
    Xtr, Ytr = X[tr].to(device), y[tr].to(device)
    Xev, Yev = X[ev].to(device), y[ev].to(device)
    head = nn.Linear(X.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=5e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(head(Xtr), Ytr).backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        correct = (head(Xev).argmax(dim=-1) == Yev).float().cpu().numpy()
    return {'Acc': float(correct.mean()), 'ev_idx': ev.cpu().numpy(), 'correct': correct}


def n_cls(y: torch.Tensor) -> int:
    return int(y.max().item()) + 1 if y.numel() else 0


def relabel(y: torch.Tensor):
    """Map arbitrary integer labels (possibly negative / sparse) to contiguous [0, K)."""
    uniq, inv = torch.unique(y, return_inverse=True)
    return inv.long(), int(uniq.numel())


# B1. exact direct-logit attribution

@torch.no_grad()
def direct_logit_attr(rnn, o_cols: torch.Tensor, tgts: torch.Tensor, device):
    """Per-column contribution to the correct-class logit from the linear head.
    concat head: W[:, cH:(c+1)H] @ o_c ; mean head (optim): (1/C) W @ o_c."""
    N, C, H = o_cols.shape
    W = rnn.head.weight.detach().to(device)                  # [V, in]
    o = o_cols.to(device)
    tg = tgts.to(device).view(-1, 1)
    contrib = torch.zeros(N, C, device=device)
    # pooled_head was tied to optim in older checkpoints; it is configurable now
    pooled = getattr(rnn, 'pooled_head', rnn.optim)
    for c in range(C):
        Wc = W if pooled else W[:, c * H:(c + 1) * H]         # [V, H]
        logit_c = o[:, c, :] @ Wc.T                           # [N, V]
        if pooled:
            logit_c = logit_c / C
        contrib[:, c] = logit_c.gather(1, tg).squeeze(1)
    argmax_frac = torch.zeros(C, device=device)
    am = contrib.argmax(dim=1)
    for c in range(C):
        argmax_frac[c] = (am == c).float().mean()
    return (contrib.mean(0).cpu().numpy(), contrib.abs().mean(0).cpu().numpy(),
            argmax_frac.cpu().numpy())


# B3. subset selection (<= K_max columns)

def subset_cache(feats: torch.Tensor, tgts: torch.Tensor, n_classes: int, k_max: int,
                 device, epochs: int, seed: int) -> dict:
    """Readout accuracy for every column subset of size 1..k_max (exhaustive; C=4 is cheap)."""
    C = feats.shape[1]
    N = feats.shape[0]
    cache = {}
    for k in range(1, k_max + 1):
        for S in combinations(range(C), k):
            X = feats[:, S, :].reshape(N, -1)
            cache[frozenset(S)] = train_readout(X, tgts, n_classes, device,
                                                 epochs=epochs, seed=seed)['Acc']
    return cache


def greedy_forward(cache: dict, C: int, k_max: int):
    sel, accs = [], []
    for _ in range(min(k_max, C)):
        best, best_a = None, -1.0
        for c in range(C):
            if c in sel:
                continue
            a = cache[frozenset(sel + [c])]
            if a > best_a:
                best_a, best = a, c
        sel.append(best)
        accs.append(best_a)
    return sel, accs


def minimal_sufficient(cache: dict, C: int, k_max: int, full_acc: float, eps: float):
    for k in range(1, min(k_max, C) + 1):
        best_S, best_a = None, -1.0
        for S in combinations(range(C), k):
            a = cache[frozenset(S)]
            if a > best_a:
                best_a, best_S = a, S
        if best_a >= full_acc - eps:
            return list(best_S), best_a
    return list(range(min(k_max, C))), full_acc


def marginal_values(cache: dict, C: int, k_max: int, chance: float) -> np.ndarray:
    """Fair per-column value: mean marginal gain acc(S+c)-acc(S) over all coalitions
    S (not containing c) of size < k_max. Exact truncated-coalition Shapley for small C."""
    vals = np.zeros(C)
    for c in range(C):
        others = [x for x in range(C) if x != c]
        gains = []
        for k in range(0, min(k_max, C)):
            for S in combinations(others, k):
                accS = chance if k == 0 else cache[frozenset(S)]
                gains.append(cache[frozenset(S + (c,))] - accS)
        vals[c] = float(np.mean(gains))
    return vals


# B2. causal ablation via col_keep_mask

def ablate_collect(rnn, make_gen, device, n_collect: int, keep_cols, mode: str):
    C = rnn.n_columns
    mask = torch.zeros(C, device=device)
    mask[list(keep_cols)] = 1.0
    rnn.col_keep_mask, rnn.col_ablate_mode = mask, mode
    D, _ = collect(rnn, make_gen(), device, n_collect)
    rnn.col_keep_mask = None
    return D


# plots

def plot_bars(values: dict, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(max(4, len(values) * 0.8), 3.2))
    keys = list(values.keys())
    ax.bar(range(len(keys)), [values[k] for k in keys], color='steelblue')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha='right')
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def plot_curve(curve: dict, title: str, xlabel: str, ylabel: str):
    keys = [k for k in curve if curve[k][0] == curve[k][0]]  # drop nan bins
    fig, ax = plt.subplots(figsize=(max(4, len(keys) * 0.7), 3.2))
    ax.plot(range(len(keys)), [curve[k][0] for k in keys], 'o-', color='darkorange')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha='right')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_line(series: np.ndarray, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(5, 3.0))
    ax.plot(series, color='seagreen')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_heatmap(mat: np.ndarray, row_labels, col_labels, title: str):
    fig, ax = plt.subplots(figsize=(max(4, len(col_labels) * 0.9), max(3, len(row_labels) * 0.5)))
    im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v == v:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        color='white' if v < 0.6 else 'black', fontsize=7)
    ax.set_title(title); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


# consolidated relationship figures (fold many per-column scalars into few charts)

def plot_agreement(metrics: dict, col_labels, title: str):
    """Heatmap [method x column]. Each method row is min-max normalized across columns
    (color) while the raw value is annotated, so different-scale attribution methods can be
    compared side by side: agreement = same column bright across all rows."""
    names = list(metrics)
    M = np.array([[float(v) for v in metrics[n]] for n in names], dtype=float)  # [methods, cols]
    Mn = np.zeros_like(M)
    for i in range(M.shape[0]):
        lo, hi = np.nanmin(M[i]), np.nanmax(M[i])
        Mn[i] = (M[i] - lo) / (hi - lo) if hi > lo else np.full_like(M[i], 0.5)
    fig, ax = plt.subplots(figsize=(max(4, len(col_labels) * 1.1), max(3, len(names) * 0.55)))
    im = ax.imshow(Mn, aspect='auto', cmap='magma', vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] == M[i, j]:
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                        color='white' if Mn[i, j] < 0.55 else 'black', fontsize=7)
    ax.set_title(title); fig.colorbar(im, ax=ax, fraction=0.046, label='rank (row-normalized)')
    fig.tight_layout()
    return fig


def plot_gap_overlay(curves: dict, title: str):
    """Recall-vs-gap for several conditions (full model + each ablation) on shared axes."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ref = next(iter(curves.values()))
    keys = [k for k in ref if k != 'miss']
    # matplotlib's default cycle is 10 colors: at 12 columns + full + keep-subset it
    # wraps and -C0/-C10, -C1/-C11, -C2/keep become indistinguishable
    cmap = plt.get_cmap('turbo')
    n = max(len(curves) - 1, 1)
    for i, (label, curve) in enumerate(curves.items()):
        ys = [curve.get(k, (float('nan'),))[0] for k in keys]
        is_full = label == 'full'
        color = 'black' if is_full else cmap(i / n)
        style = '-' if is_full else '--'
        lw = 2.6 if is_full else 1.3
        ax.plot(range(len(keys)), ys, 'o' + style, color=color, lw=lw, ms=3, label=label)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=45, ha='right')
    ax.set_xlabel('store->query gap'); ax.set_ylabel('Acc'); ax.set_title(title)
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def plot_subset_scaling(best_by_size: dict, greedy_accs, full_acc: float, chance: float, title: str):
    """Readout accuracy vs number of columns: exhaustive-best vs greedy, capped at K_max."""
    sizes = sorted(best_by_size)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(sizes, [best_by_size[k][1] for k in sizes], 'o-', color='steelblue', label='best (exhaustive)')
    ax.plot(sizes, list(greedy_accs)[:len(sizes)], 's--', color='darkorange', label='greedy')
    ax.axhline(full_acc, color='green', ls=':', label=f'all columns ({full_acc:.2f})')
    ax.axhline(chance, color='gray', ls=':', label=f'chance ({chance:.2f})')
    ax.set_xticks(sizes); ax.set_xlabel('# columns in subset'); ax.set_ylabel('SDQ readout Acc')
    ax.set_title(title); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


# main

def main():
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--device', default='cuda')
    # 800 steps leaves only ~44 queries in the [32,64) gap bin, i.e. the long-gap end
    # of the recall curve -- exactly where ablations differ -- is pure noise
    ap.add_argument('--n_collect', type=int, default=4000)
    ap.add_argument('--subset_max_k', type=int, default=4, help='max columns per subset (<=4)')
    ap.add_argument('--ablate_mode', default='zero', choices=['zero', 'mean'])
    ap.add_argument('--rolling_window', type=int, default=200)
    ap.add_argument('--gap_edges', default='2,4,8,16,32,64,128',
                    help='comma-separated upper edges for the recall-vs-gap curve')
    ap.add_argument('--epochs', type=int, default=300, help='readout/probe epochs')
    ap.add_argument('--subset_epochs', type=int, default=200)
    ap.add_argument('--suff_eps', type=float, default=0.02, help='minimal-sufficient tolerance')
    ap.add_argument('--name', default=None)
    # 'hard' puts only ~264 of 178K queries in the [32,64) gap bin; gens.long (T=40,
    # p_store/p_query 0.15) is eval-only and populates the long-gap end of the curve
    ap.add_argument('--eval_gen', default=None, help='gens.<name> to evaluate on (default: training gen)')
    args, extra = ap.parse_known_args()
    overrides = {k: parse_str(v) for k, v in
                 (a.lstrip('-').split('=', 1) for a in extra if '=' in a)}
    edges = [int(x) for x in args.gap_edges.split(',') if x]
    k_max = min(args.subset_max_k, 4)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config, rnn_type = ckpt['config'], ckpt['model_type']
    print(f'Loaded {rnn_type} @ step {ckpt["step"]:,} from {args.checkpoint}')

    gen_name = args.eval_gen or config['gen']
    if gen_name not in config['gens']:
        raise SystemExit(f'gen {gen_name!r} not in checkpoint config: {list(config["gens"])}')
    gen_cfg = config['gens'][gen_name]
    if gen_name != config['gen']:
        print(f'Evaluating on gens.{gen_name} (trained on gens.{config["gen"]}): {gen_cfg}')
    make_gen = lambda: StoreDistractQueryGenerator(**gen_cfg, n_envs=config['n_envs'],
                                                   seed=0, ignore_index=CE_ignore_index)
    gen = make_gen()
    rnn = build_model(rnn_type, config['models'][rnn_type], gen).to(device)
    rnn.load_state_dict(ckpt['model_state'])
    V, C = gen.V, rnn.n_columns
    chance = 1.0 / V

    # logger
    log_cfg = dict(config.get('log', {}))
    for k, v in overrides.items():
        if k.startswith('log.'):
            log_cfg[k[4:]] = v
    log_cfg.setdefault('enabled', True)
    log_cfg.setdefault('logger', 'comet')
    log_cfg.setdefault('project', 'grid-rnn-sdq')
    log_cfg['name'] = args.name or log_cfg.get('name') or f'{rnn_type}_col_analysis_{gen_name}'
    logger = create_logger({'log': log_cfg} | {'model': rnn_type})
    step = int(ckpt['step'])

    # ---- collect baseline (model's own behavior) ----
    D, attn = collect(rnn, gen, device, args.n_collect)
    feats, o_cols, tgts, gaps = D['feats'], D['o_cols'], D['tgts'], D['gaps']
    print(f'Collected N={feats.shape[0]} queries, feats={tuple(feats.shape)}, '
          f'pooled_head={getattr(rnn, "pooled_head", rnn.optim)}')

    # ---- sliding accuracy (replaces the single mean) ----
    base = sliding_report(D['correct'], gaps, edges, args.rolling_window)
    print_sliding('model (full)', base)
    log_sliding(logger, 'full', base, step)

    # ---- attention interaction + activation norms (kept) ----
    to_c0 = {f'C{c}': float(attn[:, c, 0].mean()) for c in range(1, C)}
    from_c0 = {f'C{c}': float(attn[:, 0, c].mean()) for c in range(1, C)}
    interaction = {f'C{c}': to_c0[f'C{c}'] + from_c0[f'C{c}'] for c in range(1, C)}
    act_norm = {f'C{c}': float(feats[:, c, :].norm(dim=-1).mean()) for c in range(C)}
    print('interaction with C0:', {k: round(v, 3) for k, v in interaction.items()})
    print('activation norm    :', {k: round(v, 3) for k, v in act_norm.items()})
    av = AttnFlowVisualizer(n_layers=rnn.n_layers, n_columns=C)
    for li in range(rnn.n_layers):
        av._buffers[li].append(attn[li])
    av.log(logger, step=step)
    attn_centrality = {f'C{c}': float(attn[:, :, c].mean()) for c in range(C)}  # mean attn received

    # ---- A. content probing: what each column stores ----
    raw_vars = {
        'stored_val': D['stored_qk'], 'distract': D['distract'],
        'net_count': D['net_qk'], 'queried_key': D['qkey'], 'target': tgts,
    }
    # relabel to contiguous non-negative classes (net_count can be negative / sparse)
    variables = {}
    for k, v in raw_vars.items():
        yl, nc = relabel(v)
        if nc >= 2:                                                    # drop degenerate
            variables[k] = (yl, nc)
    var_names = list(variables)
    probe_mat = np.full((C, len(var_names)), np.nan)                    # [col, var]
    full_probe = {}
    for j, vn in enumerate(var_names):
        y, nc = variables[vn]
        for c in range(C):
            probe_mat[c, j] = train_readout(feats[:, c, :], y, nc, device, epochs=args.epochs)['Acc']
        full_probe[vn] = train_readout(feats.reshape(feats.shape[0], -1), y, nc, device,
                                       epochs=args.epochs)['Acc']
    selectivity = {f'C{c}': float(np.nanmax(probe_mat[c]) - np.nanmean(probe_mat[c])) for c in range(C)}
    print('\ncontent probe [col x var]:')
    for c in range(C):
        print(f'  C{c}:', {vn: round(float(probe_mat[c, j]), 3) for j, vn in enumerate(var_names)})
    print('full-set probe :', {k: round(v, 3) for k, v in full_probe.items()})
    print('selectivity    :', {k: round(v, 3) for k, v in selectivity.items()})
    log_figure(logger, plot_heatmap(probe_mat, [f'C{c}' for c in range(C)], var_names,
                                    'Content probe accuracy'), 'probe/matrix', step)
    for c in range(C):
        for j, vn in enumerate(var_names):
            logger.track(probe_mat[c, j], name=f'probe/{vn}/C{c}', step=step)
    for k, v in selectivity.items():
        logger.track(v, name=f'probe/selectivity/{k}', step=step)

    # ---- B1. exact direct-logit attribution ----
    logit_share = [float('nan')] * C
    if o_cols is not None:
        signed, absmean, amfrac = direct_logit_attr(rnn, o_cols, tgts, device)
        logit_share = [float(amfrac[c]) for c in range(C)]
        print('\ndirect-logit (signed to correct class):', {f'C{c}': round(float(signed[c]), 3) for c in range(C)})
        print('direct-logit (mean |contrib|)         :', {f'C{c}': round(float(absmean[c]), 3) for c in range(C)})
        print('direct-logit (argmax share)           :', {f'C{c}': round(float(amfrac[c]), 3) for c in range(C)})
        for c in range(C):
            logger.track(float(signed[c]), name=f'attr/direct_logit/C{c}', step=step)
            logger.track(float(amfrac[c]), name=f'attr/argmax_share/C{c}', step=step)
    else:
        print('\n[warn] no o_cols in extras -> skipping direct-logit attribution')

    # ---- B3. subset selection (readout value), capped at k_max ----
    cache = subset_cache(feats, tgts, V, k_max, device, args.subset_epochs, seed=0)
    per_col = {f'C{c}': cache[frozenset([c])] for c in range(C)}
    full_key = frozenset(range(C)) if C <= k_max else None
    full_acc = cache[full_key] if full_key in cache else max(
        cache[frozenset(S)] for S in combinations(range(C), min(k_max, C)))
    gsel, gaccs = greedy_forward(cache, C, k_max)
    suff_S, suff_a = minimal_sufficient(cache, C, k_max, full_acc, args.suff_eps)
    margin = marginal_values(cache, C, k_max, chance)
    best_by_size = {}
    for k in range(1, min(k_max, C) + 1):
        S = max(combinations(range(C), k), key=lambda s: cache[frozenset(s)])
        best_by_size[k] = (list(S), cache[frozenset(S)])
    print('\nsubset readout (target):')
    print('  per-column     :', {k: round(v, 3) for k, v in per_col.items()})
    print('  best by size   :', {k: (S, round(a, 3)) for k, (S, a) in best_by_size.items()})
    print(f'  greedy order   : {gsel}  accs={[round(a,3) for a in gaccs]}')
    print(f'  min-sufficient : {suff_S}  acc={round(suff_a,3)}  (full={round(full_acc,3)}, eps={args.suff_eps})')
    print('  marginal value :', {f'C{c}': round(float(margin[c]), 3) for c in range(C)})
    for c in range(C):
        logger.track(float(margin[c]), name=f'attr/marginal_value/C{c}', step=step)
        logger.track(per_col[f'C{c}'], name=f'ablation/readout_acc/single/C{c}', step=step)
    logger.track(full_acc, name='ablation/readout_acc/full', step=step)
    logger.track(suff_a, name='ablation/readout_acc/min_sufficient', step=step)
    logger.track(float(len(suff_S)), name='ablation/min_sufficient_size', step=step)

    # ---- B2. causal ablation on the model's OWN head ----
    print('\ncausal ablation (drop 1 column, model own acc):')
    gap_curves = {'full': base['gap_curve']}                 # for the recall-vs-gap overlay
    causal_drop = [0.0] * C
    for c in range(C):
        keep = [x for x in range(C) if x != c]
        Da = ablate_collect(rnn, make_gen, device, args.n_collect, keep, args.ablate_mode)
        rep = sliding_report(Da['correct'], Da['gaps'], edges, args.rolling_window)
        causal_drop[c] = base['mean'] - rep['mean']
        gap_curves[f'-C{c}'] = rep['gap_curve']
        print(f'  drop C{c}: acc={rep["mean"]:.3f}  (delta={causal_drop[c]:+.3f})')
        logger.track(rep['mean'], name=f'causal/drop_C{c}/acc', step=step)
        logger.track(causal_drop[c], name=f'causal/drop_C{c}/delta', step=step)
    # keep only the minimal-sufficient subgroup
    Ds = ablate_collect(rnn, make_gen, device, args.n_collect, suff_S, args.ablate_mode)
    rep_s = sliding_report(Ds['correct'], Ds['gaps'], edges, args.rolling_window)
    gap_curves[f'keep{suff_S}'] = rep_s['gap_curve']
    print_sliding(f'keep-only {suff_S}', rep_s)
    logger.track(rep_s['mean'], name='causal/keep_subgroup/acc', step=step)

    # ---- B4. redundancy (linear CKA between columns) ----
    ncka = min(2000, feats.shape[0])
    idx = torch.randperm(feats.shape[0])[:ncka]
    cka = np.eye(C)
    for i in range(C):
        for j in range(i + 1, C):
            v = linear_cka(feats[idx, i, :].numpy(), feats[idx, j, :].numpy())
            cka[i, j] = cka[j, i] = v
    print('\ncolumn CKA (redundancy):')
    for i in range(C):
        print(f'  C{i}:', [round(float(cka[i, j]), 2) for j in range(C)])
    log_figure(logger, plot_heatmap(cka, [f'C{c}' for c in range(C)], [f'C{c}' for c in range(C)],
                                    'Column CKA (redundancy)'), 'redundancy/cka', step)

    # ---- consolidated relationship figures (fold the per-column scalars into 3 charts) ----
    col_labels = [f'C{c}' for c in range(C)]
    agreement = {
        'readout_single': [per_col[f'C{c}'] for c in range(C)],
        'marginal_value': [float(margin[c]) for c in range(C)],
        'causal_drop':    causal_drop,
        'logit_share':    logit_share,
        'act_norm':       [act_norm[f'C{c}'] for c in range(C)],
        'attn_centrality':[attn_centrality[f'C{c}'] for c in range(C)],
    }
    agreement = {k: v for k, v in agreement.items() if any(x == x for x in v)}  # drop all-nan rows
    log_figure(logger, plot_agreement(agreement, col_labels,
               'Column importance: attribution methods vs columns'), 'summary/attribution_agreement', step)
    log_figure(logger, plot_gap_overlay(gap_curves,
               'Recall vs gap: full model + causal ablations'), 'summary/recall_vs_gap', step)
    log_figure(logger, plot_subset_scaling(best_by_size, gaccs, full_acc, chance,
               'Readout accuracy vs subset size (<=K_max)'), 'summary/subset_scaling', step)

    logger.end()
    print('\nDone.')


if __name__ == '__main__':
    main()
