"""Aligned text segments and independent SDQ distribution/fixed-gap evaluations."""

import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from knitwork.common.utils import CE_ignore_index
from knitwork.gens.sdq import StoreDistractQueryGenerator
from knitwork.gens.text import TextGenerator


@torch.no_grad()
def evaluate_text(model, data, cfg, *, segment_path=None, check_stop=None):
    device = next(model.parameters()).device
    n_envs = cfg['n_envs']
    segment_len = cfg['segment_len']
    steps = len(data) // n_envs
    if cfg.get('max_tokens') is not None:
        steps = min(steps, int(cfg['max_tokens']) // n_envs)
    if steps < 1:
        raise ValueError('Evaluation data/cap must allow at least one step per stream')
    gen = TextGenerator(data, n_envs=n_envs, device=device, seed=cfg['seed'], ignore_index=CE_ignore_index)
    gen = gen.to(device)
    n_segments = math.ceil(steps / segment_len)
    losses = torch.zeros(n_envs, n_segments, dtype=torch.float64, device=device)
    counts = torch.zeros(n_envs, n_segments, dtype=torch.int64, device=device)
    correct = torch.zeros((), dtype=torch.int64, device=device)
    state = None
    was_training = model.training
    model.eval()
    try:
        for t in range(steps):
            if check_stop is not None:
                check_stop()
            obs = gen.next()
            state = model.rnn.reset_state(state, obs['reset_mask'])
            y, state, _ = model(obs['tokens'].view(1, -1), state)
            valid = obs['targets'] != CE_ignore_index
            ce = F.cross_entropy(y, obs['targets'], ignore_index=CE_ignore_index, reduction='none')
            losses[:, t // segment_len] += ce.double()
            counts[:, t // segment_len] += valid
            correct += ((y.argmax(-1) == obs['targets']) & valid).sum()
    finally:
        model.train(was_training)
    count = int(counts.sum())
    if count == 0:
        raise ValueError('Text evaluation has no valid targets')
    loss = float(losses.sum()) / count
    if segment_path is not None:
        np.savez_compressed(
            Path(segment_path), loss_sums=losses.cpu().numpy(), counts=counts.cpu().numpy(),
            stream_starts=gen.pos0.cpu().numpy(), segment_len=segment_len,
        )
    return {'Loss': loss, 'BPC': loss / math.log(2), 'Acc': float(correct) / count, 'scored_tokens': count}


@torch.no_grad()
def evaluate_sdq(model, generator_cfg, cfg, *, result_path=None, check_stop=None):
    device = next(model.parameters()).device
    gen = StoreDistractQueryGenerator(
        **generator_cfg, n_envs=cfg['n_envs'], seed=cfg['seed'], ignore_index=CE_ignore_index,
    )
    state = None
    edges = np.asarray(cfg['gap_edges'])
    counts = np.zeros(len(edges) + 1, dtype=np.int64)
    correct = np.zeros_like(counts)
    loss_sum = 0.0
    query_count = 0
    total_correct = 0
    was_training = model.training
    model.eval()
    try:
        for _ in range(cfg['sdq_steps']):
            if check_stop is not None:
                check_stop()
            obs = gen.next()
            state = model.rnn.reset_state(state, torch.from_numpy(obs['reset_mask']).to(device))
            y, state, _ = model(torch.from_numpy(obs['tokens']).to(device).view(1, -1), state)
            targets = torch.from_numpy(obs['targets']).to(device)
            valid = targets != CE_ignore_index
            active = y[valid]
            successes = (active.argmax(-1) == targets[valid]).cpu().numpy()
            loss_sum += float(F.cross_entropy(y, targets, ignore_index=CE_ignore_index, reduction='sum'))
            query_count += int(valid.sum())
            total_correct += int(successes.sum())
            gaps = obs['sq_gaps']
            # Misses form bin 0; observed stores are binned by absolute gap.
            bins = np.where(gaps < 0, 0, 1 + np.searchsorted(edges[1:], gaps, side='right'))
            counts += np.bincount(bins, minlength=len(counts))
            correct += np.bincount(bins, weights=successes, minlength=len(counts)).astype(np.int64)
    finally:
        model.train(was_training)
    if query_count == 0:
        raise ValueError('SDQ evaluation has no queries')
    if result_path is not None:
        np.savez_compressed(result_path, gap_edges=edges, counts=counts, correct=correct)
    metrics = {'Loss': loss_sum / query_count, 'Acc': total_correct / query_count, 'queries': query_count}
    for i, (n, k) in enumerate(zip(counts, correct)):
        metrics[f'gap_bin{i}/count'] = int(n)
        if n:
            metrics[f'gap_bin{i}/Acc'] = int(k) / int(n)
    nonmiss_count = int(counts[1:].sum())
    if nonmiss_count:
        metrics['Acc_nonmiss'] = int(correct[1:].sum()) / nonmiss_count
    return metrics


def fixed_gap_batch(generator_cfg, *, gap, n_examples, seed):
    """A forced initial store and final query with exactly ``gap`` steps between.

    Intermediate events follow the final curriculum's store/query probabilities,
    using other keys to preserve the designated store. Targets retain the native
    distractor accumulator and store/query counter semantics.
    """
    K, V = generator_cfg['n_keys'], generator_cfg['n_vals']
    if K < 2 or gap < 1:
        raise ValueError('Fixed-gap probes require at least two keys and gap >= 1')
    rng = np.random.default_rng(seed)
    ixs = np.arange(n_examples)
    chosen = rng.integers(K, size=n_examples)
    values = rng.integers(V, size=n_examples)
    stored = np.zeros((n_examples, K), dtype=np.int64)
    store_count = np.zeros_like(stored)
    query_count = np.zeros_like(stored)
    distract = np.zeros(n_examples, dtype=np.int64)
    tokens = np.empty((gap + 1, n_examples), dtype=np.int64)
    tokens[0] = chosen * V + values
    stored[ixs, chosen] = values
    store_count[ixs, chosen] = int(generator_cfg['count_stored'])
    for t in range(1, gap):
        event = rng.random(n_examples)
        other = rng.integers(K - 1, size=n_examples)
        other += other >= chosen
        v = rng.integers(V, size=n_examples)
        stores = event < generator_cfg['p_store']
        queries = (~stores) & (event < generator_cfg['p_store'] + generator_cfg['p_query'])
        fillers = ~(stores | queries)
        tokens[t, stores] = other[stores] * V + v[stores]
        stored[ixs[stores], other[stores]] = v[stores]
        store_count[ixs[stores], other[stores]] += int(generator_cfg['count_stored'])
        tokens[t, queries] = K * V + V + other[queries]
        query_count[ixs[queries], other[queries]] += int(generator_cfg['count_queried'])
        tokens[t, fillers] = K * V + v[fillers]
        distract[fillers] = (distract[fillers] + v[fillers]) % V
    tokens[gap] = K * V + V + chosen
    query_count[ixs, chosen] += int(generator_cfg['count_queried'])
    targets = (stored[ixs, chosen] + distract + store_count[ixs, chosen] - query_count[ixs, chosen]) % V
    return tokens, targets


@torch.no_grad()
def evaluate_fixed_gaps(model, generator_cfg, cfg, *, result_path=None, check_stop=None):
    device = next(model.parameters()).device
    metrics = {}
    rows = []
    was_training = model.training
    model.eval()
    try:
        for gap in cfg['fixed_gaps']:
            tokens, targets = fixed_gap_batch(
                generator_cfg, gap=gap, n_examples=cfg['fixed_gap_examples'], seed=cfg['probe_seed'] + gap,
            )
            predictions = []
            losses = []
            for start in range(0, len(targets), cfg['n_envs']):
                end = min(start + cfg['n_envs'], len(targets))
                state = model.rnn.init_state(end - start)
                for x in tokens[:, start:end]:
                    if check_stop is not None:
                        check_stop()
                    y, state, _ = model(torch.from_numpy(x).to(device).view(1, -1), state)
                target = torch.from_numpy(targets[start:end]).to(device)
                predictions.extend(y.argmax(-1).cpu().tolist())
                losses.extend(F.cross_entropy(y, target, reduction='none').cpu().tolist())
            success = np.asarray(predictions) == targets
            metrics[f'gap{gap}/Acc'] = float(success.mean())
            metrics[f'gap{gap}/Loss'] = float(np.mean(losses))
            metrics[f'gap{gap}/count'] = len(targets)
            rows.append(success)
    finally:
        model.train(was_training)
    metrics['macro_Acc'] = float(np.mean([metrics[f'gap{gap}/Acc'] for gap in cfg['fixed_gaps']]))
    if result_path is not None:
        np.savez_compressed(result_path, gaps=cfg['fixed_gaps'], correct=np.stack(rows))
    return metrics
