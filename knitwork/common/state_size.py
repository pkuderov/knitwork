"""Params / carried-state accounting for text configs; optional solver for a size knob.

uv run python -m knitwork.common.state_size knitwork/exps/text/config/mid.yaml [--keys a,b]
uv run python -m knitwork.common.state_size <cfg> --solve rnn_L2 --target state=3392
uv run python -m knitwork.common.state_size <cfg> --solve delta_net --target params=1e6
"""
from __future__ import annotations

import argparse

import torch
import yaml

from knitwork.common.utils import count_learnable_params
from knitwork.models.utils import REGISTRY, build_model

# state entries that are not carried information: transformer bookkeeping, unused grnn_lru per-layer outputs
_SKIP = {'valid', 'pos', 'outs'}


def _leaves(x, key=None):
    if isinstance(x, torch.Tensor):
        if key not in _SKIP:
            yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from _leaves(v, k)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _leaves(v, key)


def state_floats(rnn) -> int:
    """Floats carried between time steps per sequence; views of already counted tensors are skipped."""
    if hasattr(rnn, 'carried_state_floats'):
        return rnn.carried_state_floats()
    ts = list(_leaves(rnn.init_state(1)))
    ids = {id(t) for t in ts}
    return sum(t.numel() for t in ts if t._base is None or id(t._base) not in ids)


def model_type(key: str) -> str:
    return max((m for m in REGISTRY if key == m or key.startswith(m + '_')), key=len)


def build(cfg: dict, key: str, overrides: dict | None = None, vocab: int = 27):
    wrapper_cfg = cfg[f'{cfg["wrapper_model"]}_wrapper'] | dict(
        input_size=vocab, output_size=vocab, dtype=torch.float32, device='cpu',
    )
    return build_model(
        wrapper_type=cfg['wrapper_model'], wrapper_cfg=wrapper_cfg,
        rnn_type=model_type(key), rnn_cfg=cfg[key] | (overrides or {}),
    )


def measure(cfg, key, overrides=None):
    m = build(cfg, key, overrides)
    return count_learnable_params(m), state_floats(m.rnn), m.rnn.hidden_size


def solve(cfg, key, metric, target, var='hidden_size', lo=4, hi=4096):
    """Smallest value of `var` whose params/state is >= target (both are monotonic in the size knob)."""
    idx = {'params': 0, 'state': 1}[metric]
    while lo < hi:
        mid = (lo + hi) // 2
        if measure(cfg, key, {var: mid})[idx] >= target:
            hi = mid
        else:
            lo = mid + 1
    # pick the closer of lo and lo-1
    best = min((lo - 1, lo), key=lambda v: abs(measure(cfg, key, {var: v})[idx] - target))
    return best, measure(cfg, key, {var: best})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--keys', default=None, help='comma-separated config sections; default: all registered')
    ap.add_argument('--solve', default=None)
    ap.add_argument('--target', default=None, help='params=N or state=N')
    ap.add_argument('--var', default='hidden_size')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.solve:
        metric, target = args.target.split('=')
        v, (p, s, h) = solve(cfg, args.solve, metric, float(target), var=args.var)
        print(f'{args.solve}: {args.var}={v} -> params={p:,} state={s:,} hidden={h}')
        return

    keys = args.keys.split(',') if args.keys else [
        k for k, v in cfg.items()
        if isinstance(v, dict) and k not in ('log', 'eval', 'gens', 'lr', 'trackers', 'communication', 'token_wrapper')
        and any(k == m or k.startswith(m + '_') for m in REGISTRY)
    ]
    print(f'{"config":28s} {"params":>12s} {"state":>12s} {"hidden":>7s}')
    for k in keys:
        try:
            p, s, h = measure(cfg, k)
            print(f'{k:28s} {p:12,d} {s:12,d} {h:7d}')
        except Exception as e:
            print(f'{k:28s} ERROR {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
