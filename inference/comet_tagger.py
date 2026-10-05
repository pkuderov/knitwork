"""Comet tag scheme for the AISTATS text series: one source of truth for new runs (log.tags) and for tagging existing ones.

Run names: `mid/<model>` or `lru_mid/<variant>`, plus `/seed<S>` for seed replicates.
Tags are flat `key:value` strings, so Comet's tag filter can combine them
(e.g. `family:mosaic` AND `role:main`):
  series:aistats27   tier:mid        data:text8   split:90-5-5   budget:1e9
  family:<rnn|mosaic|mosaic-lru|lru-sparse|delta-net|hgrn2|mlstm|transformer|mamba|rims|brims>
  model:<cfg spec>   arch:L2C4 (grids)   variant:<lru-sparse variant>   ablation:topk1
  seed:<S>           role:main (seed 42) | role:seed-rep
  reg:comm-aux       (MoSAIC with communication loss + noise)       fbnorm:on  (LRU feedback RMS norm)

uv run python inference/comet_tagger.py --logs /storage/annenkov_vd/knitwork/queue/logs [--dry]    # by experiment keys found in job logs
uv run python inference/comet_tagger.py --project knitwork-aistat --workspace team-rl-exp [--dry]   # by listing the project (can time out)
Needs COMET_API_KEY. Idempotent: only missing tags are added.
"""
from __future__ import annotations

import argparse
import os
import re

BASE_SEED = 42
FAMILY = {
    'rnn': 'rnn', 'grnn': 'mosaic', 'grnn_lru': 'mosaic-lru', 'delta_net': 'delta-net', 'hgrn2': 'hgrn2',
    'mlstm': 'mlstm', 'transformer': 'transformer', 'mamba': 'mamba', 'rims': 'rims', 'brims': 'brims',
}
ARCH_RE = re.compile(r'(?P<task>text|mqar)_lru_architecture_mid/(?P<variant>[a-z_]+)(?:/seed(?P<seed>\d+))?(?:/ep(?P<ep>\d+))?\s*$')
ROLLOUT_RE = re.compile(r'rollout_mid/(?P<fam>grnn_gru|grnn_lru|gru|lru)/r(?P<r>\d+)/(?P<tag>[A-Za-z0-9]+)\s*$')
ABLATION_RE = re.compile(r'grnn_ablation_mid/(?P<variant>[a-z_]+)/(?P<tag>[A-Za-z0-9]+)\s*$')
MIKASA_RE = re.compile(r'rl_mid_shortlist/(?P<run>\d\d)_(?P<env>[A-Za-z]+)_(?P<model>[a-z_]+)/(?P<budget>\d+)slots/seed(?P<seed>\d+)\s*$')
PLAN_FAMILY = {'grnn_gru': 'mosaic', 'grnn_lru': 'mosaic-lru', 'gru': 'rnn', 'lru': 'lru'}
# (noise_std, communication loss weight, entropy weight) of the Grid-GRU ablation files
ABLATION = {'full': (0.05, 0.05, 0.005), 'none': (0, 0, 0), 'no_noise': (0, 0.05, 0.005), 'no_comm': (0.05, 0, 0.005),
            'no_entropy': (0.05, 0.05, 0), 'noise_only': (0.05, 0, 0), 'comm_only': (0, 0.05, 0), 'entropy_only': (0, 0, 0.005)}
NAME_RE = re.compile(r'(?P<series>mid|lru_mid)/(?P<model>[^/\s]+)(?:/seed(?P<seed>\d+))?\s*$')


def tags_for(series: str, model: str, seed: int = BASE_SEED) -> list[str]:
    tags = ['series:aistats27', 'tier:mid', 'data:text8', 'split:90-5-5', 'budget:1e9',
            f'seed:{seed}', 'role:main' if seed == BASE_SEED else 'role:seed-rep', f'model:{series}/{model}']
    if series == 'lru_mid':
        tags += ['family:lru-sparse', f'variant:{model}', 'fbnorm:on']
        return tags
    base, _, arch = model.partition('.')
    tags.append(f'family:{FAMILY.get(base, base)}')
    m = re.fullmatch(r'(L\d+C?\d*)(?:_(topk\d+))?', arch)
    if m and base in ('grnn', 'grnn_lru', 'rnn'):
        tags.append(f'arch:{m.group(1)}')
        if m.group(2):
            tags.append(f'ablation:{m.group(2)}')
    if base == 'grnn':
        tags.append('reg:comm-aux')
    if base == 'grnn_lru':
        tags.append('fbnorm:on')
    return tags


def arch_tags(task: str, variant: str, seed: int, epochs: int = 32) -> list[str]:
    budget = ['data:text8', 'split:90-5-5', 'budget:1e9'] if task == 'text' else ['data:mqar-v8192', f'budget:{epochs}ep']
    return ['series:lru-arch', 'tier:mid', f'task:{"text8" if task == "text" else task}', *budget,
            'family:lru-experimental' if variant != 'dynamic_control' else 'family:lru-sparse', f'mechanism:{variant}',
            'fbnorm:on', f'seed:{seed}', 'role:main' if seed == BASE_SEED else 'role:seed-rep',
            f'model:{task}_lru_architecture_mid/{variant}']


def plan_tags(name: str, seed: int = BASE_SEED) -> list[str] | None:
    """Tags of the rollout / Grid-GRU ablation pilots (2026-10-03 plan)."""
    common = ['tier:mid', 'data:text8', 'split:90-5-5', f'seed:{seed}', 'role:main' if seed == BASE_SEED else 'role:seed-rep']
    if m := ROLLOUT_RE.search(name or ''):
        return ['series:rollout-pilot', f'family:{PLAN_FAMILY[m["fam"]]}', f'model:{m["fam"]}', f'rollout:r{m["r"]}',
                f'budget:{m["tag"]}', *common]
    if m := ABLATION_RE.search(name or ''):
        noise, comm, ent = ABLATION[m['variant']]
        return ['series:grnn-ablation', 'family:mosaic', f'ablation:{m["variant"]}', f'noise:{noise}', f'comm:{comm}', f'entropy:{ent}',
                f'budget:{m["tag"]}', *common]
    return None


def mikasa_tags(name: str) -> list[str] | None:
    if not (m := MIKASA_RE.search(name or '')):
        return None
    seed = int(m['seed'])
    return ['series:mikasa-rl-shortlist', 'tier:mid', 'task:mikasa', f'env:{m["env"]}', f'model:{m["model"]}',
            f'budget:{m["budget"]}slots', f'seed:{seed}', 'role:main' if seed == BASE_SEED else 'role:seed-rep']


def tags_from_name(name: str, seed: int | None = None) -> list[str] | None:
    if (t := mikasa_tags(name)) is not None:
        return t
    if (p := plan_tags(name, seed or BASE_SEED)) is not None:
        return p
    a = ARCH_RE.search(name or '')
    if a:
        return arch_tags(a['task'], a['variant'], int(a['seed']) if a['seed'] else (seed or BASE_SEED), int(a['ep']) if a['ep'] else 32)
    m = NAME_RE.search(name or '')
    if not m:
        return None
    s = int(m['seed']) if m['seed'] else (seed if seed is not None else BASE_SEED)
    return tags_for(m['series'], m['model'], s)


KEY_RE = re.compile(r'Experiment is live on comet\.com \S+/([0-9a-f]{32})')


def tag_by_logs(log_dir, dry):
    from pathlib import Path
    from comet_ml import API
    api = API(api_key=os.environ['COMET_API_KEY'])
    for lf in sorted(Path(log_dir).glob('*.log')):
        m = KEY_RE.search(lf.read_text(errors='ignore'))
        if not m:
            continue
        exp = api.get_experiment_by_key(m[1])
        name = exp.get_name()
        tags = tags_from_name(name)
        if tags is None:
            print(f'skip {lf.name}: unrecognised name {name!r}')
            continue
        new = [t for t in tags if t not in set(exp.get_tags())]
        print(f'{lf.stem}: +{len(new)}')
        if new and not dry:
            exp.add_tags(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default=None, help='dir with job logs; tag the experiments found there')
    ap.add_argument('--project', default='knitwork-aistat')
    ap.add_argument('--workspace', default='team-rl-exp')
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()
    if args.logs:
        return tag_by_logs(args.logs, args.dry)

    from comet_ml import API
    api = API(api_key=os.environ['COMET_API_KEY'])
    n_tagged = n_skipped = 0
    for exp in api.get_experiments(args.workspace, project_name=args.project):
        name = exp.get_name() if hasattr(exp, 'get_name') else exp.name
        tags = tags_from_name(name)
        if tags is None:
            n_skipped += 1
            print(f'skip (unrecognised name): {name!r}')
            continue
        have = set(exp.get_tags())
        new = [t for t in tags if t not in have]
        print(f'{name!r}: +{len(new)} {new if args.dry else ""}')
        if new and not args.dry:
            exp.add_tags(new)
        n_tagged += 1
    print(f'done: {n_tagged} runs processed, {n_skipped} skipped')


if __name__ == '__main__':
    main()
