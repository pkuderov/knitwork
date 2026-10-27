"""Inspect or explicitly execute the communication-control matrix locally."""

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys

import torch
import yaml

from knitwork.exps.communication.protocol import accounting, config_fingerprint, make_model, resolve_config, tensor_digest


def cases(matrix_path):
    matrix_path = Path(matrix_path)
    matrix = yaml.safe_load(matrix_path.read_text())
    for task, filename in matrix['tasks'].items():
        path = matrix_path.parent / filename
        base = yaml.safe_load(path.read_text())
        if base['exp'] != task:
            raise ValueError(f'Matrix/config task mismatch: {path}')
        for topology in matrix['topologies']:
            for recipe in matrix['recipes']:
                for seed in matrix['seeds']:
                    for condition in matrix['conditions']:
                        cfg = copy.deepcopy(base)
                        cfg.update(topology=topology, recipe=recipe, seed=seed, data_seed=seed, model=f'grnn_communication.{condition}')
                        yield path, cfg


def profile_dense_forward(model):
    model.eval()
    with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profile:
        model(torch.zeros(1, 1, dtype=torch.int64), model.init_state(1))
    return sum(event.flops for event in profile.key_averages() if event.key in ('aten::mm', 'aten::bmm', 'aten::addmm'))


def profile_dense_training(model):
    model.train()
    model.zero_grad(set_to_none=True)
    state = model.init_state(2)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profile:
        loss = 0
        for _ in range(3):
            out, state, _ = model(torch.zeros(1, 2, dtype=torch.int64), state)
            loss = loss + torch.nn.functional.cross_entropy(out, torch.zeros(2, dtype=torch.int64))
        loss.backward()
    flops = sum(event.flops for event in profile.key_averages() if event.key in ('aten::mm', 'aten::bmm', 'aten::addmm'))
    return flops // 6


def check_matrix(matrix_path, *, profile=False):
    rows = []
    groups = {}
    training_groups = {}
    for _, cfg in cases(matrix_path):
        resolved = resolve_config(cfg)
        with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(resolved['seed'])
            model = make_model(resolved)
        counts = accounting(model)
        low, high = resolved['parameter_range']
        if not low <= counts['parameters_nominal_trainable'] <= high:
            raise ValueError(f"Not a 1M model: {resolved['run_id']}")
        group = (resolved['exp'], resolved['topology'], resolved['recipe'], resolved['seed'])
        invariant = (
            counts['parameters_nominal_trainable'], counts['state_scalars'],
            counts['dense_forward_flops_per_token'], tensor_digest(model.state_dict()),
        )
        if group in groups and invariant != groups[group]:
            raise ValueError(f'Matching failed in {group}')
        groups[group] = invariant
        row = {'run_id': resolved['run_id'], 'effective_tokens': resolved['effective_tokens'], **counts}
        if profile:
            row['profiled_dense_forward_flops_per_token'] = profile_dense_forward(model)
            if row['profiled_dense_forward_flops_per_token'] != counts['dense_forward_flops_per_token']:
                raise ValueError(f"Dense FLOP formula/profile mismatch: {resolved['run_id']}")
            dense_training = profile_dense_training(model)
            row['profiled_dense_forward_backward_flops_per_token'] = dense_training
            if group in training_groups and training_groups[group] != dense_training:
                raise ValueError(f'Dense forward/backward cost differs in {group}')
            training_groups[group] = dense_training
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('matrix', nargs='?', default=str(Path(__file__).parent / 'config/matrix.yaml'))
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--logger', choices=['comet', 'local'], default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--task', choices=['text8', 'sdq'])
    parser.add_argument('--topology', choices=['L2C4', 'L3C4'])
    parser.add_argument('--condition', default=None)
    parser.add_argument('--report', default=None)
    args = parser.parse_args()
    if args.check or args.profile:
        torch.set_num_threads(1)
        report = check_matrix(args.matrix, profile=args.profile)
        for row in report:
            print(f"{row['run_id']}: params={row['parameters_nominal_trainable']:,}, state={row['state_scalars']:,}, dense={row['dense_forward_flops_per_token']:,}")
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
        if not args.execute:
            return
    count = 0
    matched = 0
    for path, cfg in cases(args.matrix):
        condition = cfg['model'].split('.', 1)[1]
        if args.task and cfg['exp'] != args.task:
            continue
        if args.topology and cfg['topology'] != args.topology:
            continue
        if args.condition and condition != args.condition:
            continue
        matched += 1
        command = [
            sys.executable, '-m', 'knitwork.exps.communication.run', str(path),
            f"--model={cfg['model']}", f"--topology={cfg['topology']}",
            f"--recipe={cfg['recipe']}", f"--seed={cfg['seed']}", f"--data_seed={cfg['data_seed']}", '--resume=auto',
        ]
        if args.logger is not None:
            command.append('--log.logger=None' if args.logger == 'local' else '--log.logger=comet')
        if args.device:
            cfg['device'] = args.device
            command.append(f'--device={args.device}')
        if args.output_dir:
            cfg['output_dir'] = args.output_dir
            command.append(f'--output_dir={args.output_dir}')
        resolved = resolve_config(cfg)
        summary_path = Path(cfg['output_dir']) / resolved['run_id'] / 'summary.json'
        if args.execute and summary_path.exists() and json.loads(summary_path.read_text())['completed']:
            metadata = json.loads((summary_path.parent / 'metadata.json').read_text())
            if metadata['config_fingerprint'] != config_fingerprint(resolved):
                raise ValueError(f'Completed run uses another configuration: {summary_path}; choose another output directory')
            print(f"Already completed: {resolved['run_id']}")
            continue
        print(shlex.join(command), flush=True)
        count += 1
        if args.execute:
            subprocess.run(command, check=True)
    if not matched:
        raise ValueError('No pending conditions match the requested filters')
    print(f'{count} commands; execute={args.execute}')


if __name__ == '__main__':
    main()
