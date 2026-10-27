"""Frozen-checkpoint column similarity, memory probes and readout diagnostics."""

import argparse
import importlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knitwork.common.utils import CE_ignore_index
from knitwork.exps.communication.protocol import config_fingerprint, file_digest, make_model
from knitwork.gens.sdq import StoreDistractQueryGenerator
from knitwork.gens.text import load_dataset, split_train_val_test, tokenize_splits


DEFAULTS = {
    'root': 'runs/communication_1m',
    'output': 'docs/experiments/communication_diagnostics_2026-10-12',
    'device': 'cpu', 'threads': 1, 'streams': 16, 'warmup_steps': 128,
    'text_steps': 1024, 'sdq_steps': 2048,
    'train_seed': 2026101201, 'validation_seed': 2026101202,
    'probe_seed': 2026101203, 'probe_updates': 400, 'probe_batch': 512,
    'probe_lr': 0.01, 'probe_weight_decay': 0.0001,
}
READOUTS = ['C0', 'C1', 'C2', 'C3', 'Mean', 'Concat']


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def linear_cka(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError('CKA requires aligned samples and at least two observations')
    x = x - x.mean(0)
    y = y - y.mean(0)
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(np.linalg.norm(x.T @ y)**2 / denominator) if denominator > 0 else None


def cka_matrices(states):
    # Observations, layers, columns, features. Each pair uses identical observations.
    return [[[linear_cka(layer[:, i], layer[:, j]) for j in range(layer.shape[1])]
        for i in range(layer.shape[1])] for layer in states.transpose(1, 0, 2, 3)]


def readout_features(top, label):
    if label.startswith('C') and label != 'Concat':
        return top[:, int(label[1:])]
    if label == 'Mean':
        return top.mean(1)
    if label == 'Concat':
        return top.reshape(len(top), -1)
    raise ValueError(label)


def metrics(logits, targets):
    targets = torch.as_tensor(targets, dtype=torch.long)
    logits = torch.as_tensor(logits, dtype=torch.float32)
    loss = F.cross_entropy(logits, targets).item()
    return {'Loss': loss, 'BPC': loss / math.log(2), 'Acc': (logits.argmax(-1) == targets).float().mean().item(), 'count': len(targets)}


def fit_probe(train, targets, validation, validation_targets, classes, settings):
    train = torch.as_tensor(train, dtype=torch.float32)
    validation = torch.as_tensor(validation, dtype=torch.float32)
    targets = torch.as_tensor(targets, dtype=torch.long)
    mean = train.mean(0)
    scale = train.std(0, unbiased=False).clamp_min(1e-4)
    x = (train - mean) / scale
    vx = (validation - mean) / scale
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings['probe_seed'])
        head = nn.Linear(x.shape[1], classes)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        optimizer = torch.optim.AdamW(head.parameters(), lr=settings['probe_lr'], weight_decay=settings['probe_weight_decay'])
        generator = torch.Generator().manual_seed(settings['probe_seed'])
        trace = []
        for step in range(settings['probe_updates']):
            batch = torch.randint(len(x), (settings['probe_batch'],), generator=generator)
            loss = F.cross_entropy(head(x[batch]), targets[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if (step + 1) % 50 == 0:
                trace.append({'update': step + 1, 'training_batch_Loss': loss.item()})
        with torch.no_grad():
            logits = head(vx)
            weight = head.weight / scale
            bias = head.bias - weight @ mean
    return {
        'validation': metrics(logits, validation_targets),
        'training': metrics(head(x).detach(), targets), 'trace': trace,
    }, {'weight': weight.detach(), 'bias': bias.detach(), 'mean': mean, 'scale': scale}, logits.argmax(-1).numpy()


@torch.no_grad()
def collect_text(model, data, settings, seed):
    rng = np.random.default_rng(seed)
    steps = settings['text_steps']
    starts = rng.integers(0, len(data) - steps - 1, size=settings['streams'])
    state = model.rnn.init_state(settings['streams'])
    states, targets, logits = [], [], []
    for t in range(steps):
        tokens = torch.as_tensor(data[starts + t], dtype=torch.long).view(1, -1)
        output, state, _ = model(tokens, state)
        if t >= settings['warmup_steps']:
            states.append(state['h'].permute(2, 0, 1, 3).numpy().copy())
            targets.append(data[starts + t + 1].copy())
            logits.append(output.numpy().copy())
    return {'states': np.concatenate(states), 'targets': np.concatenate(targets), 'logits': np.concatenate(logits), 'starts': starts}


@torch.no_grad()
def collect_sdq(model, cfg, settings, seed, *, lesions=False):
    gen = StoreDistractQueryGenerator(**cfg['eval_generator'], n_envs=settings['streams'], seed=seed, ignore_index=CE_ignore_index)
    state = model.rnn.init_state(settings['streams'])
    states, targets, values, gaps, logits, lesion_logits = [], [], [], [], [], []
    for t in range(settings['sdq_steps']):
        obs = gen.next()
        state = model.rnn.reset_state(state, torch.from_numpy(obs['reset_mask']))
        previous = state
        output, state, _ = model(torch.from_numpy(obs['tokens']).view(1, -1), state)
        if t < settings['warmup_steps']:
            continue
        query_ids = np.flatnonzero(obs['targets'] != CE_ignore_index)
        query_gaps = obs['sq_gaps']
        # Raw stored-value probes require an actual preceding store.
        query_ids = query_ids[query_gaps >= 0]
        query_gaps = query_gaps[query_gaps >= 0]
        if not len(query_ids):
            continue
        keys = obs['tokens'][query_ids] - gen.ix_query
        states.append(state['h'][:, :, query_ids].permute(2, 0, 1, 3).numpy().copy())
        targets.append(obs['targets'][query_ids].copy())
        values.append(gen.stored[query_ids, keys].copy())
        gaps.append(query_gaps.copy())
        logits.append(output[query_ids].numpy().copy())
        if lesions:
            alternatives = []
            for column in range(model.rnn.n_columns):
                h = previous['h'][:, :, query_ids].clone()
                h[:, column] = 0
                altered = {'h': h, 'out': h[-1]}
                y, _, _ = model(torch.from_numpy(obs['tokens'][query_ids]).view(1, -1), altered)
                alternatives.append(y.numpy().copy())
            lesion_logits.append(np.stack(alternatives, axis=1))
    result = {
        'states': np.concatenate(states), 'targets': np.concatenate(targets),
        'stored_values': np.concatenate(values), 'gaps': np.concatenate(gaps), 'logits': np.concatenate(logits),
    }
    if lesions:
        result['lesion_logits'] = np.concatenate(lesion_logits)
    return result


def analyze_run(record, model, text_data, settings, output):
    cfg = record['config']
    collect = collect_text if cfg['exp'] == 'text8' else collect_sdq
    if cfg['exp'] == 'text8':
        train = collect(model, text_data[0], settings, settings['train_seed'])
        val = collect(model, text_data[1], settings, settings['validation_seed'])
        target_fields = ['targets']
    else:
        train = collect(model, cfg, settings, settings['train_seed'])
        val = collect(model, cfg, settings, settings['validation_seed'], lesions=True)
        target_fields = ['targets', 'stored_values']
    result = {
        'run_id': cfg['run_id'], 'task': cfg['exp'], 'condition': cfg['condition'],
        'checkpoint': record['checkpoint'], 'step': record['step'],
        'training_probe_samples': len(train['targets']), 'validation_probe_samples': len(val['targets']),
        'cka': cka_matrices(val['states']), 'native_sample': metrics(val['logits'], val['targets']),
        'probes': {},
    }
    np.savez_compressed(output / 'features.npz',
        train_top=train['states'][:, -1], validation_states=val['states'],
        train_targets=train['targets'], validation_targets=val['targets'],
        **({'train_stored_values': train['stored_values'], 'validation_stored_values': val['stored_values'], 'validation_gaps': val['gaps']} if cfg['exp'] == 'sdq' else {}),
    )
    for target_field in target_fields:
        probes = {}
        for label in READOUTS:
            features = readout_features(train['states'][:, -1], label)
            validation = readout_features(val['states'][:, -1], label)
            scores, head, predictions = fit_probe(features, train[target_field], validation, val[target_field], model.output_size, settings)
            probes[label] = scores
            torch.save({'run_id': cfg['run_id'], 'source_checkpoint': record['checkpoint'], 'target': target_field, 'readout': label, 'settings': settings, **head}, output / f'probe-{target_field}-{label}.pt')
            if label == 'Concat' and target_field == 'targets':
                weights = head['weight'].reshape(model.output_size, 4, model.hidden_size)
                centered = weights - weights.mean(0, keepdim=True)
                scale = head['scale'].reshape(4, model.hidden_size)
                norms = (centered * scale).square().sum(dim=(0, 2)).sqrt()
                result['concat_standardized_weight_norm'] = norms.tolist()
                result['concat_weight_share'] = (norms / norms.sum()).tolist()
                baseline = scores['validation']
                ablations = []
                for column in range(4):
                    altered = validation.copy().reshape(-1, 4, model.hidden_size)
                    altered[:, column] = head['mean'].reshape(4, model.hidden_size)[column].numpy()
                    altered_logits = torch.as_tensor(altered.reshape(len(altered), -1)) @ head['weight'].T + head['bias']
                    score = metrics(altered_logits, val['targets'])
                    ablations.append({**score, 'delta_BPC': score['BPC'] - baseline['BPC'], 'delta_Acc': score['Acc'] - baseline['Acc']})
                result['concat_column_replacement'] = ablations
            if cfg['exp'] == 'sdq':
                edges = [0, 8, 32, 128, 256, 512, float('inf')]
                scores['gaps'] = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    mask = (val['gaps'] >= lo) & (val['gaps'] < hi)
                    scores['gaps'].append({'lo': lo, 'hi': hi if math.isfinite(hi) else None, 'count': int(mask.sum()), 'Acc': float((predictions[mask] == val[target_field][mask]).mean()) if mask.any() else None})
        result['probes'][target_field] = probes
    if cfg['exp'] == 'sdq':
        result['query_state_lesions'] = []
        for column in range(4):
            score = metrics(val['lesion_logits'][:, column], val['targets'])
            result['query_state_lesions'].append({**score, 'delta_Acc': score['Acc'] - result['native_sample']['Acc']})
    return result


def read_records(root):
    records = []
    for path in sorted(root.glob('*/L2C4/clean/*/seed42-data42/summary.json')):
        summary = json.loads(path.read_text())
        if not summary['completed']:
            continue
        cfg = yaml.safe_load((path.parent / 'config.yaml').read_text())
        metadata = json.loads((path.parent / 'metadata.json').read_text())
        if metadata['config_fingerprint'] != config_fingerprint(cfg):
            raise ValueError(f'Config changed: {path}')
        for name, expected in metadata['source_hashes'].items():
            if file_digest(ROOT / name) != expected:
                raise ValueError(f'Source changed: {name}')
        checkpoint = path.parent / 'final.pt'
        records.append({
            'config': cfg, 'checkpoint': {'path': str(checkpoint.resolve()), 'sha256': file_digest(checkpoint)},
            'step': summary['tokens'], 'native_source_hashes': metadata['source_hashes'],
            'config_fingerprint': metadata['config_fingerprint'], 'data_identity': metadata['data_identity'],
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULTS['output'])
    parser.add_argument('--root', default=DEFAULTS['root'])
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    settings = dict(DEFAULTS, output=args.output, root=args.root)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(settings['threads'])
    if not args.report_only:
        records = read_records(ROOT / args.root)
        manifest = {
            'settings': settings,
            'analysis_source_hashes': {name: file_digest(ROOT / name) for name in (
                'inference/communication_diagnostics.py', 'inference/communication_diagnostics_report.py',
                'inference/report_plot_style.py',
            )},
            'runs': [{k: v for k, v in record.items() if k != 'config'} | {'run_id': record['config']['run_id']} for record in records],
        }
        manifest_path = output / 'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Diagnostic input set/settings changed; choose a new output directory')
        write_json(manifest_path, manifest)
        text_record = next(record for record in records if record['config']['exp'] == 'text8')
        dataset_cfg = text_record['config']
        dataset = Path(dataset_cfg['dataset']['path'])
        expected_data = text_record['data_identity']
        if 'sha256' in expected_data and file_digest(dataset) != expected_data['sha256']:
            raise ValueError('Text dataset changed')
        raw = load_dataset(dataset)
        splits = split_train_val_test(raw, dataset_cfg['dataset']['val_size'], dataset_cfg['dataset']['test_size'])
        text_data, charset = tokenize_splits(*splits)
        del raw, splits
        for index, record in enumerate(records):
            cfg = record['config']
            directory = output / 'runs' / cfg['exp'] / cfg['condition']
            directory.mkdir(parents=True, exist_ok=True)
            result_path = directory / 'result.json'
            if result_path.exists():
                print(f'Skip complete diagnostic {cfg["run_id"]}', flush=True)
                continue
            print(f'[{index+1}/{len(records)}] Analyze {cfg["run_id"]}', flush=True)
            started = time.monotonic()
            with torch.random.fork_rng(devices=[]):
                model = make_model(cfg, vocab=len(charset) if cfg['exp'] == 'text8' else None)
            checkpoint = torch.load(record['checkpoint']['path'], map_location='cpu', weights_only=False)
            if checkpoint['step'] != record['step']:
                raise ValueError('Final checkpoint is not the declared final training step')
            model.load_state_dict(checkpoint['model'])
            model.eval().requires_grad_(False)
            result = analyze_run(record, model, text_data, settings, directory)
            result['analysis_seconds'] = time.monotonic() - started
            write_json(result_path, result)
            print(f'Done {cfg["run_id"]} in {result["analysis_seconds"]:.1f}s; native sample {result["native_sample"]}', flush=True)
        print('All checkpoint diagnostics collected', flush=True)
    report = importlib.import_module('inference.communication_diagnostics_report')
    report.main(output)


if __name__ == '__main__':
    main()
