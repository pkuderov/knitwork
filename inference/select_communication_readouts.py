"""Calibrate small readouts on held-out training streams; reuse frozen features."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.communication_diagnostics import READOUTS, metrics, readout_features, write_json
from inference.communication_diagnostics_report import main as render
from knitwork.common.utils import CE_ignore_index
from knitwork.exps.communication.protocol import file_digest, make_model
from knitwork.gens.sdq import StoreDistractQueryGenerator


def stream_groups(cfg, settings, targets):
    if cfg['exp'] == 'text8':
        groups = np.tile(np.arange(settings['streams']), settings['text_steps'] - settings['warmup_steps'])
    else:
        gen = StoreDistractQueryGenerator(**cfg['eval_generator'], n_envs=settings['streams'], seed=settings['train_seed'], ignore_index=CE_ignore_index)
        pieces, expected = [], []
        for step in range(settings['sdq_steps']):
            obs = gen.next()
            if step >= settings['warmup_steps']:
                indices = np.flatnonzero(obs['targets'] != CE_ignore_index)[obs['sq_gaps'] >= 0]
                pieces.append(indices)
                expected.append(obs['targets'][indices])
        groups = np.concatenate(pieces)
        np.testing.assert_array_equal(np.concatenate(expected), targets)
    if len(groups) != len(targets):
        raise ValueError('Stream groups differ from cached samples')
    return groups


def initial_weights(model, label):
    weight, bias = model.head.weight.detach(), model.head.bias.detach()
    if label == 'Concat':
        if model.rnn.readout == 'mean':
            weight = weight.repeat(1, 4) / 4
        else:
            weight = torch.cat((weight, torch.zeros(model.output_size, 3 * model.hidden_size)), dim=1)
    return {'weight': weight, 'bias': bias}


def fit_selected(train, targets, validation, validation_targets, groups, classes, settings, initial=None):
    train = torch.as_tensor(train, dtype=torch.float32)
    targets = torch.as_tensor(targets, dtype=torch.long)
    validation = torch.as_tensor(validation, dtype=torch.float32)
    calibration = torch.as_tensor(groups >= settings['streams'] * 3 // 4)
    fitting = ~calibration
    if not fitting.any() or not calibration.any():
        raise ValueError('Require separate fitting/calibration streams')
    mean = train[fitting].mean(0)
    scale = train[fitting].std(0, unbiased=False).clamp_min(1e-4)
    fit_x, select_x = (train[fitting] - mean) / scale, (train[calibration] - mean) / scale
    fit_y, select_y = targets[fitting], targets[calibration]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings['probe_seed'])
        head = nn.Linear(train.shape[1], classes)
        with torch.no_grad():
            if initial is None:
                head.weight.zero_()
                head.bias.zero_()
            else:
                head.weight.copy_(initial['weight'] * scale)
                head.bias.copy_(initial['bias'] + initial['weight'] @ mean)
        origin = copy.deepcopy(head.state_dict())
        with torch.no_grad():
            best = {'loss': F.cross_entropy(head(select_x), select_y).item(), 'update': 0, 'penalty': None, 'weights': copy.deepcopy(origin)}
        trace = []
        for penalty in settings['probe_penalties']:
            head.load_state_dict(origin)
            optimizer = torch.optim.AdamW(head.parameters(), lr=settings['probe_lr'], weight_decay=settings['probe_weight_decay'])
            generator = torch.Generator().manual_seed(settings['probe_seed'])
            for step in range(settings['probe_updates']):
                indices = torch.randint(len(fit_x), (settings['probe_batch'],), generator=generator)
                ce = F.cross_entropy(head(fit_x[indices]), fit_y[indices])
                loss = ce + penalty * (head.weight - origin['weight']).square().sum() / classes
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if (step + 1) % 50 == 0 or step + 1 == settings['probe_updates']:
                    with torch.no_grad():
                        score = F.cross_entropy(head(select_x), select_y).item()
                    trace.append({'update': step + 1, 'penalty': penalty, 'training_batch_Loss': ce.item(), 'calibration_Loss': score})
                    if score < best['loss']:
                        best = {'loss': score, 'update': step + 1, 'penalty': penalty, 'weights': copy.deepcopy(head.state_dict())}
        head.load_state_dict(best['weights'])
        with torch.no_grad():
            weight = head.weight / scale
            bias = head.bias - weight @ mean
            logits = validation @ weight.T + bias
            info = {
                'validation': metrics(logits, validation_targets), 'training': metrics(head(fit_x), fit_y),
                'calibration': metrics(head(select_x), select_y), 'trace': trace,
                'selection': {'update': best['update'], 'penalty': best['penalty'], 'fitting_samples': int(fitting.sum()), 'calibration_samples': int(calibration.sum())},
            }
    return info, {'weight': weight.detach(), 'bias': bias.detach(), 'mean': mean, 'scale': scale}, logits.argmax(-1).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='docs/experiments/communication_diagnostics_2026-10-12')
    parser.add_argument('--output', default='docs/experiments/communication_diagnostics_2026-10-12_selected')
    args = parser.parse_args()
    source, output = ROOT / args.source, ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((source / 'manifest.json').read_text())
    settings = dict(original['settings'], output=str(output), probe_penalties=[0.0, 0.01, 0.1], probe_selection='last quarter of probe-training streams', probe_initialization='native-head-compatible for task targets, zero for stored value')
    manifest = dict(original, settings=settings, feature_source_manifest={'path': str(source / 'manifest.json'), 'sha256': file_digest(source / 'manifest.json')})
    manifest['analysis_source_hashes'] = {name: file_digest(ROOT / name) for name in (
        'inference/select_communication_readouts.py', 'inference/communication_diagnostics.py',
        'inference/communication_diagnostics_report.py', 'inference/report_plot_style.py',
    )}
    if (output / 'manifest.json').exists() and json.loads((output / 'manifest.json').read_text()) != manifest:
        raise ValueError('Selection protocol changed; choose another output directory')
    write_json(output / 'manifest.json', manifest)
    torch.set_num_threads(1)
    for record in original['runs']:
        task, _, _, condition, _ = record['run_id'].split('/')
        old = source / 'runs' / task / condition
        directory = output / 'runs' / task / condition
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / 'result.json').exists():
            continue
        print(f'Select readouts: {task}/{condition}', flush=True)
        cfg = yaml.safe_load((Path(record['checkpoint']['path']).parent / 'config.yaml').read_text())
        if file_digest(record['checkpoint']['path']) != record['checkpoint']['sha256']:
            raise ValueError('Source checkpoint changed')
        model = make_model(cfg).eval().requires_grad_(False)
        model.load_state_dict(torch.load(record['checkpoint']['path'], map_location='cpu', weights_only=False)['model'])
        result = json.loads((old / 'result.json').read_text())
        result['feature_source'] = {'path': str(old / 'features.npz'), 'sha256': file_digest(old / 'features.npz')}
        result['probes'] = {}
        with np.load(old / 'features.npz') as data:
            groups = stream_groups(cfg, settings, data['train_targets'])
            fields = ['targets'] if task == 'text8' else ['targets', 'stored_values']
            for field in fields:
                scores = {}
                for label in READOUTS:
                    train = readout_features(data['train_top'], label)
                    val = readout_features(data['validation_states'][:, -1], label)
                    info, head, predicted = fit_selected(train, data[f'train_{field}'], val, data[f'validation_{field}'], groups, model.output_size, settings, initial_weights(model, label) if field == 'targets' else None)
                    scores[label] = info
                    torch.save({'source_checkpoint': record['checkpoint'], 'settings': settings, 'target': field, 'readout': label, **head}, directory / f'probe-{field}-{label}.pt')
                    if label == 'Concat' and field == 'targets':
                        weight = head['weight'].reshape(model.output_size, 4, model.hidden_size)
                        norms = ((weight - weight.mean(0, keepdim=True)) * head['scale'].reshape(4, model.hidden_size)).square().sum(dim=(0, 2)).sqrt()
                        result['concat_standardized_weight_norm'] = norms.tolist()
                        result['concat_weight_share'] = (norms / norms.sum()).tolist()
                        result['concat_column_replacement'] = []
                        for column in range(4):
                            altered = val.copy().reshape(-1, 4, model.hidden_size)
                            altered[:, column] = head['mean'].reshape(4, model.hidden_size)[column].numpy()
                            score = metrics(torch.as_tensor(altered.reshape(len(altered), -1)) @ head['weight'].T + head['bias'], data['validation_targets'])
                            result['concat_column_replacement'].append({**score, 'delta_BPC': score['BPC'] - info['validation']['BPC'], 'delta_Acc': score['Acc'] - info['validation']['Acc']})
                    if task == 'sdq':
                        info['gaps'] = []
                        edges = [0, 8, 32, 128, 256, 512, np.inf]
                        for lo, hi in zip(edges[:-1], edges[1:]):
                            mask = (data['validation_gaps'] >= lo) & (data['validation_gaps'] < hi)
                            info['gaps'].append({'lo': lo, 'hi': hi if np.isfinite(hi) else None, 'count': int(mask.sum()), 'Acc': float((predicted[mask] == data[f'validation_{field}'][mask]).mean()) if mask.any() else None})
                result['probes'][field] = scores
        feature_link = directory / 'features.npz'
        if not feature_link.exists():
            feature_link.symlink_to((old / 'features.npz').resolve())
        write_json(directory / 'result.json', result)
        print(f'Done: {task}/{condition}', flush=True)
    render(output)


if __name__ == '__main__':
    main()
