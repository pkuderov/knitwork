"""Delete a column's recurrent memory before a token, using cached native trajectories."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.communication_diagnostics import metrics, write_json
from knitwork.exps.communication.protocol import file_digest, make_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='docs/experiments/communication_diagnostics_2026-10-12_selected')
    args = parser.parse_args()
    output = ROOT / args.directory
    manifest = json.loads((output / 'manifest.json').read_text())
    settings = manifest['settings']
    torch.set_num_threads(1)
    rows = []
    for record in manifest['runs']:
        if not record['run_id'].startswith('text8/'):
            continue
        run = Path(record['checkpoint']['path']).parent
        cfg = yaml.safe_load((run / 'config.yaml').read_text())
        metadata = json.loads((run / 'metadata.json').read_text())
        if file_digest(record['checkpoint']['path']) != record['checkpoint']['sha256']:
            raise ValueError('Source checkpoint changed')
        directory = output / 'runs/text8' / cfg['condition']
        print(f'Text8 native memory lesions: {cfg["condition"]}', flush=True)
        model = make_model(cfg).eval().requires_grad_(False)
        model.load_state_dict(torch.load(record['checkpoint']['path'], map_location='cpu', weights_only=False)['model'])
        dataset = Path(cfg['dataset']['path'])
        n_val, n_test = cfg['dataset']['val_size'], cfg['dataset']['test_size']
        with dataset.open('rb') as handle:
            handle.seek(dataset.stat().st_size - n_val - n_test)
            raw = np.frombuffer(handle.read(n_val), dtype=np.uint8)
        charset = np.asarray(metadata['data_identity']['charset'])
        lookup = np.full(256, -1, dtype=np.int64)
        lookup[charset] = np.arange(len(charset))
        data = lookup[raw]
        starts = np.random.default_rng(settings['validation_seed']).integers(0, len(data) - settings['text_steps'] - 1, size=settings['streams'])
        with np.load(directory / 'features.npz') as cache:
            states = cache['validation_states'].reshape(-1, settings['streams'], 2, 4, model.hidden_size)
            targets = cache['validation_targets'].reshape(-1, settings['streams'])
        baseline, alternatives, scored = [], [], []
        indices = list(range(1, len(states), 4))
        for index in indices:
            t = settings['warmup_steps'] + index
            tokens = torch.from_numpy(data[starts + t]).view(1, -1)
            h = torch.from_numpy(states[index-1]).permute(1, 2, 0, 3).contiguous()
            y, actual_state, _ = model(tokens, {'h': h, 'out': h[-1]})
            expected = torch.from_numpy(states[index]).permute(1, 2, 0, 3)
            torch.testing.assert_close(actual_state['h'], expected, rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(targets[index], data[starts + t + 1])
            baseline.append(y.numpy())
            scored.append(targets[index])
            results = []
            for column in range(4):
                deleted = h.clone()
                deleted[:, column] = 0
                alternative, _, _ = model(tokens, {'h': deleted, 'out': deleted[-1]})
                results.append(alternative.numpy())
            alternatives.append(np.stack(results, axis=1))
        targets = np.concatenate(scored)
        baseline = np.concatenate(baseline)
        alternatives = np.concatenate(alternatives)
        original = metrics(baseline, targets)
        lesions = []
        for column in range(4):
            score = metrics(alternatives[:, column], targets)
            lesions.append({**score, 'delta_BPC': score['BPC'] - original['BPC']})
        if cfg['condition'] == 'none_column0':
            np.testing.assert_array_equal(alternatives[:, 1:], np.repeat(baseline[:, None], 3, axis=1))
        rows.append({'run_id': cfg['run_id'], 'condition': cfg['condition'], 'native': original, 'column_lesions': lesions,
            'checkpoint': record['checkpoint'], 'features_sha256': file_digest(directory / 'features.npz')})
        np.savez_compressed(directory / 'native_text_lesions.npz', targets=targets, baseline_logits=baseline, lesion_logits=alternatives, selected_timestep_indices=indices)
    write_json(output / 'text8_lesions.json', {'method': 'Zero both layers of one column immediately before a token; every fourth cached validation step after first scored step. No trajectory propagation, refit or backbone update.',
        'source': {'path': str(Path(__file__).resolve()), 'sha256': file_digest(__file__)}, 'rows': rows})
    print('Text8 native lesions saved', flush=True)


if __name__ == '__main__':
    main()
