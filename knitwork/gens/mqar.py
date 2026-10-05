"""Single-pass MQAR, following the pinned Zoology generator without global RNG changes."""

import numpy as np
import torch


SOURCE_COMMIT = '1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb'
SOURCE_URL = f'https://github.com/HazyResearch/zoology/blob/{SOURCE_COMMIT}/zoology/data/multiquery_ar.py'
IGNORE_INDEX = -100


def generate_mqar(
    *, vocab_size, num_examples, input_seq_len, num_kv_pairs, seed,
    power_a=0.01, random_non_queries=True,
):
    """Return CPU int64 inputs/labels (N,T); the value target aligns with the query key.

    Keys and values are independently sampled without replacement from disjoint
    halves of the vocabulary. Every stored pair is queried once, in an even slot
    of the suffix sampled without replacement with Zoology's power-law weights.
    Token zero is the filler when random_non_queries=False; otherwise filler is
    sampled over the whole vocabulary, including zero. Only designated queries
    are scored, even if a random filler happens to repeat a stored key.
    """
    for name, value in dict(
        vocab_size=vocab_size, num_examples=num_examples,
        input_seq_len=input_seq_len, num_kv_pairs=num_kv_pairs, seed=seed,
    ).items():
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f'{name} must be an integer')
    if num_examples < 1 or num_kv_pairs < 1:
        raise ValueError('num_examples and num_kv_pairs must be positive')
    if input_seq_len % 2 or input_seq_len < 4 * num_kv_pairs:
        raise ValueError('input_seq_len must be even and at least 4 * num_kv_pairs')
    if vocab_size <= input_seq_len:
        raise ValueError('vocab_size must exceed input_seq_len, as in Zoology')
    if not np.isfinite(power_a) or power_a <= 0:
        raise ValueError('power_a must be finite and positive')
    if not 0 <= seed < 2 ** 32:
        raise ValueError('seed must be in [0, 2**32)')
    rng = np.random.RandomState(seed)
    half = vocab_size // 2
    # Preserve the reference draw order across examples, without tiling vocabulary arrays.
    key_vocab, value_vocab = np.arange(1, half), np.arange(half, vocab_size)
    keys = np.stack([rng.choice(key_vocab, num_kv_pairs, replace=False) for _ in range(num_examples)])
    values = np.stack([rng.choice(value_vocab, num_kv_pairs, replace=False) for _ in range(num_examples)])
    context = 2 * num_kv_pairs
    slots = (input_seq_len - context) // 2
    weights = power_a * np.arange(1, slots + 1, dtype=np.float64) ** (power_a - 1)
    weights /= weights.sum()
    positions = context + 2 * np.stack([
        rng.choice(slots, num_kv_pairs, replace=False, p=weights) for _ in range(num_examples)
    ])
    inputs = torch.zeros((num_examples, input_seq_len), dtype=torch.long)
    labels = torch.full_like(inputs, IGNORE_INDEX)
    inputs[:, :context:2] = torch.from_numpy(keys)
    inputs[:, 1:context:2] = torch.from_numpy(values)
    rows = torch.arange(num_examples)[:, None]
    positions = torch.from_numpy(positions)
    inputs[rows, positions] = torch.from_numpy(keys)
    labels[rows, positions] = torch.from_numpy(values)
    if random_non_queries:
        generator = torch.Generator().manual_seed(seed)
        filler = torch.randint(vocab_size, inputs.shape, generator=generator)
        inputs = torch.where(inputs == 0, filler, inputs)
    return {'inputs': inputs, 'labels': labels}


def build_splits(config):
    """Fixed independent split/segment seeds, recorded with each generated segment."""
    seeds = [config[f'{split}_seed'] for split in ('train', 'val', 'test')]
    segments = [config[split] for split in ('train', 'val', 'test')]
    if any(not group for group in segments):
        raise ValueError('Every split needs at least one MQAR segment')
    # Reject accidental overlap of segment seed ranges across splits.
    all_seeds = [seed + index for seed, group in zip(seeds, segments) for index in range(len(group))]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError('MQAR split seed ranges must be disjoint')
    result = {}
    for split, seed, group in zip(('train', 'val', 'test'), seeds, segments):
        result[split] = []
        seen = set()
        for index, segment in enumerate(group):
            length, pairs = segment['input_seq_len'], segment['num_kv_pairs']
            key = f'T{length}_K{pairs}'
            if key in seen:
                raise ValueError(f'Duplicate {split} segment {key}')
            seen.add(key)
            data = generate_mqar(
                vocab_size=config['vocab_size'], seed=seed + index,
                power_a=config['power_a'], random_non_queries=config['random_non_queries'],
                **segment,
            )
            result[split].append(data | {'key': key, 'seed': seed + index, **segment})
    return result


def epoch_batches(segments, batch_size, rng):
    """Shuffle all examples once per epoch; batches have a common sequence length."""
    batches = []
    for index, segment in enumerate(segments):
        order = rng.permutation(len(segment['inputs']))
        batches.extend((index, order[start:start + batch_size]) for start in range(0, len(order), batch_size))
    rng.shuffle(batches)
    yield from batches
