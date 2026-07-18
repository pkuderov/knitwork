from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from knitwork.common.utils import CE_ignore_index


class TextGenerator(torch.nn.Module):
    """
    Vectorized text-stream generator over one flat array.

    Semantics:
    - The dataset is treated as one long cyclic stream.
    - We maintain n_envs parallel cursors into that stream.
    - Each call to next() emits:
        tokens[i]  = data[pos[i]]
        targets[i] = data[pos[i] + 1]   (with wraparound)
    - Then all cursors advance by 1.
    - reset_mask[i] is True for envs wrapped the dataset, i.e. non-contiguous transition
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        n_envs: int, ignore_index: int, 
        device, seed: int = None
    ):
        super().__init__()
        data = np.asarray(data)
        assert data.ndim == 1, f"Expected flat 1D token array, got shape={data.shape}"
        assert n_envs >= 1

        self.device = device
        self.ignore_index = ignore_index

        self.rng = torch.Generator(device)
        if seed is not None:
            self.rng.manual_seed(seed)

        data = torch.from_numpy(data)
        self.register_buffer('data', data)
        self.data_len = len(self.data)
        self.n_envs = n_envs

        pos = torch.arange(n_envs, dtype=torch.int64) * self.data_len // n_envs
        self.register_buffer('pos', pos)

    def next(self):
        tokens = self.data[self.pos].clone()

        self.pos += 1
        wrap_mask = self.pos >= self.data_len
        self.pos[wrap_mask] = 0

        targets = self.data[self.pos].clone()
        targets[wrap_mask] = self.ignore_index

        return {
            "tokens": tokens,
            "targets": targets,
            "reset_mask": wrap_mask,
        }

    def next_rollout(self, rollout: int):
        result = [self.next() for _ in range(rollout)]
        keys = list(result[0].keys())
        return {
            k: np.stack([r[k] for r in result]) 
            for k in keys
        }
    
    def get_stats(self):
        return {}


def split_train_test(data: np.ndarray, train_frac: int | float = 0.95):
    cut = len(data) * train_frac if train_frac <= 1.0 else train_frac
    cut = min(int(cut), len(data) - 1)

    train_data = data[:cut]
    eval_data = data[cut:]
    return train_data, eval_data


def tokenize(data):
    """
    Tokenize passed sequence by translating each character to a token — an index 
    of the character in the ordered set of all unique characters [of the sequence].
    """
    chars = np.unique(data)
    # mapping m[char] -> token stored in the dense array, auxiliary for the vectorized translation
    char_to_token_arr = np.full(chars.max() + 1, -1, dtype=int)
    char_to_token_arr[chars] = np.arange(len(chars))

    tokenized_data = char_to_token_arr[data].copy()
    return tokenized_data, chars


def load_dataset(path: str | Path, dtype=np.uint8):
    path = os.path.expandvars(str(path))
    path = os.path.expanduser(path)
    path = os.path.realpath(path)
    return np.fromfile(path, dtype=dtype)


def main():
    data = load_dataset(Path("~/data/text/text8.txt").expanduser())
    tokenized_data, chars = tokenize(data)
    print(len(data))
    print(len(chars))
    print(chars)
    print(tokenized_data.shape)
    print(tokenized_data[:10])
    print(chars[tokenized_data[:10]].tobytes().decode('utf-8'))

    gen = TextGenerator(tokenized_data, n_envs=5, seed=42, ignore_index=CE_ignore_index)
    for _ in range(5):
        out = gen.next()
        print("tokens     ", out["tokens"], "    ", chars[out["tokens"]].tobytes().decode('utf-8'))
        print("targets    ", out["targets"], "    ", chars[out["targets"]].tobytes().decode('utf-8'))
        print("reset_mask ", out["reset_mask"])
        print()


if __name__ == "__main__":
    main()
