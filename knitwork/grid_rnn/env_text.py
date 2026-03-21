from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import numpy as np

from knitwork.common.scheduler import Scheduler
from knitwork.common.utils import FpsCounter, format_readable_num

@dataclass
class TextEnvConfig:
    # Masking
    # NB: common default for torch.nn.CrossEntropyLoss(ignore_index=...)
    ignore_index: int = -100


@dataclass
class TextEnvStats:
    def tick(self):
        return False

    def decay(self, lr: float = 0.01):
        if not self.tick():
            return


class TextGenerator:
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
        n_envs: int,
        seed: int = None,
        cfg: TextEnvConfig | None = None,
    ):
        data = np.asarray(data)
        assert data.ndim == 1, f"Expected flat 1D token array, got shape={data.shape}"
        assert n_envs >= 1

        self.cfg = cfg or TextEnvConfig()
        self.rng = np.random.default_rng(seed)

        self.data = data
        self.data_len = len(self.data)
        self.n_envs = n_envs

        self.pos = np.linspace(0, self.data_len, num=n_envs, endpoint=False, dtype=int)

    def next(self):
        tokens = self.data[self.pos]

        self.pos += 1
        wrap_mask = self.pos >= self.data_len
        self.pos[wrap_mask] = 0

        targets = self.data[self.pos]
        targets[wrap_mask] = self.cfg.ignore_index

        return {
            "tokens": tokens.copy(),
            "targets": targets.copy(),
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
    if isinstance(path, str):
        path = Path(path)
    path = path.expanduser().as_posix()
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

    gen = TextGenerator(tokenized_data, n_envs=5)
    for _ in range(5):
        out = gen.next()
        print("tokens     ", out["tokens"], "    ", chars[out["tokens"]].tobytes().decode('utf-8'))
        print("targets    ", out["targets"], "    ", chars[out["targets"]].tobytes().decode('utf-8'))
        print("reset_mask ", out["reset_mask"])
        print()


if __name__ == "__main__":
    main()
