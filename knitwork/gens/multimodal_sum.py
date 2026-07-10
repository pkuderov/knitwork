from __future__ import annotations

from pathlib import Path

import numpy as np

from knitwork.common.tracker import Tracker
from knitwork.common.utils import safe_div

DEFAULT_CACHE_DIR = Path(__file__).parent / "data" / "mdsum_cache"


class _ModalityBank:
    """Per-digit O(1) sampling over a cached feature bank (one modality)."""

    def __init__(self, npz_path: Path, *, split: str, rng: np.random.Generator):
        data = np.load(npz_path)
        idx = data[f"split_{split}_idx"]
        self.features = data["features"][idx]
        self.labels = data["labels"][idx]
        self.dim = self.features.shape[1]
        self.rng = rng
        self.per_digit_idx = [
            np.flatnonzero(self.labels == d) for d in range(10)
        ]
        for d, ixs in enumerate(self.per_digit_idx):
            assert len(ixs) > 0, f"no samples found for digit {d} in {npz_path}"

    def sample(self, digits: np.ndarray) -> np.ndarray:
        # digits: (n,) int in 0..9 -> (n, dim) feature vectors
        out = np.empty((len(digits), self.dim), dtype=np.float32)
        for d in range(10):
            mask = digits == d
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            ixs = self.per_digit_idx[d][self.rng.integers(0, len(self.per_digit_idx[d]), size=cnt)]
            out[mask] = self.features[ixs]
        return out


class MultimodalDigitSumGenerator:
    """
    Multimodal Digit-Sum (MDS) benchmark generator.

    Two "signal" columns receive real digit-bearing modalities (MNIST image,
    FSDD spoken digit), arriving sparsely and independently over time.
    `n_buffer_columns` extra columns carry pure Gaussian noise, never digit-bearing.
    On query steps (p_query), the target is the running sum of all digits that
    arrived (across both signal columns) since the last reset; other steps are
    `ignore_index` in the loss, exactly like SDQ.
    """

    def __init__(
            self, *,
            n_envs: int, seed: int, ignore_index: int,
            T: float,
            n_buffer_columns: int = 2,
            p_arrive: float = 0.25,
            p_query: float = 0.15,
            buffer_noise_std: float = 1.0,
            max_events_per_episode: int = 4,
            split: str = "train",
            cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ):
        self.n_envs = n_envs
        self.ignore_index = ignore_index
        self.T = T
        self.n_buffer_columns = n_buffer_columns
        self.p_arrive = p_arrive
        self.p_query = p_query
        self.buffer_noise_std = buffer_noise_std
        self.max_events_per_episode = max_events_per_episode

        self.rng = np.random.default_rng(seed)
        cache_dir = Path(cache_dir)
        self.image_bank = _ModalityBank(cache_dir / "mnist_features.npz", split=split, rng=self.rng)
        self.audio_bank = _ModalityBank(cache_dir / "fsdd_features.npz", split=split, rng=self.rng)
        self.image_dim = self.image_bank.dim
        self.audio_dim = self.audio_bank.dim
        self.buffer_dim = max(self.image_dim, self.audio_dim)

        # 0 .. 9 * max_events_per_episode inclusive
        self.n_sum_classes = 9 * self.max_events_per_episode + 1

        self.sum_accum = np.zeros(n_envs, dtype=np.int64)
        self.n_events = np.zeros(n_envs, dtype=np.int64)
        self.n_steps = np.zeros(n_envs, dtype=np.int64)

        lr_stats = 3e-4
        self.stats = Tracker(lr=lr_stats)

    @property
    def p_term(self):
        return 1.0 / self.T

    def reset(self, ixs):
        if len(ixs) == 0:
            return
        self.stats.put({"episodes": len(ixs), "ep_lens": self.n_steps[ixs].sum()}, inc_step=False)
        self.sum_accum[ixs] = 0
        self.n_events[ixs] = 0
        self.n_steps[ixs] = 0

    def _sample_arrivals(self, can_arrive: np.ndarray, bank: _ModalityBank):
        n_envs = self.n_envs
        arrive_mask = (self.rng.random(n_envs) < self.p_arrive) & can_arrive
        feat = np.zeros((n_envs, bank.dim), dtype=np.float32)
        digit = np.full(n_envs, -1, dtype=np.int64)

        ixs = np.flatnonzero(arrive_mask)
        if len(ixs) > 0:
            d = self.rng.integers(0, 10, size=len(ixs))
            feat[ixs] = bank.sample(d)
            digit[ixs] = d
        return feat, digit, arrive_mask

    def next(self) -> dict:
        n_envs = self.n_envs

        reset_mask = self.rng.random(n_envs) < self.p_term
        self.reset(np.flatnonzero(reset_mask))

        # sequential capping: audio's arrival check sees image's just-applied count,
        # so n_events (and therefore sum_accum) never exceeds max_events_per_episode
        can_arrive_image = self.n_events < self.max_events_per_episode
        image_feat, digit_image, arrived_image = self._sample_arrivals(can_arrive_image, self.image_bank)
        self.sum_accum += np.where(digit_image >= 0, digit_image, 0)
        self.n_events += arrived_image.astype(np.int64)

        can_arrive_audio = self.n_events < self.max_events_per_episode
        audio_feat, digit_audio, arrived_audio = self._sample_arrivals(can_arrive_audio, self.audio_bank)
        self.sum_accum += np.where(digit_audio >= 0, digit_audio, 0)
        self.n_events += arrived_audio.astype(np.int64)

        buffer_feat = self.rng.normal(
            0.0, self.buffer_noise_std,
            size=(n_envs, self.n_buffer_columns, self.buffer_dim)
        ).astype(np.float32)

        query_mask = self.rng.random(n_envs) < self.p_query
        target = np.where(query_mask, self.sum_accum, self.ignore_index).astype(np.int64)

        self.n_steps += 1
        self.stats.put({
            "steps": n_envs,
            "arrivals": int(arrived_image.sum() + arrived_audio.sum()),
            "queries": int(query_mask.sum()),
        })

        return {
            "image_feat": image_feat,
            "audio_feat": audio_feat,
            "buffer_feat": buffer_feat,
            "query_mask": query_mask,
            "reset_mask": reset_mask,
            "target": target,
            "arrived_digit_image": digit_image,
            "arrived_digit_audio": digit_audio,
            # number of digits accumulated so far this episode (incl. this step) —
            # used to bucket query accuracy by task difficulty
            "n_events": self.n_events.copy(),
            # ground-truth running sum, always valid (unlike `target`, not gated
            # by query_mask) — used to probe where in the network the sum is
            # linearly represented, regardless of whether a query fired
            "sum_now": self.sum_accum.copy(),
        }

    def next_rollout(self, rollout: int) -> dict:
        result = [self.next() for _ in range(rollout)]
        keys = list(result[0].keys())
        return {k: np.stack([r[k] for r in result]) for k in keys}

    def set_metaparams(self, T=None, p_arrive=None, p_query=None, buffer_noise_std=None, n_buffer_columns=None):
        if T is not None:
            self.T = T
        if p_arrive is not None:
            self.p_arrive = p_arrive
        if p_query is not None:
            self.p_query = p_query
        if buffer_noise_std is not None:
            self.buffer_noise_std = buffer_noise_std
        if n_buffer_columns is not None:
            self.n_buffer_columns = n_buffer_columns

    def get_stats(self) -> dict:
        st = self.stats.get()
        n = st.get("steps", 0.0)
        ep = st.get("episodes", 0.0)
        return {
            "ep_lens": safe_div(st.get("ep_lens", 0.0), ep),
            "arrival_rate": safe_div(st.get("arrivals", 0.0), n),
            "query_rate": safe_div(st.get("queries", 0.0), n),
        }
