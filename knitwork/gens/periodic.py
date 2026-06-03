"""
Periodic Sequence Generator

Проверка памяти через периодические паттерны.

Задача: предсказать следующий символ в бесконечно зацикленной
последовательности фиксированного периода P.

    t=0  : a b c d e | a b c d e | a b c d e | ...
    input:  a b c d e   a b c d e   ...
    target: b c d e a   b c d e a   ...

Сложность задачи растёт с P: при большем периоде нужно помнить
больше шагов назад.

 pure      — чистый цикл 0..P-1, 0..P-1, ...
 noisy     — с вероятностью p_noise символ заменяется случайным
 multi     — несколько разных периодов, каждый env получает свой
 phase     — каждый env стартует в случайной фазе
"""

from __future__ import annotations

import numpy as np
from knitwork.common.utils import CE_ignore_index


class PeriodicGenerator:
    """

    period : int | list[int]
        Период(ы). Если список — каждому env назначается свой.
    n_envs : int
    mode : {"pure", "noisy", "phase"}
        pure  — детерминированный цикл
        noisy — с p_noise вероятностью замена на random token
        phase — случайный начальный сдвиг (env должен угадать фазу)
    p_noise : float
        Используется только при mode="noisy".
    seed : int
    ignore_index : int
    """

    def __init__(
        self,
        *,
        period: int | list[int],
        n_envs: int,
        mode: str = "pure",
        p_noise: float = 0.05,
        seed: int = 0,
        ignore_index: int = CE_ignore_index,
    ):
        self.rng = np.random.default_rng(seed)
        self.n_envs = n_envs
        self.mode = mode
        self.p_noise = p_noise
        self.ignore_index = ignore_index

        if isinstance(period, int):
            self.periods = np.full(n_envs, period, dtype=np.int64)
        else:
            periods = np.asarray(period, dtype=np.int64)
            self.periods = np.array(
                [periods[i % len(periods)] for i in range(n_envs)],
                dtype=np.int64,
            )

        self.max_period = int(self.periods.max())
        self.n_tokens = self.max_period
        self.V = self.max_period

        if mode == "phase":
            self.pos = np.array(
                [self.rng.integers(0, self.periods[i]) for i in range(n_envs)],
                dtype=np.int64,
            )
        else:
            self.pos = np.zeros(n_envs, dtype=np.int64)

        self._step = 0

    @property
    def reset_every(self):
        return None  # нет естественного сброса (бесконечный цикл)

    def next(self) -> dict[str, np.ndarray]:
        tokens = (self.pos % self.periods).astype(np.int64)

        if self.mode == "noisy":
            noise_mask = self.rng.random(self.n_envs) < self.p_noise
            random_tokens = self.rng.integers(0, self.max_period, self.n_envs)
            tokens = np.where(noise_mask, random_tokens, tokens)

        self.pos += 1

        # targets = следующий символ (чистый, без шума)
        targets = (self.pos % self.periods).astype(np.int64)

        reset_mask = np.zeros(self.n_envs, dtype=bool)


        self._step += 1
        return {
            "tokens":     tokens.copy(),
            "targets":    targets.copy(),
            "reset_mask": reset_mask,
            "phase":      (self.pos % self.periods).copy(),
        }

    def next_rollout(self, rollout: int) -> dict[str, np.ndarray]:
        result = [self.next() for _ in range(rollout)]
        keys = list(result[0].keys())
        return {k: np.stack([r[k] for r in result]) for k in keys}

    def get_stats(self) -> dict:
        return {
            "periods": self.periods.tolist(),
            "mode": self.mode,
            "step": self._step,
        }