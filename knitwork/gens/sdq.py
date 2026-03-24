from __future__ import annotations

from dataclasses import dataclass
from pprint import pprint

import numpy as np

from knitwork.common.scheduler import Scheduler
from knitwork.common.tracker import TrackerCollection
from knitwork.common.utils import CE_ignore_index, FpsCounter, format_readable_num, safe_div


class StoreDistractQueryGenerator:
    T: float
    p_store: float
    p_query: float

    def __init__(
            self, *, 
            n_keys: int, n_vals: int, T: float, 
            n_envs: int,
            p_store: float = 0.2, p_query: float = 0.2,
            count_stored: bool = False, count_queried: bool = False,
            ignore_index: int,
            seed: int = 0
        ):
        self.K = n_keys
        self.V = n_vals
        self.T = T
        self.n_store_tokens = self.K * self.V
        self.n_query_tokens = self.K
        self.n_distract_tokens = self.V

        # Token indices shifts
        self.ix_store = 0
        self.ix_distract = self.ix_store + self.n_store_tokens
        self.ix_query = self.ix_distract + self.n_distract_tokens
        self.n_tokens = self.ix_query + self.n_query_tokens

        self.rng = np.random.default_rng(seed)
        self.n_envs = n_envs
        self.inc_stored = int(count_stored)
        self.inc_queried = int(count_queried)
        self.ignore_index = ignore_index

        assert p_store + p_query < 1.0
        self.p_store = p_store
        self.p_query = p_query

        self.tokens = np.empty(n_envs, dtype=int)
        self.targets = np.full(n_envs, self.ignore_index, dtype=int)

        self.stored = np.zeros((n_envs, self.K), dtype=int)
        self.distract_accum = np.zeros(n_envs, dtype=int)
        self.n_steps = np.zeros(n_envs, dtype=int)

        self.stored_cnt = np.zeros_like(self.stored, dtype=int)
        self.queried_cnt = np.zeros_like(self.stored, dtype=int)
        self.step_stored = np.full_like(self.stored, fill_value=-1)
        self.step_queried = np.full_like(self.stored, fill_value=-1)

        lr_stats = 3e-4
        self.stats = TrackerCollection(lrs=dict(ep=lr_stats, step=lr_stats))

    @property
    def p_term(self):
        return 1.0 / self.T
    
    @property
    def lr_stats(self):
        return 0.2 / self.T

    def reset(self, ixs):
        if len(ixs) == 0:
            return

        self.stats['ep'].put({
            'episodes': len(ixs),
            'ep_lens': self.n_steps[ixs].sum()
        })

        self.stored[ixs, :] = 0
        self.distract_accum[ixs] = 0
        self.n_steps[ixs] = 0

        self.stored_cnt[ixs, :] = 0
        self.queried_cnt[ixs, :] = 0

        self.step_stored[ixs, :] = -1
        self.step_queried[ixs, :] = -1

    def next(self) -> dict:
        n_envs = self.n_envs
        self.targets[:] = self.ignore_index

        # Sample resets
        reset_mask = self.rng.random(n_envs) < self.p_term
        reset_ixs = np.flatnonzero(reset_mask)
        self.reset(reset_ixs)

        # Sample next tokens' type
        token_type = self.rng.random(n_envs)
        mask_store = token_type < self.p_store
        mask_store_query = token_type < self.p_store + self.p_query
        mask_query = ~mask_store & mask_store_query
        mask_distract = ~mask_store_query
        
        self.handle_store(ixs=np.flatnonzero(mask_store))
        sq_gaps = self.handle_query(ixs=np.flatnonzero(mask_query))
        self.handle_distract(ixs=np.flatnonzero(mask_distract))

        self.n_steps += 1
        self.stats['step'].put({
            'steps': n_envs,
        })

        return {
            'tokens': self.tokens.copy(),
            'targets': self.targets.copy(),
            'reset_mask': reset_mask,
            'sq_gaps': sq_gaps,
        }

    def handle_store(self, ixs):
        tokens = self.rng.integers(
            low=0, high=self.n_store_tokens,
            size=len(ixs)
        )
        k, v = np.divmod(tokens, self.V)

        self.stats['step'].put(
            {
                'stores': len(ixs),
                'overwrites': np.count_nonzero(self.step_stored[ixs, k] >= 0),
                's_gaps': (self.n_steps[ixs] - np.maximum(0, self.step_stored[ixs, k])).sum(),
            }, 
            inc_step=False
        )

        self.tokens[ixs] = tokens + self.ix_store
        self.stored[ixs, k] = v
        self.step_stored[ixs, k] = self.n_steps[ixs]
        self.stored_cnt[ixs, k] += self.inc_stored

    def handle_distract(self, ixs):
        tokens = self.rng.integers(
            low=0, high=self.n_distract_tokens,
            size=len(ixs)
        )

        self.tokens[ixs] = tokens + self.ix_distract
        self.distract_accum[ixs] = (self.distract_accum[ixs] + tokens) % self.V

    def handle_query(self, ixs):
        tokens = k = self.rng.integers(
            low=0, high=self.n_query_tokens,
            size=len(ixs)
        )

        mask_misses = self.step_stored[ixs, k] < 0
        sq_gaps = self.n_steps[ixs] - self.step_stored[ixs, k]
        sq_gaps[mask_misses] = -1

        self.stats['step'].put(
            {
                'queries': len(ixs),
                'misses': np.count_nonzero(mask_misses),
                'q_gaps': (self.n_steps[ixs] - np.maximum(0, self.step_queried[ixs, k])).sum(),
                'sq_gaps': sq_gaps[~mask_misses].sum(),
            },
            inc_step=False
        )

        self.step_queried[ixs, k] = self.n_steps[ixs]
        self.queried_cnt[ixs, k] += self.inc_queried
        self.tokens[ixs] = tokens + self.ix_query

        self.targets[ixs] = (
            self.stored[ixs, k] + self.distract_accum[ixs]
            + self.stored_cnt[ixs, k] - self.queried_cnt[ixs, k]
        ) % self.V

        return sq_gaps

    def set_metaparams(self, T, p_store, p_query):
        self.T = T

        assert p_store + p_query < 1.0
        self.p_store = p_store
        self.p_query = p_query
    
    def get_stats(self):
        st = self.stats.get()
        ep = st['episodes']
        n = st['steps']

        ep_lens = safe_div(st['ep_lens'], ep)
        stores = safe_div(st['stores'], n)
        queries = safe_div(st['queries'], n)
        
        overwrites = safe_div(st['overwrites'], st['stores'])
        misses = safe_div(st['misses'], st['queries'])
        s_gaps = safe_div(st['s_gaps'], st['stores'])
        q_gaps = safe_div(st['q_gaps'], st['queries'])

        non_misses = st['queries'] - st['misses']
        sq_gaps = safe_div(st['sq_gaps'], non_misses)

        res = {
            'ep_lens': ep_lens,
            'stores': stores,
            'queries': queries,
            'overwrites': overwrites,
            'misses': misses,
            's_gaps': s_gaps,
            'q_gaps': q_gaps,
            'sq_gaps': sq_gaps,
        }
        return {k: float(v) for k, v in res.items()}


@dataclass
class SdqStats:
    episodes: float = 0.0
    ep_lens: float = 0.0

    # total number of emitted tokens across all envs
    steps: float = 0.0

    stores: float = 0.0
    queries: float = 0.0
    overwrites: float = 0.0
    misses: float = 0.0

    # gap sums (in steps) averaged later
    s_gaps: float = 0.0
    q_gaps: float = 0.0
    sq_gaps: float = 0.0

    _ix_step = 0
    _ix_decay = 0

    @property
    def window(self):
        return safe_div(self.ep_lens, self.episodes, default=10.0)
    
    def tick(self):
        t = min(500.0, self.window * 2.0)
        self._ix_step += 1
        return (self._ix_step - self._ix_decay) >= t

    def decay(self, lr: float = 0.01):
        if not self.tick():
            return

        lr = max(4e-3, lr)
        b = 1.0 - lr

        self.episodes *= b
        self.ep_lens *= b
        self.steps *= b
        self.stores *= b
        self.queries *= b
        self.overwrites *= b
        self.misses *= b
        self.s_gaps *= b
        self.q_gaps *= b
        self.sq_gaps *= b

        self._ix_decay = self._ix_step


def main():
    gen = StoreDistractQueryGenerator(
        n_keys=4, n_vals=10, T=10, n_envs=64, seed=42,
        p_store=0.3, p_query=0.3, ignore_index=CE_ignore_index
    )

    n_steps = 10_000_000
    step = 0

    print_stats_schedule = Scheduler(1_000_000)
    curriculum_step_schedule = Scheduler(100_000)
    fps_counter = FpsCounter()

    while step < n_steps:
        gen.next()
        step += gen.n_envs

        if curriculum_step_schedule.tick(gen.n_envs):
            K = 10
            dT, dp_store, dp_query = 1.0, -0.004, -0.001
            dT, dp_store, dp_query = dT/K, dp_store/K, dp_query/K

            gen.set_metaparams(
                T=gen.T + dT,
                p_store=max(gen.p_store + dp_store, 0.1),
                p_query=max(gen.p_query + dp_query, 0.2)
            )

        if print_stats_schedule.tick(gen.n_envs):
            print(f'Step {format_readable_num(step)} / {format_readable_num(n_steps)}')
            pprint(gen.get_stats(), sort_dicts=False, indent=4)
    
    fps = fps_counter.fps(n_iters=step)
    print(format_readable_num(fps))

if __name__ == "__main__":
    main()
