from collections import defaultdict

import numpy as np

from knitwork.common.base import prefix_dict


class EmaTracker:
    """Track floating values in aggregated EMA form."""

    lr: float
    stats: dict

    def __init__(self, lr):
        self.lr = lr
        self.stats = dict()
        self._step = 1.0

    def put(self, values: dict, *, inc_step=True, prefix: str = None):
        """Update EMA of the tracked values."""
        if values is None or len(values) == 0:
            return

        lr = max(self.lr, 1.0 / self._step)
        if inc_step:
            # step tracking is needed only for lr "warmup"-period decay.
            self._step += 1.0

        values = prefix_dict(values, prefix)
        for k, v in values.items():
            delta = v - self.stats.setdefault(k, 0.0)
            self.stats[k] += lr * delta

    def set(self, values: dict, *, prefix: str = None):
        """Replace tracked values with new ones, i.e as if lr = 1.0."""
        if values is None or len(values) == 0:
            return

        values = prefix_dict(values, prefix)
        for k, v in values.items():
            self.stats[k] = v

    def get(self):
        return self.stats

    def __getitem__(self, key):
        return self.stats[key]

    @property
    def is_empty(self):
        return len(self.stats) == 0

    def clear(self):
        # we don't clear the tracker (=its history)
        pass

    def flush(self):
        return self.stats.copy()


class ListTracker:
    """
    Track everything by just accumulating the history in a list.
    The resulting statistics is 
    a) either an average of accumulated values
        over the last period [between flushes] 
    b) or the whole history, when it's just a list of scalars,
        in case it's a figure or the data to make it.
    """
    stats: dict

    def __init__(self):
        self.stats = defaultdict(list)
        self.is_fig = dict()

    def put(self, values, prefix=None):
        """Append values to the history."""
        if values is None or len(values) == 0:
            return

        values = prefix_dict(values, prefix)
        for k, v in values.items():
            self.stats[k].append(v)

    def set(self, values, prefix=None):
        """Replace tracked values with new ones: clear the history + append new."""
        # To track only a single last value
        if values is None or len(values) == 0:
            return

        values = prefix_dict(values, prefix)
        for k, v in values.items():
            self.stats[k].clear()
            self.stats[k].append(v)

    def get(self):
        return {k: self._aggregate(k) for k, v in self.stats.items() if len(v) > 0}

    def __getitem__(self, key):
        return self.stats[key]

    @property
    def is_empty(self):
        return len(self.stats) == 0

    def clear(self):
        self.stats.clear()

    def flush(self):
        stats = self.stats.copy()
        self.stats.clear()
        return stats

    def _aggregate(self, key):
        """Aggregate the history into a single value or return raw data if it's a figure."""
        val = self.stats[key]
        if len(val) == 0:
            return None

        is_fig = self.is_fig.get(key, None)
        if is_fig is None:
            v0 = val[0]
            is_fig = not(
                isinstance(v0, (int, float, np.number))
                or (isinstance(v0, np.ndarray) and v0.ndim < 1)
            )
            self.is_fig[key] = is_fig
        return float(np.mean(val)) if not is_fig else val


class TrackerCollection:
    """A wrapper class to combine multiple trackers into one."""
    trackers: dict[str, EmaTracker | ListTracker]

    def __init__(self, lrs: dict[str, float | None]):
        self.trackers = {
            k: EmaTracker(lr) if lr is not None else ListTracker()
            for k, lr in lrs.items()
        }
    
    def put(self, values: dict, *, prefix=None, key=None):
        """Add new or update tracked values with the new ones."""
        if values is None or len(values) == 0:
            return

        if key is not None:
            self.trackers[key].put(values, prefix=prefix)
        else:
            # assume values are two-level dict, with 
            # the first level being the tracker keys
            for k, v in values.items():
                self.trackers[k].put(v, prefix=prefix)
    
    def set(self, values: dict, *, prefix=None, key=None):
        """Add new or replace tracked values with the new ones."""
        if values is None or len(values) == 0:
            return

        if key is not None:
            self.trackers[key].set(values, prefix=prefix)
        else:
            # assume values are two-level dict, with 
            # the first level being the tracker keys
            for k, v in values.items():
                self.trackers[k].set(v, prefix=prefix)
    
    def get(self):
        """Get accumulated tracked stats."""
        return {
            k: v
            for tracker in self.trackers.values()
            for k, v in tracker.get().items()
        }

    def __getitem__(self, key):
        return self.trackers[key]

    @property
    def is_empty(self):
        return all(tracker.is_empty for tracker in self.trackers.values())

    def clear(self):
        for tracker in self.trackers.values():
            tracker.clear()

    def flush(self):
        stats = self.get().copy()
        self.clear()
        return stats


def make_tracker(config=None):
    if config is None:
        return ListTracker()
    
    if isinstance(config, dict):
        return TrackerCollection(config)

    return EmaTracker(config)
