class Tracker:
    lr: float
    stats: dict

    def __init__(self, lr):
        self.lr = lr
        self.stats = dict()
        self._step = 1.0

    def put(self, values, inc_step=True):
        lr = max(self.lr, 1.0 / self._step)
        if inc_step:
            self._step += 1.0

        for k, v in values.items():
            delta = v - self.stats.setdefault(k, 0.0)
            self.stats[k] += lr * delta

    def get(self):
        return self.stats

    def __getitem__(self, key):
        return self.stats[key]


class TrackerCollection:
    trackers: dict[str, Tracker]

    def __init__(self, lrs: dict[str, float]):
        self.trackers = {
            k: Tracker(lr)
            for k, lr in lrs.items()
        }
    
    def put(self, values, key):
        self.trackers[key].put(values)
    
    def get(self):
        return {
            k: v
            for tracker in self.trackers.values()
            for k, v in tracker.get().items()
        }

    def __getitem__(self, key):
        return self.trackers[key]
