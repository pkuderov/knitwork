from dataclasses import dataclass


@dataclass
class MetricsCollector:
    lr: float = 5e-4

    def __post_init__(self):
        self.stats = dict()
        self._step = 0.0

    def put(self, values):
        self._step += 1.0
        lr = max(self.lr, 1.0 / self._step)

        for k, v in values.items():
            delta = v - self.stats.setdefault(k, 0.0)
            self.stats[k] += lr * delta

    def get(self):
        return self.stats

    def __getitem__(self, key):
        return self.stats[key]
