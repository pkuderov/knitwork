from knitwork.common.scheduler import Scheduler

_AXES = ('T', 'p_store', 'p_query')


class CurriculumScheduler:
    """
    Curriculum scheduler with pluggable acceptance modes.

    Modes:
      speed     — original: advance when avg_speed > 0 (loss/acc improving on average)
      threshold — advance when avg_speed > 0 AND metric crosses threshold
      plateau   — advance when metric is stable near threshold (low std over history)
      multiaxis — each difficulty axis advances independently by its own accuracy gate
    """
    def __init__(
            self, scheduler: Scheduler, key: str,
            mode: str = 'speed',
            minimization: bool = True,
            threshold: float = 0.0,
            plateau_len: int = 5,
            plateau_tol: float = 0.02,
            t_threshold: float = 0.75,
            store_threshold: float = 0.80,
            query_threshold: float = 0.70,
            allowed_range=(0.25, 4.0),
            reinforce_factors=(1.25, 0.97),
            lr=0.1,
    ):
        self.scheduler = scheduler
        self.key = key
        self.mode = mode
        min_sc, max_sc = allowed_range
        sc = self.scheduler.schedule
        self.min_schedule, self.max_schedule = sc * min_sc, sc * max_sc
        self.sign = -1 if minimization else 1
        self.penalty_scale = max(reinforce_factors)
        self.reinforce_scale = min(reinforce_factors)
        self.threshold = threshold
        self.plateau_len = plateau_len
        self.plateau_tol = plateau_tol
        self.t_threshold = t_threshold
        self.store_threshold = store_threshold
        self.query_threshold = query_threshold

        self.last_val = 0.0
        self.avg_speed = 0.0
        self.history: list = []
        self.cnt = 0
        self.lr = lr
        self.cnt_accepted = 0

    def tick(self, metrics: dict, n_steps=1) -> dict:
        """Returns per-axis advance flags: {'T': bool, 'p_store': bool, 'p_query': bool}."""
        if not self.scheduler.tick(n_steps):
            return {k: False for k in _AXES}

        self.cnt += 1
        lr = max(self.lr, 1.0 / self.cnt)

        val = metrics[self.key]
        speed = self.sign * (val - self.last_val)
        accel = speed - self.avg_speed
        self.avg_speed += lr * accel
        self.last_val = val

        # adaptive schedule tempo
        if 1.05 * speed < self.avg_speed and self.scheduler.schedule < self.max_schedule:
            self.scheduler.set_new(int(self.scheduler.schedule * self.penalty_scale))
        if self.avg_speed > 0.0 and self.scheduler.schedule > self.min_schedule:
            self.scheduler.set_new(int(self.scheduler.schedule * self.reinforce_scale))

        axes = self._accept(val, metrics)
        self.cnt_accepted += int(any(axes.values()))
        return axes

    def _accept(self, val: float, metrics: dict) -> dict:
        if self.mode == 'speed':
            ok = self.avg_speed > 0.0
            return {k: ok for k in _AXES}

        if self.mode == 'threshold':
            at_threshold = (self.sign * val) >= (self.sign * self.threshold)
            ok = self.avg_speed > 0.0 and at_threshold
            return {k: ok for k in _AXES}

        if self.mode == 'plateau':
            self.history.append(val)
            if len(self.history) > self.plateau_len:
                self.history.pop(0)
            if len(self.history) < 3:
                return {k: False for k in _AXES}
            h = self.history
            mean = sum(h) / len(h)
            std = (sum((x - mean) ** 2 for x in h) / len(h)) ** 0.5
            is_plateau = std < self.plateau_tol
            at_threshold = (self.sign * val) >= (self.sign * self.threshold)
            ok = is_plateau and at_threshold
            return {k: ok for k in _AXES}

        if self.mode == 'multiaxis':
            return {
                'T':       metrics.get('Acc',       0.0) >= self.t_threshold,
                'p_store': metrics.get('Acc/store', 0.0) >= self.store_threshold,
                'p_query': metrics.get('Acc/query', 0.0) >= self.query_threshold,
            }

        raise ValueError(f'Unknown curriculum mode: {self.mode!r}')
