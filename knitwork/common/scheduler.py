from __future__ import annotations

from numba.experimental import jitclass


@jitclass
class Scheduler:
    remain: int
    schedule: int
    slowdown: float
    total_events: int

    def __init__(self, schedule: int | None = 0, slowdown: float = 0.0):
        if schedule is None:
            schedule = 0

        self.schedule = schedule
        self.slowdown = slowdown

        self.remain = self.schedule
        self.total_events = 0
    
    def set_new(self, schedule: int):
        self.schedule = schedule
        self.remain = self.schedule

    @property
    def is_infinite(self):
        return self.schedule == 0

    @property
    def ticks_passed(self):
        """
        Returns the number of ticks passed since the last reset.
        NB: for infinite scheduler, this is the total ticks passed.
        """
        return self.schedule - self.remain

    def tick(self, n: int = 1) -> int:
        self.remain -= n

        if self.is_infinite:
            return 0

        n_events = 0
        while self.remain <= 0:
            if self.slowdown != 0.0:
                self.schedule = int(self.schedule * (1.0 + self.slowdown))
            self.remain += self.schedule
            n_events += 1
        
        self.total_events += n_events
        return n_events

    def reset(self):
        self.remain = self.schedule