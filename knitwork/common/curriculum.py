from knitwork.common.scheduler import Scheduler


class CurriculumScheduler:
    def __init__(
            self, scheduler: Scheduler, key: str, 
            allowed_range=(0.25, 4.0), minimization: bool = True,
            reinforce_factors=(1.25, 0.97), lr=0.1
    ):
        self.scheduler = scheduler

        self.key = key
        min_sc, max_sc = allowed_range
        sc = self.scheduler.schedule
        self.min_schedule, self.max_schedule = sc * min_sc, sc * max_sc
        self.sign = -1 if minimization else 1
        self.penalty_scale = max(reinforce_factors)
        self.reinforce_scale = min(reinforce_factors)

        self.last_val = 0.0
        self.avg_speed = 0.0
        self.cnt = 0
        self.lr = lr
        self.cnt_accepted = 0

    def tick(self, metrics: dict, n_steps=1):
        if not self.scheduler.tick(n_steps):
            return False

        self.cnt += 1
        lr = self.lr
        if self.cnt < 100:
            lr = max(lr, 1/self.cnt)

        val = metrics[self.key]
        # normalize sign => positive speed — good
        speed = self.sign * (val - self.last_val)
        accel = speed - self.avg_speed
 
        self.avg_speed += lr * accel
        self.last_val = val

        # == accel < -0.05 * speed
        if 1.05 * speed < self.avg_speed and self.scheduler.schedule < self.max_schedule:
            # if slowing down or accelerating backward, slow down curriculum tempo
            new_sc = int(self.scheduler.schedule * self.penalty_scale)
            self.scheduler.set_new(new_sc)
        if self.avg_speed > 0.0 and self.scheduler.schedule > self.min_schedule:
            # if going forward, slightly increase curriculum tempo
            new_sc = int(self.scheduler.schedule * self.reinforce_scale)
            self.scheduler.set_new(new_sc)

        # accept curriculum step if going forward
        accept_step = self.avg_speed > 0.0
        self.cnt_accepted += accept_step
        return accept_step
