from knitwork.common.scheduler import Scheduler, create_scheduler
from knitwork.common.utils import to_readable_num


class DynamicParameter:
    def __init__(
            self, val: float, *, name: str = '',

            # set target explicitly or via relative to the initial value
            tar: float = None, rel: float = 1.0,

            schedule: dict | Scheduler = None,

            # mismatch-based change, factor = 1 - lr; val += lr * (tar - val)
            factor: float = None, lr: float = None,

            # const change by fraction of the full delta D = (target - initial)
            # i.e. val += fraction * D, 
            # or you can set the number of change events n_linear_steps to induce the fraction
            fraction: float = None, n_linear_steps: int = None,

            # warmup for the parameter, can be set as a separate DynamicParameter, its config or just a scheduler
            warmup: dict | Scheduler = None,

            print_debug: bool = False
    ):
        self.param = DynamicParameterBase(
            val=val, name=name, tar=tar, rel=rel, schedule=schedule, 
            factor=factor, lr=lr, fraction=fraction, n_linear_steps=n_linear_steps,
            print_debug=print_debug
        )

        if warmup is not None:
            warmup_cfg = dict(
                val=0.0, tar=self.param.val, name=f'{self.param.name}_warmup', warmup=None
            )
            if isinstance(warmup, dict):
                warmup = DynamicParameter(**(warmup_cfg | warmup))
            elif isinstance(warmup, Scheduler):
                warmup = DynamicParameter(**warmup_cfg, schedule=warmup)
        self.warmup = warmup
    
    @property
    def val(self):
        return self.warmup.val if self.is_warmup_stage() else self.param.val
    
    @property
    def base_val(self):
        return self.param.base_val

    @property
    def name(self):
        return self.param.name

    def step(self, n_steps=1):
        if self.is_warmup_stage():
            return self.warmup.step(n_steps)
        else:
            return self.param.step(n_steps)
    
    def is_enough(self):
        # don't need to check warmup separately, since param defines the final stage anyway
        return self.param.is_enough()

    def is_warmup_stage(self):
        return self.warmup is not None and not self.warmup.is_enough()


class DynamicParameterBase:
    name: str
    base_val: float
    val: float
    tar: float

    scheduler: Scheduler

    is_lr_based: bool
    lr: float
    fraction: float
    delta: float


    def __init__(
            self, val: float, *, name: str = '',

            # set target explicitly or via relative to the initial value
            tar: float = None, rel: float = 1.0,

            schedule: int | dict | Scheduler = None,

            # mismatch-based change, factor = 1 - lr; val += lr * (tar - val)
            factor: float = None, lr: float = None,

            # const change by fraction of the full delta D = (target - initial)
            # i.e. val += fraction * D, 
            # or you can set the number of change events n_linear_steps to induce the fraction
            fraction: float = None, n_linear_steps: int = None,

            print_debug: bool = False
    ):
        assert tar is not None or rel is not None

        self.scheduler = create_scheduler(schedule)
        assert (
            self.scheduler.is_infinite
            or factor is not None or lr is not None 
            or fraction is not None or n_linear_steps is not None
        )

        self.name = name
        self.val = self.base_val = val
        self.tar = tar if tar is not None else val * rel

        self.is_lr_based = True
        self.lr = 0.0
        if factor is not None or lr is not None:
            self.lr = 1 - factor if factor is not None else lr

        self.fraction = 0.0
        self.delta = self.tar - self.val
        if fraction is not None or n_linear_steps is not None:
            self.fraction = fraction if fraction is not None else 1.0 / n_linear_steps
            self.is_lr_based = False

        self._print_debug = print_debug
        if self._print_debug:
            self.print_state("Init")

    def step(self, n_steps=1):
        n_changes = self.scheduler.tick(n_steps)
        if n_changes == 0:
            return False

        for _ in range(n_changes):
            self.apply_change()
            if self.is_enough():
                # turn off scheduling at all
                self.scheduler.set_new(0)
                break

        if self._print_debug:
            self.print_state("New")
        return True

    def apply_change(self):
        if self.is_lr_based:
            # mismatch-based change
            self.val += self.lr * (self.tar - self.val)
        else:
            # constant change
            self.val += self.fraction * self.delta
    
    def is_enough(self):
        return abs(self.val - self.tar) <= 1e-4 * (abs(self.val) + abs(self.tar))

    def print_state(self, prefix):
        v, sfx = to_readable_num(self.scheduler.schedule)
        print(f'{prefix} {self.name}: {self.val:.6f} for {v:.2f}{sfx} steps')
        return True
