from knitwork.common.dynamic_param import DynamicParameter


class DynamicLearningRate(DynamicParameter):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('name', 'LR')
        super().__init__(*args, **kwargs)
        self.optimiser = None

    def connect_to_optimiser(self, optimiser):
        self.optimiser = optimiser

    def step(self):
        if super().step():
            for pg in self.optimiser.param_groups:
                pg['lr'] = self.val
