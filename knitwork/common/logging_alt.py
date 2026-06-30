from __future__ import annotations

import os

import numpy as np

from knitwork.common.base import to_readable_num
from knitwork.common.config import ns_to_dict
from knitwork.common.timing import Timer
from knitwork.common.tracking import make_tracker


class Logger:
    """Base logger that tracks metrics and prints them according to schedule."""

    def __init__(
            self, logger, *, log_schedule, log_perf=True, suppress_printing=False,
            tracker=None
    ):
        if isinstance(logger, dict):
            logger = self._start_run(config=logger)

        self.logger = logger
        self.log_schedule = log_schedule
        self.suppress_printing = suppress_printing

        self.tracker = make_tracker(tracker)

        self.timer = Timer() if log_perf else None
        self.last_flush_step = 0
        self.next_flush = log_schedule

    def accumulate(self, metrics, *, prefix=None, **tracker_kwargs):
        self.tracker.put(metrics, prefix=prefix, **tracker_kwargs)

    def set_summary(self, metrics, *, prefix=None, **tracker_kwargs):
        self.tracker.set(metrics, prefix=prefix, **tracker_kwargs)

    def log(self, step, metrics=None, *, prefix=None, flush=False, force=False, **tracker_kwargs):
        """Log == accumulate + [optional and scheduled] flush."""
        self.accumulate(metrics, prefix=prefix, **tracker_kwargs)
        if flush and (step >= self.next_flush or force):
            self.flush(step)

    def flush(self, step, *, suppress_printing=None):
        """Flush collected stats to logger."""
        # if self.tracker.is_empty:
        #     return

        scalar_metrics, figure_metrics = {}, {}
        for k, v in self.tracker.get().items():
            if isinstance(v, list):
                # for now, we only store figures as a single object, so extract it from the history
                figure_metrics[k] = v[0]
            else:
                scalar_metrics[k] = v

        scalar_metrics['global_step'] = step
        if self.timer is not None and step > self.last_flush_step:
            self.timer.commit(step - self.last_flush_step, new_after=True)
            scalar_metrics['perf/fps'] = self.timer.fps(last=True)

        if self.logger is not None:
            self._log(step, scalars=scalar_metrics, figures=figure_metrics)

        if suppress_printing is None:
            suppress_printing = self.suppress_printing
        if not suppress_printing:
            print_metrics(step, scalar_metrics)

        self.tracker.clear()
        self.last_flush_step = step
        self.next_flush = step + self.log_schedule

    def finish(self):
        if self.logger is None:
            return
        self._finish_run()

    @classmethod
    def _start_run(cls, config):
        assert config['log']['logger'] is None
        return None

    def _log(self, step, scalars, figures):
        ...

    def _finish_run(self):
        ...


class CustomWandBLogger(Logger):
    """Wandb logger with configurable logging frequency."""

    def __init__(self, *, logger, log_schedule, log_perf=True, suppress_printing=False, tracker=None):
        super().__init__(
            logger=logger, log_schedule=log_schedule,
            log_perf=log_perf, suppress_printing=suppress_printing, tracker=tracker
        )

    def _log(self, step, scalars, figures):
        self.logger.log(scalars | figures)

    def _finish_run(self):
        self.logger.finish()

    @classmethod
    def _start_run(cls, config):
        log_cfg = config['log']
        assert log_cfg['logger'] == 'wandb'
        assert "WANDB_API_KEY" in os.environ

        import wandb
        return wandb.init(
            project=log_cfg['project'],
            tags=log_cfg.get('tags', []),
            name=log_cfg.get('name', None),
            config=config,
            save_code=False,
        )


class CustomCometLogger(Logger):
    """Comet ML logger with configurable logging frequency."""

    def __init__(self, *, logger, log_schedule, log_perf=True, suppress_printing=False, tracker=None):
        super().__init__(
            logger=logger, log_schedule=log_schedule,
            log_perf=log_perf, suppress_printing=suppress_printing, tracker=tracker
        )

    def _log(self, step, scalars, figures):
        self.logger.log_metrics(scalars, step=step)

        for fig_name, fig in figures.items():
            self.logger.log_figure(figure_name=fig_name, figure=fig, step=step)

    def _finish_run(self):
        self.logger.end()

    @classmethod
    def _start_run(cls, config):
        log_cfg = config['log']
        assert log_cfg['logger'] == 'comet'
        assert "COMET_API_KEY" in os.environ

        import comet_ml
        run = comet_ml.start(
            project_name=log_cfg['project'],
            experiment_config=comet_ml.ExperimentConfig(
                log_code=False, log_graph=False,
                display_summary_level=0,
                log_git_metadata=False, log_git_patch=False,

                auto_metric_logging=False,
                name=log_cfg.get('name', None),
                tags=log_cfg.get('tags', []),
            )
        )
        run.log_parameters(config)
        return run


class CustomAimLogger(Logger):
    """Aim logger with configurable logging frequency."""

    def __init__(self, *, logger, log_schedule, log_perf=True, suppress_printing=False, tracker=None):
        super().__init__(
            logger=logger, log_schedule=log_schedule,
            log_perf=log_perf, suppress_printing=suppress_printing, tracker=tracker
        )

    def _log(self, step, scalars, figures):
        self.logger.log_metrics(scalars, step=step)

        for fig_name, fig in figures.items():
            self.logger.log_figure(figure_name=fig_name, figure=fig, step=step)

    def _finish_run(self):
        self.logger.end()

    @classmethod
    def _start_run(cls, config):
        log_cfg = config['log']
        assert log_cfg['logger'] == 'aim'

        import aim
        run = aim.Run(experiment=log_cfg['project'])
        run.name = log_cfg.get('name') or ''
        print(f'Logging to Aim: {run.hash} "{run.name}" ({log_cfg["project"]})')
        run['hparams'] = config | dict(project=log_cfg['project'])

        return run


def start_logger(config, log_perf=True, suppress_printing=False, tracker=None) -> Logger:
    config = _make_serializable(ns_to_dict(config))
    log_cfg = config['log']
    logger_type = log_cfg['logger']
 
    match logger_type:
        case 'wandb':
            logger_factory = CustomWandBLogger
        case 'comet':
            logger_factory = CustomCometLogger
        case _:
            logger_factory = Logger

    log_perf = log_cfg.get('log_perf', log_perf)
    suppress_printing = log_cfg.get('suppress_printing', suppress_printing)

    return logger_factory(
        logger=config,
        log_schedule=log_cfg['schedule'],
        log_perf=log_perf,
        suppress_printing=suppress_printing,
        tracker=tracker
    )


def _make_serializable(obj):
    if isinstance(obj, set):
        return sorted(list(obj))
    if isinstance(obj, dict):
        return {
            k: _make_serializable(v) 
            for k, v in obj.items()
            if not isinstance(v, np.ndarray)
        }
    return obj


def print_metrics(step, metrics):
    _step, _sfx = to_readable_num(step)
    msgs = [f'[{_step:.0f}{_sfx}]: ']
    msgs.extend(
        f'  {k}: {v:.5f}'
        for k, v in metrics.items()
    )
    print(''.join(msgs))
