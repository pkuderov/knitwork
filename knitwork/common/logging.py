
from __future__ import annotations

import os
from typing import Any

_UNSET = object()  # sentinel: step not explicitly provided


class NullLogger:
    """Absorbs all tracking calls when logging is disabled."""

    def track(self, value: Any, name: str = '', step: Any = _UNSET) -> None:
        pass

    @property
    def name(self) -> str:
        return ''

    @name.setter
    def name(self, value: str) -> None:
        pass

    def end(self) -> None:
        pass


class Logger:
    """Uniform .track() interface over AIM or Comet backends."""

    def __init__(self, backend: str, run):
        self._backend = backend  # 'aim' | 'comet'
        self._run = run

    @property
    def name(self) -> str:
        return self._run.name if self._backend == 'aim' else (self._run.name or '')

    @name.setter
    def name(self, value: str) -> None:
        if self._backend == 'aim':
            self._run.name = value
        else:
            self._run.set_name(value)

    def track(self, value: Any, name: str = '', step: Any = _UNSET) -> None:
        if self._backend == 'aim':
            # AIM requires name=None when tracking a dict
            _name = None if isinstance(value, dict) else name
            if step is _UNSET:
                self._run.track(value, name=_name)
            else:
                self._run.track(value, name=_name, step=step)
        else:
            self._track_comet(value, name, step)

    def _track_comet(self, value: Any, name: str, step: Any) -> None:
        # aim.Image → extract PIL and log as image
        try:
            from aim import Image as AimImage
            if isinstance(value, AimImage):
                _step = None if step is _UNSET else step
                try:
                    pil_img = value.to_pil_image()
                    self._run.log_image(pil_img, name=name, step=_step)
                except Exception:
                    pass
                return
        except ImportError:
            pass

        # PIL Image directly
        try:
            from PIL import Image as PILImage
            if isinstance(value, PILImage.Image):
                _step = None if step is _UNSET else step
                self._run.log_image(value, name=name, step=_step)
                return
        except ImportError:
            pass

        # dict of scalars — extract global_step if step not explicitly provided
        if isinstance(value, dict):
            flat = _flatten_dict(value)
            if step is _UNSET:
                _step = int(flat.pop('global_step', 0))
            else:
                flat.pop('global_step', None)
                _step = step
            # filter NaN/Inf which Comet silently drops but may cause issues
            flat = {k: v for k, v in flat.items() if v == v and abs(v) != float('inf')}
            if flat:
                self._run.log_metrics(flat, step=_step)
            return

        # scalar
        _step = 0 if step is _UNSET else step
        try:
            self._run.log_metric(name=name, value=float(value), step=_step)
        except (TypeError, ValueError):
            pass

    def end(self) -> None:
        if self._backend == 'comet':
            self._run.end()


# ── helpers ───────────────────────────────────────────────────────────────────

def _flatten_dict(d: dict, prefix: str = '') -> dict:
    out = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=key))
        elif isinstance(v, (int, float)):
            out[key] = v
        else:
            try:
                out[key] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _project_tags(project_name: str) -> list[str]:
    name = project_name.lower()
    tags = ['grnn']
    for token in ('sdq', 'text', 'mikasa', 'treasure'):
        if token in name:
            tags.append(token)
            break
    return tags


# ── factory ───────────────────────────────────────────────────────────────────

def create_logger(config: dict) -> Logger | NullLogger:
    cfg_log = config.get('log', {})
    if not cfg_log.get('enabled', False):
        return NullLogger()
    match cfg_log.get('logger', 'aim'):
        case 'aim':
            return _make_aim_logger(config)
        case 'comet':
            return _make_comet_logger(config)
        case _:
            return NullLogger()


def _make_aim_logger(config: dict) -> Logger:
    from aim import Run
    cfg_log = config['log']
    run = Run(experiment=cfg_log['project'])
    run.name = cfg_log.get('name') or ''
    print(f'Logging to Aim: {run.hash} "{run.name}" ({cfg_log["project"]})')
    run['hparams'] = config | dict(project=cfg_log['project'])
    return Logger('aim', run)


def _make_comet_logger(config: dict) -> Logger:
    import sys
    import comet_ml
    import comet_ml.config as _cml_cfg
    cfg_log = config['log']

    # prevent main-thread deadlock when network is intermittent:
    # shorter HTTP timeout + smaller upload queue so log_metrics() never blocks long
    _cml_cfg.COMET_TIMEOUT_HTTP    = int(os.environ.get('COMET_TIMEOUT_HTTP', '15'))
    _cml_cfg.COMET_HTTP_TIMEOUT    = int(os.environ.get('COMET_TIMEOUT_HTTP', '15'))
    _cml_cfg.COMET_WORKER_TIMEOUT  = int(os.environ.get('COMET_TIMEOUT_HTTP', '15'))

    api_key = os.environ.get('COMET_API_KEY') or cfg_log.get('comet_api_key', '')
    workspace = cfg_log.get('comet_workspace') or None
    tags = _project_tags(cfg_log.get('project', ''))

    # comet_ml patches sys.stdout regardless of auto_output_logging; restore it after init
    _stdout = sys.stdout
    _stderr = sys.stderr
    experiment = comet_ml.Experiment(
        api_key=api_key,
        # `project` selects the tag set (_project_tags) and is not the Comet project.
        # comet_project picks the Comet project itself; default keeps old behavior.
        project_name=(
            cfg_log.get('comet_project')
            or os.environ.get('COMET_PROJECT_NAME')
            or 'knitwork'
        ),
        workspace=workspace,
        log_code=False,
        log_git_metadata=False,
        log_git_patch=False,
        log_env_details=False,
        log_env_gpu=False,
        log_env_cpu=False,
        log_env_network=False,
        log_env_disk=False,
        log_env_host=False,
        auto_output_logging=False,
    )
    sys.stdout = _stdout
    sys.stderr = _stderr

    run_name = cfg_log.get('name') or ''
    if run_name:
        experiment.set_name(run_name)
    experiment.add_tags(tags)
    experiment.log_parameters(_flatten_dict(config))

    print(f'Logging to Comet: "{run_name}" tags={tags}')
    return Logger('comet', experiment)
