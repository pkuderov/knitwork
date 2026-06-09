"""Write a lightweight status JSON for the running experiment.

The queue_runner sets KNITWORK_STATUS_FILE to the target path.
Call write_status() at each log interval to keep it fresh.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def write_status(step: int, metrics: dict) -> None:
    path = os.environ.get("KNITWORK_STATUS_FILE")
    if not path:
        return
    def _try_float(v):
        try:
            f = float(v)
            return None if f != f else round(f, 5)  # drop NaN
        except (TypeError, ValueError):
            return None

    payload = {
        "step": step,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {k: fv for k, v in metrics.items()
                    if k != "global_step" and (fv := _try_float(v)) is not None},
    }
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)
