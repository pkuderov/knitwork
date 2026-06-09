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
    payload = {
        "step": step,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {k: round(float(v), 5) for k, v in metrics.items()
                    if isinstance(v, (int, float)) and k != "global_step"},
    }
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)
