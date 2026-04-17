from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_json") and isinstance(record.extra_json, dict):
            base.update(record.extra_json)
        return json.dumps(base, default=str)


def setup_json_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "run.jsonl"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Clear any pre-existing handlers to keep output predictable.
    for h in list(root.handlers):
        root.removeHandler(h)

    fh = RotatingFileHandler(
        log_path,
        encoding="utf-8",
        maxBytes=5_000_000,
        backupCount=5,
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(JsonFormatter())
    root.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root.addHandler(sh)

