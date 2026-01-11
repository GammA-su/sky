"""Training logging utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console


class Logger:
    def __init__(self, log_path: str | None = None) -> None:
        self.console = Console()
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = self.log_path.open("w", encoding="utf-8")
        else:
            self._fp = None

    def log(self, data: dict[str, Any]) -> None:
        msg = " ".join(f"{k}={v}" for k, v in data.items())
        self.console.print(msg)
        if self._fp:
            self._fp.write(json.dumps(data) + "\n")
            self._fp.flush()

    def close(self) -> None:
        if self._fp:
            self._fp.close()
