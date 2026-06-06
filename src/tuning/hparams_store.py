"""Load/save tuned hyperparameters without importing model runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_best_hyperparams(path: str | Path = "data/processed/best_hyperparams.json") -> Dict[str, Any]:
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_best_hyperparams(params: Dict[str, Any], path: str | Path = "data/processed/best_hyperparams.json") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_convert(params), fh, indent=2)
    print(f"Best hyperparameters saved: {path}")
