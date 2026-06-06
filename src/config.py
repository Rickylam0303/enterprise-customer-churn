"""Load project configuration from config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the project config dict."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["_project_root"] = str(PROJECT_ROOT)
    config["paths"] = {
        key: str(PROJECT_ROOT / value) for key, value in config["paths"].items()
    }
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Return a resolved path from config['paths']."""
    return Path(config["paths"][key])
