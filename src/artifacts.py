"""Paths and file identities for reproducible evaluation outputs."""
from __future__ import annotations
import hashlib
from pathlib import Path


def file_sha256(path) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def output_path(config, key, default_name) -> Path:
    if key in config["paths"]:
        return Path(config["paths"][key])
    metrics = Path(config["paths"]["metrics"])
    prefix = "" if metrics.name == "metrics.json" else metrics.stem + "-"
    return metrics.parent / (prefix + default_name)


def record_path(config, path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(config["_project_root"]).resolve()).as_posix()
    except ValueError:
        return str(path)


def recorded_path(config, value) -> Path:
    return Path(config["_project_root"]) / value


def verify_artifact(config, record) -> Path:
    path = recorded_path(config, record["path"])
    if not path.is_file() or file_sha256(path) != record["sha256"]:
        raise ValueError(f"Missing or changed run artifact: {path}")
    return path
