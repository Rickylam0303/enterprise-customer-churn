"""Data loading and cleaning utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path
from src.schema import clean_total_charges, clean_customer_data, validate_dataset, validate_target


def load_raw_data(
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
    *, config_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load train (and optional test) CSV files from disk."""
    if train_path is None or config_path is not None:
        config = load_config(config_path)
        train_path = train_path if train_path is not None else resolve_path(config, "train")
        test_path = test_path if test_path is not None else resolve_path(config, "test")

    df_train = pd.read_csv(train_path)
    df_test = None
    if test_path and Path(test_path).exists():
        df_test = pd.read_csv(test_path)
    return df_train, df_test


def load_train_data(train_path: str | Path | None = None) -> pd.DataFrame:
    """Load validated raw fields; pipelines alone learn feature statistics."""
    path = train_path if train_path is not None else resolve_path(load_config(), "train")
    return validate_dataset(pd.read_csv(path), require_target=True)


def load_test_data(test_path: str | Path | None = None) -> pd.DataFrame:
    """Load validated raw scoring fields; no scoring-batch statistics are fitted."""
    path = test_path if test_path is not None else resolve_path(load_config(), "test")
    return validate_dataset(pd.read_csv(path), require_target=False)


def add_churn_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary Churn_binary column for EDA (train data only)."""
    out = df.copy()
    if "Churn" in out.columns:
        out["Churn_binary"] = validate_target(out["Churn"]).map({"Yes": 1, "No": 0})
    return out
