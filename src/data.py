"""Data loading and cleaning utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path
from src.features import engineer_features


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce TotalCharges to numeric and fill missing values with 0."""
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out.fillna(0, inplace=True)
    return out


def load_raw_data(
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load train (and optional test) CSV files from disk."""
    if train_path is None or test_path is None:
        config = load_config()
        train_path = train_path or resolve_path(config, "train")
        test_path = test_path or resolve_path(config, "test")

    df_train = pd.read_csv(train_path)
    df_test = None
    if test_path and Path(test_path).exists():
        df_test = pd.read_csv(test_path)
    return df_train, df_test


def load_train_data(train_path: str | Path | None = None) -> pd.DataFrame:
    """Load, clean, and engineer features for the training set."""
    config = load_config()
    path = train_path or resolve_path(config, "train")
    df = pd.read_csv(path)
    df = clean_total_charges(df)
    return engineer_features(df)


def load_test_data(test_path: str | Path | None = None) -> pd.DataFrame:
    """Load, clean, and engineer features for the test set."""
    config = load_config()
    path = test_path or resolve_path(config, "test")
    df = pd.read_csv(path)
    df = clean_total_charges(df)
    return engineer_features(df)


def add_churn_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary Churn_binary column for EDA (train data only)."""
    out = df.copy()
    if "Churn" in out.columns:
        out["Churn_binary"] = out["Churn"].map({"Yes": 1, "No": 0})
    return out
