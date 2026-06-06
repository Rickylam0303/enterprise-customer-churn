"""Feature matrix preparation and sklearn preprocessing."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "service_count",
    "tenure_x_monthly",
    "mtm_early",
    "electronic_check",
    "high_value",
    "missing_online_sec",
]

CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
]

DROP_COLS = ["Churn", "id", "customerID", "Churn_binary"]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model input column names from an engineered dataframe."""
    return [col for col in df.columns if col not in DROP_COLS]


def build_preprocessor(
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
) -> ColumnTransformer:
    """Build the ColumnTransformer used by all model pipelines."""
    num_cols = num_cols or NUMERIC_COLS
    cat_cols = cat_cols or CATEGORICAL_COLS

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Split engineered dataframe into features X and optional target y."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()

    y = None
    if "Churn" in df.columns:
        y = df["Churn"].map({"Yes": 1, "No": 0})

    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def prepare_datasets(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare stratified train/test split from an engineered training dataframe."""
    if config is None:
        from src.config import load_config

        config = load_config()

    X, y = prepare_xy(df)
    if y is None:
        raise ValueError("Training dataframe must include a 'Churn' column.")

    return split_train_test(
        X,
        y,
        test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
    )
