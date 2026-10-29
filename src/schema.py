"""Deterministic raw-profile contract shared by training and serving."""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORY_VALUES = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"}, "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    **{name: {"Yes", "No", "No internet service"} for name in [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {"Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"},
}
RAW_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def validate_tenure(tenure: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(tenure, errors="coerce")
    if (numeric.isna() | ~numeric.between(0, 100) | numeric.mod(1).ne(0)).any():
        raise ValueError("tenure must contain whole months between 0 and 100, with no missing values.")
    return numeric


def _required_numeric(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain finite numeric values without missing values.")
    return numeric


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Impute blank TotalCharges only for zero-month customers; reject corruption.

    No other column is filled. Historical charges need not equal current price ×
    tenure, because this static dataset does not describe price changes or credits.
    """
    out = df.copy()
    values = out["TotalCharges"]
    blank = values.isna() | values.astype("string").str.strip().eq("").fillna(False)
    numeric = pd.to_numeric(values, errors="coerce")
    if (numeric.isna() & ~blank).any():
        raise ValueError("TotalCharges contains a non-numeric value.")
    if blank.any():
        months = validate_tenure(out["tenure"])
        if (blank & months.ne(0)).any():
            raise ValueError("Missing TotalCharges is allowed only when tenure is 0.")
        numeric = numeric.mask(blank, 0)
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all() or numeric.lt(0).any():
        raise ValueError("TotalCharges must be finite and non-negative.")
    out["TotalCharges"] = numeric
    return out


def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize raw fields without learning any batch statistics."""
    missing = sorted(set(RAW_NUMERIC + list(CATEGORY_VALUES)) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required customer columns: {', '.join(missing)}")
    out = df.copy()
    out["tenure"] = validate_tenure(out["tenure"])
    out["SeniorCitizen"] = _required_numeric(out["SeniorCitizen"], "SeniorCitizen")
    if not out["SeniorCitizen"].isin([0, 1]).all():
        raise ValueError("SeniorCitizen must be 0 or 1.")
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)
    out["MonthlyCharges"] = _required_numeric(out["MonthlyCharges"], "MonthlyCharges")
    if out["MonthlyCharges"].lt(0).any():
        raise ValueError("MonthlyCharges must be non-negative.")
    out = clean_total_charges(out)
    for name, allowed in CATEGORY_VALUES.items():
        out[name] = out[name].astype("string").str.strip()
        if out[name].isna().any() or not out[name].isin(allowed).all():
            raise ValueError(f"{name} contains a missing or unknown category; expected {sorted(allowed)}.")
    no_phone = out["PhoneService"].eq("No")
    if (no_phone.ne(out["MultipleLines"].eq("No phone service"))).any():
        raise ValueError("PhoneService and MultipleLines are inconsistent.")
    no_internet = out["InternetService"].eq("No")
    for name in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        if no_internet.ne(out[name].eq("No internet service")).any():
            raise ValueError(f"InternetService and {name} are inconsistent.")
    return out


def validate_target(target: pd.Series) -> pd.Series:
    labels = target.astype("string").str.strip()
    if labels.isna().any() or not labels.isin(["Yes", "No"]).all():
        raise ValueError("Churn must contain only Yes or No, without missing values.")
    return labels


def validate_dataset(df: pd.DataFrame, *, require_target: bool) -> pd.DataFrame:
    out = clean_customer_data(df)
    if "id" in out:
        ids = _required_numeric(out["id"], "id")
        if ids.lt(0).any() or ids.mod(1).ne(0).any() or not ids.is_unique:
            raise ValueError("id must contain unique, non-negative integers.")
        out["id"] = ids.astype("int64")
    if require_target and "Churn" not in out:
        raise ValueError("Training data must include a 'Churn' column.")
    if "Churn" in out:
        out["Churn"] = validate_target(out["Churn"])
    return out
