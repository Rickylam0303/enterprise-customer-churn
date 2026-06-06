"""Custom feature engineering for churn prediction."""

from __future__ import annotations

import pandas as pd

SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain features used by the production pipeline."""
    out = df.copy()

    out["avg_monthly_spend"] = out["TotalCharges"] / (out["tenure"] + 1)

    out["tenure_group"] = pd.cut(
        out["tenure"],
        bins=[0, 6, 12, 24, 60, 100],
        labels=["0-6", "6-12", "1-2y", "2-5y", "5y+"],
    )

    out["service_count"] = out[SERVICE_COLS].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    out["mtm_early"] = (
        (out["Contract"] == "Month-to-month") & (out["tenure"] < 12)
    ).astype(int)

    out["electronic_check"] = (out["PaymentMethod"] == "Electronic check").astype(int)

    median_monthly = out["MonthlyCharges"].median()
    median_tenure = out["tenure"].median()
    out["high_value"] = (
        (out["MonthlyCharges"] > median_monthly) & (out["tenure"] > median_tenure)
    ).astype(int)

    out["missing_online_sec"] = (
        (out["InternetService"] != "No") & (out["OnlineSecurity"] == "No")
    ).astype(int)

    out["tenure_x_monthly"] = out["tenure"] * out["MonthlyCharges"]

    return out
