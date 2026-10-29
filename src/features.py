"""Custom feature engineering for churn prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.schema import clean_customer_data, validate_tenure

TENURE_BINS = [0, 6, 12, 24, 60, 100]
TENURE_LABELS = ["0-6", "6-12", "1-2y", "2-5y", "5y+"]


def make_tenure_groups(tenure: pd.Series, labels: list[str] | None = None) -> pd.Series:
    """Use [0, 6], (6, 12], (12, 24], (24, 60], (60, 100]."""
    return pd.cut(
        validate_tenure(tenure),
        bins=TENURE_BINS,
        labels=TENURE_LABELS if labels is None else labels,
        include_lowest=True,
    )

SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def engineer_features(
    df: pd.DataFrame, *, monthly_median: float, tenure_median: float,
) -> pd.DataFrame:
    """Add features using explicitly supplied training statistics, never batch medians."""
    out = df.copy()
    out["tenure"] = validate_tenure(out["tenure"])

    out["avg_monthly_spend"] = out["TotalCharges"] / (out["tenure"] + 1)

    out["tenure_group"] = make_tenure_groups(out["tenure"])

    out["service_count"] = out[SERVICE_COLS].eq("Yes").sum(axis=1)

    out["mtm_early"] = (
        (out["Contract"] == "Month-to-month") & (out["tenure"] < 12)
    ).astype(int)

    out["electronic_check"] = (out["PaymentMethod"] == "Electronic check").astype(int)

    out["high_value"] = (
        (out["MonthlyCharges"] > monthly_median) & (out["tenure"] > tenure_median)
    ).astype(int)

    out["missing_online_sec"] = (
        (out["InternetService"] != "No") & (out["OnlineSecurity"] == "No")
    ).astype(int)

    out["tenure_x_monthly"] = out["tenure"] * out["MonthlyCharges"]

    return out


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Clean raw profiles and learn high_value cutoffs exclusively during fit.

    The fitted instance is serialized with each model. transform recomputes all
    derived columns, so caller-supplied derived features cannot override fit state.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        cleaned = clean_customer_data(X)
        tenure_median = cleaned["tenure"].median()
        monthly_median = cleaned["MonthlyCharges"].median()
        if pd.isna(tenure_median) or pd.isna(monthly_median):
            raise ValueError("Cannot learn feature statistics from empty or missing data.")
        self.tenure_median_ = float(tenure_median)
        self.monthly_median_ = float(monthly_median)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["monthly_median_", "tenure_median_"])
        return engineer_features(
            clean_customer_data(X),
            monthly_median=self.monthly_median_,
            tenure_median=self.tenure_median_,
        )
