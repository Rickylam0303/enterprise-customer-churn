"""Historical holdout KPIs and sample-qualified action segments."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.features import make_tenure_groups


def summarize_holdout(profiles, predictions, min_count=1000):
    if not isinstance(min_count, int) or min_count < 1:
        raise ValueError("min_segment_count must be a positive integer.")
    if predictions.empty or not predictions.prediction_scope.eq("holdout").all():
        raise ValueError("Historical BI requires non-empty holdout predictions.")
    if not predictions.id.is_unique or not profiles.id.is_unique:
        raise ValueError("BI source and prediction IDs must be unique.")
    merged = profiles.merge(predictions, on="id", how="inner", validate="one_to_one", suffixes=("", "_scored"))
    if len(merged) != len(predictions) or not merged.Churn.eq(merged.Churn_scored).all():
        raise ValueError("BI scores must match source IDs and labels.")
    merged["tenure_bucket"] = make_tenure_groups(merged.tenure, ["0-6", "6-12", "12-24", "24-60", "60+"])
    merged["actual"] = merged.Churn.eq("Yes").astype(int)
    merged["high"] = merged.risk_band.eq("High").astype(int)
    segments = merged.groupby(["Contract", "tenure_bucket", "InternetService"], observed=True).agg(
        customer_count=("id", "size"), churned_count=("actual", "sum"),
        observed_churn_rate=("actual", "mean"), mean_predicted_score=("predicted_proba", "mean"),
        predicted_high_risk_count=("high", "sum"),
    ).reset_index()
    segments["eligible_for_action"] = segments.customer_count.ge(min_count)
    segments["min_segment_count"] = min_count
    segments["prediction_scope"] = "holdout"
    actions = segments[segments.eligible_for_action].sort_values(
        ["observed_churn_rate", "customer_count", "Contract", "tenure_bucket", "InternetService"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    actions.insert(0, "action_rank", np.arange(1, len(actions) + 1))
    kpis = {
        "Evaluation Population": len(merged), "Historical Churned Customers": int(merged.actual.sum()),
        "Historical Churn Rate": float(merged.actual.mean()),
        "Predicted High-Risk Customers": int(merged.high.sum()),
        "Predicted High-Risk Rate": float(merged.high.mean()),
        "Mean Calibrated Churn Score": float(merged.predicted_proba.mean()),
        "min_segment_count": min_count, "nonempty_segments": len(segments),
        "eligible_segments": len(actions), "excluded_small_segments": int((~segments.eligible_for_action).sum()),
    }
    return kpis, segments, actions
