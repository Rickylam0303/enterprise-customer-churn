"""Export slim train predictions CSV for Power BI ML bridge."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import load_config, resolve_path
from src.data import load_train_data
from src.preprocessing import prepare_xy


def assign_risk_band(probability: float, thresholds: dict[str, float]) -> str:
    """Map churn probability to Low / Medium / High (matches Streamlit demo)."""
    low = thresholds.get("low", 0.30)
    high = thresholds.get("high", 0.60)
    if probability < low:
        return "Low"
    if probability < high:
        return "Medium"
    return "High"


def export_train_predictions(
    config_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score full training set and write id, Churn, predicted_proba, risk_band CSV."""
    config = load_config(config_path)
    df = load_train_data()
    model_path = resolve_path(config, "model")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python -m src.train` first."
        )

    pipeline = joblib.load(model_path)
    X, _ = prepare_xy(df)
    probabilities = pipeline.predict_proba(X)[:, 1]
    thresholds = config.get("risk_thresholds", {})

    export_df = pd.DataFrame(
        {
            "id": df["id"].astype(int),
            "Churn": df["Churn"],
            "predicted_proba": probabilities.round(6),
            "risk_band": [
                assign_risk_band(float(proba), thresholds) for proba in probabilities
            ],
        }
    )

    output_path = resolve_path(config, "bi_predictions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False)

    summary = _summarize_export(export_df, thresholds)
    summary["output_path"] = str(output_path)
    summary["n_rows"] = len(export_df)
    return export_df, summary


def _summarize_export(df: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, Any]:
    band_counts = df["risk_band"].value_counts().to_dict()
    return {
        "risk_thresholds": thresholds,
        "risk_band_counts": band_counts,
        "avg_predicted_proba": float(df["predicted_proba"].mean()),
        "actual_churn_rate": float((df["Churn"] == "Yes").mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export train predictions for Power BI (id, Churn, proba, risk_band)."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: project root config.yaml)",
    )
    args = parser.parse_args()

    _, summary = export_train_predictions(args.config)
    print(f"Exported {summary['n_rows']:,} rows to {summary['output_path']}")
    print(f"  Avg predicted proba: {summary['avg_predicted_proba']:.4f}")
    print(f"  Actual churn rate:   {summary['actual_churn_rate']:.1%}")
    print("  Risk band counts:")
    for band in ("Low", "Medium", "High"):
        count = summary["risk_band_counts"].get(band, 0)
        print(f"    {band}: {count:,}")


if __name__ == "__main__":
    main()
