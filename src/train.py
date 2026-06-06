"""CLI to fit the production pipeline and write model artifact + metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from src.config import load_config, resolve_path
from src.data import load_train_data
from src.evaluation import compute_metrics
from src.models import build_production_pipeline, compute_scale_pos_weight
from src.preprocessing import prepare_xy, split_train_test


def train(config_path: str | Path | None = None) -> dict[str, Any]:
    """Fit production pipeline on full data; report holdout metrics from a split."""
    config = load_config(config_path)

    df = load_train_data()
    X, y = prepare_xy(df)
    if y is None:
        raise ValueError("Training data must include a 'Churn' column.")

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
    )

    scale_pos_weight = compute_scale_pos_weight(y)

    eval_pipeline = build_production_pipeline(scale_pos_weight, config)
    eval_pipeline.fit(X_train, y_train)
    holdout_metrics = compute_metrics(
        y_test,
        eval_pipeline.predict(X_test),
        eval_pipeline.predict_proba(X_test)[:, 1],
    )

    production_pipeline = build_production_pipeline(scale_pos_weight, config)
    production_pipeline.fit(X, y)

    model_path = resolve_path(config, "model")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(production_pipeline, model_path)

    result = {
        "model": "xgboost_select_from_model",
        "random_state": config.get("random_state", 42),
        "test_size": config.get("test_size", 0.2),
        "n_samples": len(X),
        "scale_pos_weight": scale_pos_weight,
        "holdout_metrics": holdout_metrics,
        "xgb_params": config.get("xgb_best_params", {}),
    }

    metrics_path = resolve_path(config, "metrics")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train churn XGBoost pipeline and save artifact + metrics."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: project root config.yaml)",
    )
    args = parser.parse_args()

    result = train(args.config)
    metrics = result["holdout_metrics"]
    model_path = load_config(args.config)["paths"]["model"]

    print("Holdout evaluation:")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
