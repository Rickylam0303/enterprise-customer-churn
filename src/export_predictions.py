"""Export historical holdout scores separately from final-model in-sample scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.artifacts import file_sha256, output_path, record_path, verify_artifact
from src.config import load_config, resolve_path
from src.data import load_train_data
from src.models import require_raw_feature_pipeline
from src.preprocessing import prepare_xy
from src.bi import summarize_holdout


def assign_risk_band(probability: float, thresholds: dict[str, float]) -> str:
    low, high = thresholds.get("low", .3), thresholds.get("high", .6)
    return "Low" if probability < low else "Medium" if probability < high else "High"


def export_train_predictions(config_path=None, *, scope="holdout"):
    """Compatibility entry point; default BI export contains only unseen holdout rows."""
    if scope not in {"holdout", "final"}:
        raise ValueError("scope must be holdout or final.")
    config = load_config(config_path)
    metrics_path = resolve_path(config, "metrics")
    if not metrics_path.is_file():
        raise ValueError("No evaluation manifest. Retrain with python -m src.train before exporting.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("schema_version") != 2:
        raise ValueError("Legacy evaluation manifest. Retrain with python -m src.train.")
    if file_sha256(config["_config_path"]) != metrics["config_file"]["sha256"]:
        raise ValueError("Selected config differs from the evaluated run; retrain first.")
    train_path = resolve_path(config, "train")
    if file_sha256(train_path) != metrics["input_data"]["sha256"]:
        raise ValueError("Selected training CSV does not match this run's data checksum; retrain first.")
    df = load_train_data(train_path)
    thresholds = {"low": .3, "high": .6, **config.get("risk_thresholds", {})}
    if thresholds != metrics["risk_thresholds"]:
        raise ValueError("Risk thresholds differ from the evaluated run; rerun evaluation before exporting.")
    if scope == "holdout":
        scores = pd.read_csv(verify_artifact(config, metrics["artifacts"]["holdout_predictions"]))
        assignments = pd.read_csv(verify_artifact(config, metrics["artifacts"]["split_assignments"]))
        expected = set(assignments.loc[assignments.partition.eq("holdout"), "id"])
        if not scores.id.is_unique or set(scores.id) != expected:
            raise ValueError("Holdout score IDs do not match the recorded split.")
        actual = df.set_index("id").loc[scores.id, "Churn"].to_numpy()
        if not np.array_equal(actual, scores.Churn.to_numpy()):
            raise ValueError("Holdout labels do not match source data.")
        ids = scores.id.to_numpy(); probabilities = scores.calibrated_score.to_numpy()
        artifact = metrics["artifacts"]["evaluation_model"]
        prediction_scope = "holdout"
        path = resolve_path(config, "bi_predictions")
    else:
        artifact = metrics["artifacts"]["final_model"]
        model_path = resolve_path(config, "model")
        if file_sha256(model_path) != artifact["sha256"]:
            raise ValueError("Configured final model does not match the evaluation run.")
        model = joblib.load(model_path); require_raw_feature_pipeline(model)
        X, _ = prepare_xy(df)
        ids = df.id.to_numpy(); actual = df.Churn.to_numpy()
        probabilities = model.predict_proba(X)[:, 1]
        prediction_scope = "final_training_in_sample"
        path = output_path(config, "final_predictions", "final_model_predictions.csv")
    exported = pd.DataFrame({"id": ids, "Churn": actual, "predicted_proba": probabilities,
        "risk_band": [assign_risk_band(float(p), thresholds) for p in probabilities],
        "prediction_scope": prediction_scope, "model_version": artifact["sha256"][:12], "run_id": metrics["run_id"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(path, index=False, float_format="%.12g")
    summary = {"output_path": str(path), "n_rows": len(exported), "input_path": str(train_path.resolve()),
        "input_sha256": metrics["input_data"]["sha256"], "run_id": metrics["run_id"],
        "prediction_scope": prediction_scope, "model_artifact": artifact,
        "risk_thresholds": thresholds, "risk_band_counts": exported.risk_band.value_counts().to_dict(),
        "avg_predicted_proba": float(exported.predicted_proba.mean()),
        "actual_churn_rate": float(exported.Churn.eq("Yes").mean()), "sha256": file_sha256(path),
        "evaluation_use": scope == "holdout"}
    if scope == "holdout":
        minimum = config.get("bi", {}).get("min_segment_count", 1000)
        kpis, segments, actions = summarize_holdout(df, exported, minimum)
        kpis.update(run_id=metrics["run_id"], prediction_scope="holdout", model_version=artifact["sha256"][:12],
            risk_thresholds=thresholds, input_sha256=summary["input_sha256"])
        kpi_path = output_path(config, "bi_kpis", "bi_kpis.json")
        kpi_path.parent.mkdir(parents=True, exist_ok=True)
        kpi_path.write_text(json.dumps(kpis, indent=2) + "\n", encoding="utf-8")
        for key, table in [("bi_segments", segments), ("bi_action_segments", actions)]:
            table = table.assign(run_id=metrics["run_id"], model_version=artifact["sha256"][:12])
            target = output_path(config, key, key + ".csv"); target.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(target, index=False, float_format="%.12g")
            summary.setdefault("bi_artifacts", {})[key] = {"path": record_path(config, target), "sha256": file_sha256(target)}
        summary["bi_artifacts"]["bi_kpis"] = {"path": record_path(config, kpi_path), "sha256": file_sha256(kpi_path)}
        summary["bi_kpis"] = kpis
    path.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return exported, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--scope", choices=["holdout", "final"], default="holdout")
    args = parser.parse_args()
    _, summary = export_train_predictions(args.config, scope=args.scope)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
