"""Compare fixed models on a holdout, then refit a separate calibrated final model."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import sys
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
import xgboost
from matplotlib.figure import Figure
from src.artifacts import file_sha256, output_path, record_path
from src.config import load_config, resolve_path
from src.data import load_train_data
from src.evaluation import compute_metrics, threshold_metrics, calibration_bins
from src.models import (build_lr_pipeline, build_rf_pipeline, build_xgb_pipeline,
    build_production_pipeline, build_calibrated_model, compute_scale_pos_weight, get_feature_pipeline)
from src.preprocessing import prepare_xy, split_train_test

DEPLOYED_RECIPE = "xgboost_fs_sigmoid"


def _fold_assignments(X, y, folds, seed):
    result = np.full(len(X), -1, dtype=int)
    for fold, (_, validation) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y)):
        result[validation] = fold
    return result


def _feature_state(model):
    pipeline = get_feature_pipeline(model)
    features = pipeline.named_steps["features"]
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    selected = names[pipeline.named_steps["feature_selection"].get_support()]
    return {"monthly_median": features.monthly_median_, "tenure_median": features.tenure_median_,
        "encoded_count": len(names), "selected_count": len(selected),
        "senior_citizen_encoded": "num__SeniorCitizen" in names,
        "senior_citizen_selected": "num__SeniorCitizen" in selected,
        "selected_features": selected.tolist(),
        "scale_pos_weight": pipeline.named_steps["classifier"].get_params()["scale_pos_weight"]}


def train(config_path: str | Path | None = None) -> dict:
    config = load_config(config_path)
    settings = config.get("evaluation", {})
    seed = int(config.get("random_state", 42)); folds = int(settings.get("calibration_folds", 3))
    threshold = float(settings.get("decision_threshold", .5))
    risk = config.get("risk_thresholds", {"low": .3, "high": .6})
    low, high = float(risk.get("low", .3)), float(risk.get("high", .6))
    thresholds = sorted(set(map(float, settings.get("thresholds", [.3, .5, .6]))) | {threshold, low, high})
    if not 0 < low < high < 1 or any(not 0 < t < 1 for t in thresholds):
        raise ValueError("Decision/risk thresholds must lie in (0, 1), with low < high.")
    train_path = resolve_path(config, "train"); input_sha = file_sha256(train_path)
    config_sha = file_sha256(config["_config_path"])
    df = load_train_data(train_path)
    if "id" not in df:
        raise ValueError("Training CSV requires a unique id for traceable split/export identities.")
    X, y = prepare_xy(df)
    X_fit, X_holdout, y_fit, y_holdout = split_train_test(X, y, test_size=config.get("test_size", .2), random_state=seed)
    if folds < 2 or y_fit.value_counts().min() < folds:
        raise ValueError("Each fit-set class must have at least calibration_folds rows; folds >= 2.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + input_sha[:8]
    run_dir = resolve_path(config, "metrics").parent / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config["_config_path"], run_dir / "config.yaml")
    for name in ["requirements.lock.txt", ".python-version"]:
        path = Path(config["_project_root"]) / name
        if path.is_file(): shutil.copy2(path, run_dir / name)
    source_hashes = {p.relative_to(Path(config["_project_root"])).as_posix(): file_sha256(p) for p in sorted((Path(config["_project_root"]) / "src").glob("*.py"))}
    print(f"Run {run_id}: fit={len(X_fit):,}, holdout={len(X_holdout):,}; calibration={folds}-fold OOF", flush=True)
    weight = compute_scale_pos_weight(y_fit)
    rf_settings = {"n_estimators": 150, "max_depth": 12, "min_samples_leaf": 10, **settings.get("random_forest", {})}
    candidates = {
        "logistic_regression": build_lr_pipeline(random_state=seed),
        "random_forest": build_rf_pipeline(random_state=seed, **rf_settings),
        "xgboost_no_fs": build_xgb_pipeline(weight, random_state=seed, classifier_params=config.get("xgb_best_params", {})),
    }
    probabilities = {}
    for name, model in candidates.items():
        print(f"Fit comparison: {name}", flush=True); model.fit(X_fit, y_fit)
        probabilities[name] = model.predict_proba(X_holdout)[:, 1]
    print("Fit evaluation model and OOF sigmoid calibration (holdout excluded)", flush=True)
    evaluation_model = build_calibrated_model(build_production_pipeline(weight, config), folds, seed)
    evaluation_model.fit(X_fit, y_fit)
    probabilities["xgboost_fs_raw"] = get_feature_pipeline(evaluation_model).predict_proba(X_holdout)[:, 1]
    probabilities[DEPLOYED_RECIPE] = evaluation_model.predict_proba(X_holdout)[:, 1]
    comparison, threshold_rows, curves, calibration_summary = [], [], [], {}
    for name, scores in probabilities.items():
        values = compute_metrics(y_holdout, scores >= threshold, scores)
        bins, ece = calibration_bins(y_holdout, scores)
        comparison.append({"model": name, "fit_rows": len(X_fit), "holdout_rows": len(X_holdout), "threshold": threshold, **values})
        threshold_rows.extend({"model": name, **row} for row in threshold_metrics(y_holdout, scores, thresholds))
        curves.append(bins.assign(model=name))
        calibration_summary[name] = {"brier": values["brier"], "log_loss": values["log_loss"], "ece_10_equal_width": ece,
            "mean_score": float(np.mean(scores)), "actual_rate": float(y_holdout.mean())}
    holdout = pd.DataFrame({"id": df.loc[X_holdout.index, "id"].to_numpy(), "source_row": X_holdout.index,
        "Churn": df.loc[X_holdout.index, "Churn"].to_numpy(), "actual": y_holdout.to_numpy(),
        "raw_score": probabilities["xgboost_fs_raw"], "calibrated_score": probabilities[DEPLOYED_RECIPE],
        "prediction_scope": "holdout", "run_id": run_id})
    assignments = pd.DataFrame({"id": df.id, "source_row": df.index, "partition": "fit"})
    assignments.loc[X_holdout.index, "partition"] = "holdout"
    assignments["evaluation_calibration_fold"] = -1
    assignments.loc[X_fit.index, "evaluation_calibration_fold"] = _fold_assignments(X_fit, y_fit, folds, seed)
    assignments["final_calibration_fold"] = _fold_assignments(X, y, folds, seed)
    tables = {"model_comparison": pd.DataFrame(comparison), "threshold_metrics": pd.DataFrame(threshold_rows),
        "calibration_curve": pd.concat(curves, ignore_index=True), "holdout_predictions": holdout, "split_assignments": assignments}
    for name, table in tables.items(): table.to_csv(run_dir / f"{name}.csv", index=False, float_format="%.12g")
    figure = Figure(figsize=(7, 5)); axis = figure.subplots()
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for name in ["xgboost_fs_raw", DEPLOYED_RECIPE]:
        table = tables["calibration_curve"].query("model == @name")
        axis.plot(table.mean_score, table.actual_rate, "o-", label=name)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted score", ylabel="Observed churn fraction", title=f"Holdout calibration | n={len(X_holdout):,}")
    axis.legend(); axis.grid(alpha=.2); figure.tight_layout(); figure.savefig(run_dir / "calibration.png", dpi=150)
    joblib.dump(evaluation_model, run_dir / "evaluation_model.joblib")
    print("Refit final model: OOF calibration on all labeled rows, then base fit on all rows", flush=True)
    final_model = build_calibrated_model(build_production_pipeline(compute_scale_pos_weight(y), config), folds, seed)
    final_model.fit(X, y); final_model.run_id_ = run_id; final_model.prediction_scope_ = "final_model"
    joblib.dump(final_model, run_dir / "final_model.joblib")
    if file_sha256(train_path) != input_sha: raise ValueError("Training CSV changed during the run; rerun required.")
    if file_sha256(config["_config_path"]) != config_sha: raise ValueError("Config changed during the run; rerun required.")
    current_hashes = {p.relative_to(Path(config["_project_root"])).as_posix(): file_sha256(p) for p in sorted((Path(config["_project_root"]) / "src").glob("*.py"))}
    if current_hashes != source_hashes: raise ValueError("Training source changed during the run; rerun required.")
    artifacts = {}
    for path in sorted(run_dir.iterdir()):
        if path.stem == "final_model": role = "deployment; all labeled rows"
        elif path.stem == "evaluation_model": role = "outer-fit model; holdout excluded from fitting"
        elif path.name in {"config.yaml", "requirements.lock.txt", ".python-version"}: role = "configuration/runtime snapshot"
        elif path.stem == "split_assignments": role = "outer partition and calibration fold identity"
        else: role = "outer-holdout evaluation evidence"
        artifacts[path.stem] = {"path": record_path(config, path), "sha256": file_sha256(path), "role": role}
    current_outputs = {}
    for key in tables:
        destination = output_path(config, key, f"{key}.csv"); destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / f"{key}.csv", destination); current_outputs[key] = record_path(config, destination)
    destination = output_path(config, "calibration_figure", "calibration.png")
    destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(run_dir / "calibration.png", destination)
    current_outputs["calibration"] = record_path(config, destination)
    model_path = resolve_path(config, "model"); model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "final_model.joblib", model_path)
    result = {
        "schema_version": 2, "run_id": run_id, "model": DEPLOYED_RECIPE, "random_state": seed,
        "test_size": config.get("test_size", .2), "n_samples": len(X), "n_train": len(X_fit), "n_holdout": len(X_holdout),
        "input_data": {"path": str(train_path.resolve()), "sha256": input_sha},
        "config_file": {"path": config["_config_path"], "sha256": config_sha},
        "evaluation_method": {"split": "stratified random holdout", "default_threshold": threshold,
            "selected_thresholds": thresholds, "threshold_selection": "prespecified; no holdout optimization or cost claim",
            "model_selection": "fixed xgboost_fs_sigmoid recipe; comparison is diagnostic, not holdout-based selection",
            "comparison_population": "same full outer-fit set and outer-holdout set for every row",
            "historical_tuning": "Original Colab search/holdout reuse cannot be reconstructed; frozen parameters are not proven optimal."},
        "calibration": {"method": "sigmoid", "folds": folds, "ensemble": False,
            "fit": "OOF scores within fit set; base then refit on that full fit set", "holdout_used_for_fit": False},
        "risk_thresholds": {"low": low, "high": high},
        "evaluation_scale_pos_weight": _feature_state(evaluation_model)["scale_pos_weight"],
        "scale_pos_weight": _feature_state(final_model)["scale_pos_weight"],
        "feature_engineering": {"evaluation": _feature_state(evaluation_model), "production": _feature_state(final_model)},
        "holdout_metrics": next(row for row in comparison if row["model"] == DEPLOYED_RECIPE),
        "uncalibrated_holdout_metrics": next(row for row in comparison if row["model"] == "xgboost_fs_raw"),
        "model_comparison": comparison, "threshold_metrics": threshold_rows, "calibration_metrics": calibration_summary,
        "parameters": {"xgboost": config.get("xgb_best_params", {}), "feature_selection": config.get("feature_selection", {}), "random_forest": rf_settings},
        "environment": {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "source_sha256": source_hashes, "artifacts": artifacts, "current_outputs": current_outputs,
        "final_model_path": record_path(config, model_path), "run_manifest": record_path(config, run_dir / "metrics.json"),
    }
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (run_dir / "metrics.json").write_text(payload, encoding="utf-8")
    metrics_path = resolve_path(config, "metrics"); metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(payload, encoding="utf-8")
    print(f"Completed {run_id}; calibrated holdout AUC={result['holdout_metrics']['roc_auc']:.8f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", default=None)
    print(json.dumps(train(parser.parse_args().config)["holdout_metrics"], indent=2))


if __name__ == "__main__": main()
