"""Generate README model numbers exclusively from a verified evaluation run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.artifacts import recorded_path, verify_artifact, file_sha256
from src.config import load_config, resolve_path

START = "<!-- MODEL_RESULTS_START -->"
END = "<!-- MODEL_RESULTS_END -->"


def verify_run(config, result):
    if file_sha256(config["_config_path"]) != result["config_file"]["sha256"]:
        raise ValueError("Selected config differs from the evaluated run; rerun training.")
    manifest = recorded_path(config, result["run_manifest"])
    if json.loads(manifest.read_text(encoding="utf-8")) != result:
        raise ValueError("Current metrics and immutable run manifest differ.")
    for name, record in result["artifacts"].items():
        verify_artifact(config, record)
        if name in result["current_outputs"]:
            current = recorded_path(config, result["current_outputs"][name])
            if not current.is_file() or file_sha256(current) != record["sha256"]:
                raise ValueError(f"Current output differs from run artifact: {name}")
    if file_sha256(resolve_path(config, "model")) != result["artifacts"]["final_model"]["sha256"]:
        raise ValueError("Final model differs from this run.")
    for name, expected in result["source_sha256"].items():
        if file_sha256(Path(config["_project_root"]) / name) != expected:
            raise ValueError(f"Training source changed since evaluation: {name}; rerun training.")


def model_results(config, result):
    def link(key, title):
        return f'[{title}]({Path(result["current_outputs"][key]).as_posix()})'
    method = result["evaluation_method"]
    lines = [START, "## Reproducible model results", "",
        f"Run `{result['run_id']}`. **{result['n_train']:,} fit / {result['n_holdout']:,} holdout** rows; "
        f"stratified {1-result['test_size']:.0%}/{result['test_size']:.0%}, seed {result['random_state']}. "
        f"Every comparison uses this same full-data split. Classification metrics below use score ≥ {method['default_threshold']:.2f}.", "",
        f"Calibration: sigmoid on {result['calibration']['folds']}-fold out-of-fold predictions **inside the fit partition only**, "
        "then refit the base on that full fit partition (`ensemble=False`). Holdout labels never fit the feature statistics, "
        "selector, class weights or calibrator in this run. The XGBoost + FS + sigmoid recipe was fixed before this comparison; "
        "the holdout is not used to pick the winner or optimize thresholds. Historical Colab tuning records are unavailable, "
        "so this is not a claim of a never-before-inspected benchmark or proven optimal hyperparameters.", "",
        "| Model | ROC-AUC | Precision | Recall | F1 | Brier ↓ | Log loss ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["model_comparison"]:
        lines.append(f"| {row['model']} | {row['roc_auc']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['brier']:.6f} | {row['log_loss']:.6f} |")
    rf = result["parameters"]["random_forest"]
    lines += ["", f"Source: {link('model_comparison','model_comparison.csv')}; exact values, parameters, environment and hashes: "
        f"[metrics.json]({Path(config['paths']['metrics']).relative_to(Path(config['_project_root'])).as_posix() if Path(config['paths']['metrics']).is_relative_to(Path(config['_project_root'])) else Path(config['paths']['metrics']).as_posix()}). "
        f"RF is a capacity-limited baseline ({rf['n_estimators']} trees, max_depth={rf['max_depth']}, min_samples_leaf={rf['min_samples_leaf']}); "
        "XGBoost no-FS/FS use the same frozen classifier settings. FS and calibration differences are diagnostic single-split results, not significance claims.", "",
        "### Prespecified thresholds — calibrated holdout model", "",
        "| Threshold | Selected | Workload | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in result["threshold_metrics"]:
        if row["model"] == result["model"]:
            lines.append(f"| {row['threshold']:.2f} | {row['selected']:,} | {row['workload']:.2%} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} |")
    risk = result["risk_thresholds"]
    lines += ["", f"Source: {link('threshold_metrics','threshold_metrics.csv')}, including TP/FP/FN/TN. "
        f"App/BI bands: Low <{risk['low']:.2f}, Medium [{risk['low']:.2f}, {risk['high']:.2f}), High ≥{risk['high']:.2f}. "
        "These thresholds were configured before evaluation; no outreach cost, capacity optimum or retention ROI is established.", "",
        "### Calibration on the same holdout", "",
        "| Score | Brier ↓ | Log loss ↓ | ECE (10 equal-width bins) ↓ | Mean score | Observed churn |",
        "|---|---:|---:|---:|---:|---:|"]
    for name in ["xgboost_fs_raw", result["model"]]:
        row = result["calibration_metrics"][name]
        lines.append(f"| {name} | {row['brier']:.6f} | {row['log_loss']:.6f} | {row['ece_10_equal_width']:.6f} | {row['mean_score']:.2%} | {row['actual_rate']:.2%} |")
    lines += ["", f"Sources: {link('calibration_curve','calibration_curve.csv')} (per-bin sample counts and rates), "
        f"{link('holdout_predictions','holdout_predictions.csv')} (ID, label, raw/calibrated score), "
        f"{link('split_assignments','split_assignments.csv')} (outer partition and calibration folds). ECE depends on binning.", "",
        f"![Holdout calibration]({Path(result['current_outputs']['calibration']).as_posix()})", "",
        "The separately refitted **final model uses all labeled rows**, with its own OOF calibration and full-data feature statistics. "
        "Its scores on those same rows are in-sample and do not produce the metrics above. The unlabeled Kaggle test CSV provides no local test AUC.", END]
    return "\n".join(lines)


def update_readme(config_path=None, *, check=False, readme_path=None):
    config = load_config(config_path)
    result = json.loads(resolve_path(config, "metrics").read_text(encoding="utf-8"))
    verify_run(config, result)
    path = Path(readme_path) if readme_path else Path(config["_project_root"]) / "README.md"
    source = path.read_text(encoding="utf-8")
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError("README must contain exactly one model-results marker pair.")
    prefix, remainder = source.split(START); _, suffix = remainder.split(END)
    expected = prefix + model_results(config, result) + suffix
    if check:
        if source != expected: raise ValueError("README model results are stale; run python -m src.reporting.")
    else:
        path.write_text(expected, encoding="utf-8")
    return result["run_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", default=None)
    parser.add_argument("--check", action="store_true"); parser.add_argument("--readme", default=None)
    args = parser.parse_args()
    print("README sources verified:", update_readme(args.config, check=args.check, readme_path=args.readme))


if __name__ == "__main__": main()
