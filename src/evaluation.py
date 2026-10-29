"""Model evaluation metrics and reporting helpers."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.pipeline import Pipeline


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Return classification metrics as a dictionary."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
    }


def evaluate_model(
    model: Pipeline | ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    *,
    plot_confusion_matrix: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate a fitted model and optionally print results and plot confusion matrix."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    if verbose:
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

    if plot_confusion_matrix:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.show()

    return metrics


def get_feature_importances(
    pipeline: Pipeline,
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract ranked feature importances from a fitted XGB/RF production pipeline."""
    from src.models import get_feature_pipeline
    pipeline = get_feature_pipeline(pipeline)
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    if "feature_selection" in pipeline.named_steps:
        support = pipeline.named_steps["feature_selection"].get_support()
        feature_names = feature_names[support]

    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_[0])
    else:
        raise ValueError("Classifier does not expose importances or coefficients.")

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": values}
    ).sort_values("importance", ascending=False)

    return importance_df.head(top_n).reset_index(drop=True)


def threshold_metrics(y, probabilities, thresholds):
    """Evaluate prespecified thresholds; never optimize them on this dataset."""
    rows = []
    for threshold in thresholds:
        predicted = np.asarray(probabilities) >= threshold
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        rows.append({
            "threshold": float(threshold), "n": len(y), "selected": int(predicted.sum()),
            "workload": float(predicted.mean()), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
        })
    return rows


def calibration_bins(y, probabilities, n_bins=10):
    frame = pd.DataFrame({"actual": np.asarray(y), "score": np.asarray(probabilities)})
    frame["bin"] = pd.cut(frame.score, np.linspace(0, 1, n_bins + 1), include_lowest=True)
    grouped = frame.groupby("bin", observed=True).agg(
        n=("actual", "size"), mean_score=("score", "mean"), actual_rate=("actual", "mean"),
    ).reset_index()
    grouped["bin"] = grouped["bin"].astype(str)
    ece = float((grouped.n * (grouped.mean_score - grouped.actual_rate).abs()).sum() / len(frame))
    return grouped, ece
