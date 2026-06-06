"""Model pipeline builders for churn classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor


def compute_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Ratio of negative to positive class samples for XGBoost."""
    y_arr = np.asarray(y)
    positives = y_arr.sum()
    negatives = len(y_arr) - positives
    return float(negatives / positives)


def _xgb_classifier(
    scale_pos_weight: float,
    random_state: int = 42,
    **kwargs: Any,
) -> XGBClassifier:
    """Create an XGBClassifier with project defaults."""
    params = {
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
        "eval_metric": "logloss",
        "n_jobs": -1,
    }
    params.update(kwargs)
    return XGBClassifier(**params)


def build_lr_pipeline(
    preprocessor=None,
    random_state: int = 42,
    with_feature_selection: bool = False,
) -> Pipeline:
    """Logistic regression baseline pipeline."""
    preprocessor = preprocessor or build_preprocessor()
    classifier = LogisticRegression(
        class_weight="balanced", random_state=random_state, max_iter=1000
    )

    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
    if with_feature_selection:
        selector = SelectFromModel(
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=0.1,
                class_weight="balanced",
                random_state=random_state,
                max_iter=1000,
            ),
            threshold=1e-5,
        )
        steps.append(("feature_selection", selector))
    steps.append(("classifier", classifier))
    return Pipeline(steps=steps)


def build_rf_pipeline(
    preprocessor=None,
    random_state: int = 42,
    n_estimators: int = 350,
    with_feature_selection: bool = False,
) -> Pipeline:
    """Random forest baseline pipeline."""
    preprocessor = preprocessor or build_preprocessor()
    classifier = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_estimators=n_estimators,
        n_jobs=-1,
    )

    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
    if with_feature_selection:
        selector = SelectFromModel(
            RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state,
                n_estimators=100,
                max_depth=10,
                n_jobs=-1,
            ),
            threshold="median",
        )
        steps.append(("feature_selection", selector))
    steps.append(("classifier", classifier))
    return Pipeline(steps=steps)


def build_xgb_pipeline(
    scale_pos_weight: float,
    preprocessor=None,
    random_state: int = 42,
    with_feature_selection: bool = False,
    feature_selection_cfg: dict[str, Any] | None = None,
    classifier_params: dict[str, Any] | None = None,
) -> Pipeline:
    """XGBoost pipeline with optional feature selection."""
    preprocessor = preprocessor or build_preprocessor()
    feature_selection_cfg = feature_selection_cfg or {}
    classifier_params = classifier_params or {}

    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
    if with_feature_selection:
        selector = SelectFromModel(
            _xgb_classifier(
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                n_estimators=feature_selection_cfg.get("n_estimators", 100),
                max_depth=feature_selection_cfg.get("max_depth", 5),
                importance_type="gain",
            ),
            threshold=feature_selection_cfg.get("threshold", "median"),
        )
        steps.append(("feature_selection", selector))

    steps.append(
        (
            "classifier",
            _xgb_classifier(
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                **classifier_params,
            ),
        )
    )
    return Pipeline(steps=steps)


def build_baseline_pipelines(
    scale_pos_weight: float,
    config: dict[str, Any] | None = None,
) -> dict[str, Pipeline]:
    """Return unfitted LR, RF, and XGB baselines without feature selection."""
    if config is None:
        from src.config import load_config

        config = load_config()

    random_state = config.get("random_state", 42)
    preprocessor = build_preprocessor()
    return {
        "lr": build_lr_pipeline(preprocessor=preprocessor, random_state=random_state),
        "rf": build_rf_pipeline(preprocessor=preprocessor, random_state=random_state),
        "xgb": build_xgb_pipeline(
            scale_pos_weight=scale_pos_weight,
            preprocessor=preprocessor,
            random_state=random_state,
        ),
    }


def build_production_pipeline(
    scale_pos_weight: float,
    config: dict[str, Any] | None = None,
) -> Pipeline:
    """Full production pipeline: preprocess → feature selection → tuned XGBoost."""
    if config is None:
        from src.config import load_config

        config = load_config()

    return build_xgb_pipeline(
        scale_pos_weight=scale_pos_weight,
        random_state=config.get("random_state", 42),
        with_feature_selection=True,
        feature_selection_cfg=config.get("feature_selection", {}),
        classifier_params=config.get("xgb_best_params", {}),
    )


def get_randomized_search_param_grid() -> dict[str, list[Any]]:
    """Hyperparameter distributions used in the original Colab RandomizedSearchCV."""
    return {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [3, 5, 7, 10],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.6, 0.8, 1.0],
        "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        "classifier__gamma": [0, 0.1, 0.2],
        "classifier__reg_alpha": [0, 0.1, 1],
        "classifier__reg_lambda": [1, 1.5, 2],
    }
