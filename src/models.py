"""Model pipeline builders for churn classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor
from src.features import FeatureEngineer


def get_feature_pipeline(model) -> Pipeline:
    """Unwrap the single full-fit estimator in ensemble=False calibration."""
    if isinstance(model, CalibratedClassifierCV):
        calibrated = getattr(model, "calibrated_classifiers_", [])
        if len(calibrated) != 1:
            raise ValueError("Expected one full-fit estimator (calibration ensemble=False).")
        return calibrated[0].estimator
    return model


def require_raw_feature_pipeline(model) -> None:
    """Reject artifacts lacking fitted raw features or the SeniorCitizen input."""
    pipeline = get_feature_pipeline(model)
    steps = getattr(pipeline, "named_steps", {})
    preprocessor = steps.get("preprocessor")
    numeric = next((cols for name, _, cols in getattr(preprocessor, "transformers", []) if name == "num"), [])
    if not isinstance(steps.get("features"), FeatureEngineer) or "SeniorCitizen" not in numeric:
        raise ValueError(
            "Legacy model artifact lacks the current raw-profile/SeniorCitizen schema. "
            "Retrain with python -m src.train, then rerun python -m src.export_predictions."
        )


def compute_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Ratio of negative to positive class samples for XGBoost."""
    y_arr = np.asarray(y)
    positives = y_arr.sum()
    negatives = len(y_arr) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Both churn classes are required for training.")
    return float(negatives / positives)


class TrainingWeightedXGBClassifier(XGBClassifier):
    """Recompute class weight from each fit's labels, including calibration folds."""

    def fit(self, X, y, **kwargs):
        self.set_params(scale_pos_weight=compute_scale_pos_weight(y))
        return super().fit(X, y, **kwargs)


def build_calibrated_model(pipeline: Pipeline, folds: int = 3, random_state: int = 42):
    """Learn sigmoid calibration from OOF training scores; refit base on all fit rows."""
    if folds < 2:
        raise ValueError("Calibration requires at least two stratified folds.")
    return CalibratedClassifierCV(
        estimator=pipeline, method="sigmoid",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state),
        ensemble=False, n_jobs=1,
    )


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
    return TrainingWeightedXGBClassifier(**params)


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

    steps: list[tuple[str, Any]] = [("features", FeatureEngineer()), ("preprocessor", preprocessor)]
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
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
) -> Pipeline:
    """Random forest baseline pipeline."""
    preprocessor = preprocessor or build_preprocessor()
    classifier = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_estimators=n_estimators,
        n_jobs=-1,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )

    steps: list[tuple[str, Any]] = [("features", FeatureEngineer()), ("preprocessor", preprocessor)]
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

    steps: list[tuple[str, Any]] = [("features", FeatureEngineer()), ("preprocessor", preprocessor)]
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
    return {
        "lr": build_lr_pipeline(random_state=random_state),
        "rf": build_rf_pipeline(random_state=random_state),
        "xgb": build_xgb_pipeline(
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
        ),
    }


def build_production_pipeline(
    scale_pos_weight: float,
    config: dict[str, Any] | None = None,
) -> Pipeline:
    """Raw profiles → fitted features → preprocess → select → tuned XGBoost."""
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
