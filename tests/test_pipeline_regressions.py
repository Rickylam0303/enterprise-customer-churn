"""Behavior regressions for F01 (fit state), F07 (paths), and F08 (boundaries).

Run: python -m unittest discover -s tests -v
Only temporary synthetic files are written; no private CSV or saved model is needed.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import yaml

from src.data import clean_total_charges, load_train_data, load_test_data
from src.export_predictions import export_train_predictions
from src.features import FeatureEngineer, make_tenure_groups
from src.models import build_lr_pipeline, require_raw_feature_pipeline
from src.preprocessing import prepare_xy, split_train_test
from src.train import train


def profiles(offset=0):
    rows = []
    for i in range(32):
        rows.append({
            "id": offset + i, "gender": "Male" if i % 2 else "Female",
            "SeniorCitizen": i % 2, "Partner": "Yes" if i % 2 else "No",
            "Dependents": "No", "tenure": (i * 3) % 73,
            "PhoneService": "Yes", "MultipleLines": "Yes" if i % 2 else "No",
            "InternetService": "Fiber optic" if i % 2 else "DSL",
            "OnlineSecurity": "Yes" if i % 3 else "No", "OnlineBackup": "No",
            "DeviceProtection": "Yes", "TechSupport": "No",
            "StreamingTV": "Yes" if i % 2 else "No", "StreamingMovies": "No",
            "Contract": "Month-to-month" if i % 3 else "Two year",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 20.0 + 2 * i,
            "TotalCharges": " " if i == 0 else str(20.0 + i * 12),
            "Churn": "Yes" if i % 4 == 0 else "No",
        })
    return pd.DataFrame(rows)


def write_config(root, name, offset=0):
    csv_path = root / f"{name}.csv"
    profiles(offset).to_csv(csv_path, index=False)
    cfg = {
        "paths": {
            "train": str(csv_path), "test": str(csv_path),
            "model": str(root / f"{name}.joblib"),
            "metrics": str(root / f"{name}-metrics.json"),
            "bi_predictions": str(root / f"{name}-predictions.csv"),
        },
        "random_state": 42, "test_size": 0.25,
        "xgb_best_params": {"n_estimators": 4, "max_depth": 2, "n_jobs": 1},
        "feature_selection": {"n_estimators": 3, "max_depth": 2, "threshold": "median"},
        "evaluation": {"calibration_folds": 2, "random_forest": {"n_estimators": 3, "max_depth": 2, "min_samples_leaf": 1}},
    }
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path, cfg


class FeatureStateTests(unittest.TestCase):
    def test_transform_requires_fit(self):
        with self.assertRaises(NotFittedError):
            FeatureEngineer().transform(profiles())

    def test_single_batch_order_and_serialization_are_identical(self):
        raw, y = prepare_xy(profiles())
        pipeline = build_lr_pipeline().fit(raw, y)
        batch = pipeline.named_steps["features"].transform(raw)
        singles = pd.concat([
            pipeline.named_steps["features"].transform(raw.loc[[i]]) for i in raw.index
        ])
        assert_frame_equal(batch, singles)
        expected = pipeline.predict_proba(raw)
        single_scores = np.vstack([pipeline.predict_proba(raw.loc[[i]]) for i in raw.index])
        np.testing.assert_allclose(expected, single_scores, rtol=0, atol=1e-12)
        np.testing.assert_allclose(expected, pipeline.predict_proba(raw.iloc[::-1])[::-1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(expected, np.vstack([pipeline.predict_proba(raw.iloc[:7]), pipeline.predict_proba(raw.iloc[7:])]), rtol=0, atol=1e-12)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(pipeline, path)
            restored = joblib.load(path)
            require_raw_feature_pipeline(restored)
            np.testing.assert_array_equal(expected, restored.predict_proba(raw))

    def test_scoring_batch_does_not_refit_or_trust_derived_columns(self):
        raw, _ = prepare_xy(profiles())
        transformer = FeatureEngineer().fit(raw.iloc[:8])
        medians = transformer.monthly_median_, transformer.tenure_median_
        scoring = raw.iloc[[-1]].copy()
        scoring["high_value"] = -999
        scoring["service_count"] = -999
        out = transformer.transform(scoring)
        self.assertEqual(int(out.high_value.iloc[0]), 1)
        self.assertGreaterEqual(int(out.service_count.iloc[0]), 0)
        self.assertEqual(medians, (transformer.monthly_median_, transformer.tenure_median_))
        self.assertNotEqual(transformer.monthly_median_, scoring.MonthlyCharges.median())
        self.assertFalse(hasattr(clone(transformer), "monthly_median_"))

    def test_only_total_charges_is_imputed(self):
        raw = profiles()
        raw.loc[0, "Partner"] = None
        cleaned = clean_total_charges(raw)
        self.assertEqual(cleaned.TotalCharges.iloc[0], 0)
        self.assertTrue(pd.isna(cleaned.Partner.iloc[0]))
        self.assertEqual(raw.TotalCharges.iloc[0], " ")

    def test_legacy_artifact_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Retrain"):
            require_raw_feature_pipeline(Pipeline([("preprocessor", StandardScaler())]))


class TenureTests(unittest.TestCase):
    def test_inclusive_zero_and_all_boundaries(self):
        months = pd.Series([0, 6, 7, 12, 13, 24, 25, 60, 61, 100])
        self.assertEqual(make_tenure_groups(months).tolist(), [
            "0-6", "0-6", "6-12", "6-12", "1-2y", "1-2y", "2-5y", "2-5y", "5y+", "5y+",
        ])

    def test_invalid_tenure_is_rejected_by_fit_and_transform(self):
        transformer = FeatureEngineer().fit(profiles())
        for value in [-1, 101, 1.5, None, float("inf"), "invalid"]:
            with self.subTest(value=value):
                raw = profiles().astype({"tenure": "object"})
                raw.loc[0, "tenure"] = value
                for operation in [FeatureEngineer().fit, transformer.transform]:
                    with self.assertRaisesRegex(ValueError, "whole months between 0 and 100"):
                        operation(raw)

    def test_zero_month_raw_profile_can_be_scored(self):
        raw, y = prepare_xy(profiles())
        pipe = build_lr_pipeline().fit(raw, y)
        self.assertEqual(pipe.named_steps["features"].transform(raw.iloc[[0]]).tenure_group.iloc[0], "0-6")
        self.assertTrue(np.isfinite(pipe.predict_proba(raw.iloc[[0]])).all())


class ConfigIntegrationTests(unittest.TestCase):
    def test_explicit_loader_paths_do_not_read_default_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, cfg = write_config(root, "chosen", offset=8000)
            with patch("src.data.load_config", side_effect=AssertionError("default config used")):
                for loader in [load_train_data, load_test_data]:
                    raw = loader(cfg["paths"]["train"])
                    self.assertEqual(raw.id.min(), 8000)
                    self.assertNotIn("high_value", raw)

    def test_train_and_export_use_two_selected_configs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, offset in [("first", 1000), ("second", 9000)]:
                with self.subTest(config=name):
                    config_path, cfg = write_config(root, name, offset)
                    fits = []
                    original_fit = CalibratedClassifierCV.fit

                    def record_fit(instance, X, y=None, **kwargs):
                        fits.append(X.copy())
                        return original_fit(instance, X, y, **kwargs)

                    with patch("src.data.load_config", side_effect=AssertionError("default config used")), patch.object(CalibratedClassifierCV, "fit", record_fit):
                        result = train(config_path)
                        exported, summary = export_train_predictions(config_path)
                        final_export, final_summary = export_train_predictions(config_path, scope="final")
                    self.assertEqual(result["n_samples"], 32)
                    self.assertEqual(result["input_data"]["path"], str(Path(cfg["paths"]["train"]).resolve()))
                    self.assertEqual(set(final_export.id), set(range(offset, offset + 32)))
                    self.assertEqual(set(final_export.prediction_scope), {"final_training_in_sample"})
                    self.assertFalse(final_summary["evaluation_use"])
                    self.assertEqual(set(exported.prediction_scope), {"holdout"})
                    self.assertTrue(exported.id.is_unique)
                    self.assertEqual(len(pd.read_csv(summary["output_path"])), 8)
                    self.assertEqual(summary["input_path"], result["input_data"]["path"])
                    self.assertEqual([len(X) for X in fits], [24, 32])
                    raw, target = prepare_xy(profiles(offset))
                    expected_train, _, expected_y, _ = split_train_test(raw, target, test_size=.25, random_state=42)
                    self.assertFalse(set(exported.id).intersection(profiles(offset).loc[expected_train.index, "id"]))
                    self.assertEqual(fits[0].index.tolist(), expected_train.index.tolist())
                    self.assertEqual(result["feature_engineering"]["evaluation"]["monthly_median"], expected_train.MonthlyCharges.median())
                    self.assertEqual(result["evaluation_scale_pos_weight"], (len(expected_y) - expected_y.sum()) / expected_y.sum())
                    saved = json.loads(Path(cfg["paths"]["metrics"]).read_text(encoding="utf-8"))
                    self.assertEqual(saved, result)

    def test_export_rejects_legacy_model_before_writing_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path, cfg = write_config(Path(folder), "old")
            joblib.dump(Pipeline([("preprocessor", StandardScaler())]), cfg["paths"]["model"])
            with self.assertRaisesRegex(ValueError, "Retrain"):
                export_train_predictions(config_path)
            self.assertFalse(Path(cfg["paths"]["bi_predictions"]).exists())


if __name__ == "__main__":
    unittest.main()
