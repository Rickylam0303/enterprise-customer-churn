"""Data, calibration/split identity and README evidence regressions."""
import json
from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd
from test_pipeline_regressions import profiles, write_config
from src.schema import clean_customer_data, validate_dataset
from src.preprocessing import prepare_xy
from src.models import build_lr_pipeline, get_feature_pipeline, TrainingWeightedXGBClassifier
from src.train import train
from src.config import load_config
from src.artifacts import recorded_path
from src.export_predictions import export_train_predictions
from src.evaluation import compute_metrics, threshold_metrics
from src.reporting import update_readme, verify_run, START, END
from src.data import load_raw_data
from unittest.mock import patch


class DataContractTests(unittest.TestCase):
    def test_senior_citizen_is_encoded(self):
        X, y = prepare_xy(profiles()); model = build_lr_pipeline().fit(X, y)
        self.assertIn("num__SeniorCitizen", model.named_steps["preprocessor"].get_feature_names_out())
        changed = X.iloc[[0]].copy(); changed["SeniorCitizen"] = 1
        features = model.named_steps["features"]
        prep = model.named_steps["preprocessor"]
        self.assertNotEqual(prep.transform(features.transform(X.iloc[[0]]))[0, 0], prep.transform(features.transform(changed))[0, 0])

    def test_corrupt_data_rejected_without_mutating_input(self):
        cases = [("SeniorCitizen", 2), ("SeniorCitizen", None), ("MonthlyCharges", -1),
            ("MonthlyCharges", np.inf), ("TotalCharges", "oops"), ("TotalCharges", -1),
            ("TotalCharges", np.inf), ("Partner", None), ("gender", "unknown"),
            ("MultipleLines", "No phone service"), ("InternetService", "No")]
        for name, value in cases:
            with self.subTest(name=name, value=value):
                data = profiles().astype({name: "object"}); data.loc[0, name] = value
                before = data.copy(deep=True)
                with self.assertRaises(ValueError): clean_customer_data(data)
                pd.testing.assert_frame_equal(data, before)

    def test_total_charges_missing_policy(self):
        raw = profiles(); raw.loc[0, "TotalCharges"] = " "
        self.assertEqual(clean_customer_data(raw).TotalCharges.iloc[0], 0)
        raw.loc[1, "TotalCharges"] = None
        with self.assertRaisesRegex(ValueError, "only when tenure is 0"): clean_customer_data(raw)

    def test_ids_target_and_required_columns(self):
        raw = profiles(); raw.loc[1, "id"] = raw.loc[0, "id"]
        with self.assertRaisesRegex(ValueError, "unique"): validate_dataset(raw, require_target=True)
        for label in [None, "Maybe"]:
            raw = profiles(); raw.loc[0, "Churn"] = label
            with self.assertRaisesRegex(ValueError, "Churn"): prepare_xy(raw)
        with self.assertRaisesRegex(ValueError, "SeniorCitizen"):
            clean_customer_data(profiles().drop(columns="SeniorCitizen"))

    def test_weights_are_fit_local(self):
        X = np.arange(24).reshape(12, 2); y = np.array([0]*9 + [1]*3)
        model = TrainingWeightedXGBClassifier(scale_pos_weight=999, n_estimators=2, max_depth=2, n_jobs=1)
        model.fit(X, y)
        self.assertEqual(model.get_params()["scale_pos_weight"], 3)

    def test_explicit_raw_source_does_not_pull_default_test_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "chosen.csv"; profiles().to_csv(path, index=False)
            with patch("src.data.load_config", side_effect=AssertionError("default config used")):
                raw, test = load_raw_data(path)
            self.assertEqual(len(raw), 32)
            self.assertIsNone(test)


class EvaluationEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        cls.config_path, _ = write_config(cls.root, "evidence", 5000)
        cls.result = train(cls.config_path); cls.config = load_config(cls.config_path)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_split_calibration_and_separate_final_identity(self):
        r = self.result
        splits = pd.read_csv(recorded_path(self.config, r["current_outputs"]["split_assignments"]))
        held = splits[splits.partition.eq("holdout")]; fit = splits[splits.partition.eq("fit")]
        self.assertTrue(held.evaluation_calibration_fold.eq(-1).all())
        self.assertTrue(fit.evaluation_calibration_fold.ge(0).all())
        self.assertFalse(set(held.id).intersection(fit.id))
        for key, subset in [("evaluation_model", fit), ("final_model", splits)]:
            artifact = joblib.load(recorded_path(self.config, r["artifacts"][key]["path"]))
            raw = profiles(5000).set_index("id").loc[subset.id]
            features = get_feature_pipeline(artifact).named_steps["features"]
            self.assertEqual(features.monthly_median_, raw.MonthlyCharges.median())
            self.assertEqual(features.tenure_median_, raw.tenure.median())
        self.assertNotEqual(r["artifacts"]["evaluation_model"]["path"], r["artifacts"]["final_model"]["path"])

    def test_reported_scores_reproduce_metrics_and_thresholds(self):
        r = self.result
        scores = pd.read_csv(recorded_path(self.config, r["current_outputs"]["holdout_predictions"]))
        p = scores.calibrated_score.to_numpy(); y = scores.actual.to_numpy()
        expected = compute_metrics(y, p >= r["evaluation_method"]["default_threshold"], p)
        for key, value in expected.items(): self.assertAlmostEqual(value, r["holdout_metrics"][key], places=9)
        actual = [v for v in r["threshold_metrics"] if v["model"] == r["model"]]
        for got, expected in zip(actual, threshold_metrics(y, p, r["evaluation_method"]["selected_thresholds"])):
            for key in ["tn", "fp", "fn", "tp", "selected", "precision", "recall"]:
                self.assertAlmostEqual(got[key], expected[key], places=9)
        self.assertEqual({v["fit_rows"] for v in r["model_comparison"]}, {r["n_train"]})
        self.assertEqual({v["holdout_rows"] for v in r["model_comparison"]}, {r["n_holdout"]})

    def test_readme_generation_detects_stale_values_and_tampered_sources(self):
        readme = self.root / "README.md"; readme.write_text(START + "\nstale\n" + END, encoding="utf8")
        update_readme(self.config_path, readme_path=readme)
        update_readme(self.config_path, check=True, readme_path=readme)
        readme.write_text(readme.read_text(encoding="utf8").replace("ROC-AUC", "stale-AUC"), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "stale"): update_readme(self.config_path, check=True, readme_path=readme)
        table = recorded_path(self.config, self.result["current_outputs"]["threshold_metrics"])
        original = table.read_bytes()
        try:
            table.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "differs"): verify_run(self.config, self.result)
        finally: table.write_bytes(original)

    def test_export_scope_and_checksum_guard(self):
        frame, manifest = export_train_predictions(self.config_path)
        self.assertEqual(len(frame), self.result["n_holdout"])
        self.assertEqual(set(frame.prediction_scope), {"holdout"})
        self.assertTrue(manifest["evaluation_use"])
        source = Path(self.result["input_data"]["path"]); original = source.read_bytes()
        try:
            source.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum"): export_train_predictions(self.config_path)
        finally: source.write_bytes(original)


if __name__ == "__main__": unittest.main()
