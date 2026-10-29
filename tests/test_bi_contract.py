"""BI role definitions and the 999/1000 action-sample boundary."""
import unittest
import pandas as pd
from test_pipeline_regressions import profiles
from src.bi import summarize_holdout


class BIContractTests(unittest.TestCase):
    def test_small_segments_do_not_enter_action_ranking(self):
        base = profiles().iloc[[0]].copy(); base["tenure"] = 6
        first = pd.concat([base]*999, ignore_index=True); first["Contract"] = "Month-to-month"
        second = pd.concat([base]*1000, ignore_index=True); second["Contract"] = "Two year"
        second["Churn"] = "No"
        single = base.copy(); single["Contract"] = "One year"
        raw = pd.concat([first, second, single], ignore_index=True); raw["id"] = raw.index
        scored = pd.DataFrame({"id": raw.id, "Churn": raw.Churn, "predicted_proba": .7,
            "risk_band": "High", "prediction_scope": "holdout"})
        kpis, all_segments, actions = summarize_holdout(raw, scored, 1000)
        self.assertEqual(len(all_segments), 3)
        self.assertEqual(actions.customer_count.tolist(), [1000])
        self.assertEqual(actions.action_rank.tolist(), [1])
        self.assertEqual(kpis["excluded_small_segments"], 2)
        self.assertEqual(kpis["Historical Churned Customers"], int(raw.Churn.eq("Yes").sum()))
        self.assertEqual(kpis["Historical Churned Customers"], 1000)
        self.assertEqual(kpis["Predicted High-Risk Customers"], 2000)

    def test_in_sample_scores_and_duplicate_ids_are_rejected(self):
        raw = profiles(); scored = pd.DataFrame({"id": raw.id, "Churn": raw.Churn,
            "predicted_proba": .7, "risk_band": "High", "prediction_scope": "final_training_in_sample"})
        with self.assertRaisesRegex(ValueError, "holdout"): summarize_holdout(raw, scored)
        scored.prediction_scope = "holdout"; scored.loc[1, "id"] = scored.id.iloc[0]
        with self.assertRaisesRegex(ValueError, "unique"): summarize_holdout(raw, scored)
