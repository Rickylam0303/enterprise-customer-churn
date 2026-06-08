# Dashboard Review — Final

**File:** `Executive Overview.pbix` · **Data:** Kaggle Playground S6E3 (~594k rows) · **Status:** ✅ Portfolio-ready

---

## Verdict

| Area | Status |
|------|--------|
| Page 1 — Executive Overview (20 visuals) | ✅ Ready |
| Page 2 — Retention Playbook (11 visuals) | ✅ Ready (minor polish optional) |
| ML bridge (`predicted_proba`, `risk_band`) | ✅ Present on Page 1 |
| Matches Python EDA story | ✅ Contract, tenure, internet, payment |
| Publish to web | ⏳ **You** — see [`PUBLISH.md`](PUBLISH.md) |

---

## Page 1 — Executive Overview

| Visual | Fields | OK? |
|--------|--------|-----|
| 5 business KPIs | Total Customers, Churn Rate, At Risk, Tenure, Monthly $ | ✅ |
| 3 ML KPIs | ML At Risk, Med+ Risk %, Avg Predicted Prob | ✅ |
| 5 slicers | Contract, Internet, Senior, Tenure, Risk band | ✅ |
| 4 churn bars | Contract, Payment, Tenure, Internet | ✅ |
| ML bar + line | Pred. high risk by contract; actual vs predicted by tenure | ✅ |

**Optional:** KPI order left→right; churn card red if >25%; bar Y-axis 0–50%.

---

## Page 2 — Retention Playbook

| Visual | Purpose | OK? |
|--------|---------|-----|
| Matrix | Contract × tenure → `Segment Churn Rate` | ✅ |
| Line | Churn by `MonthlyCharges_Bucket` | ✅ |
| Combo | `service_count` vs churn + count | ✅ |
| Table | Top segments (Contract, tenure, Internet) | ✅ |
| Playbook text | Retention action bullet | ✅ |

**Optional:** Sync slicers with Page 1; fix title `—` encoding; matrix heatmap colors; remove unused action button.

---

## Data checks

| Check | Expected |
|-------|----------|
| Rows | ~594,194 |
| Churn rate | ~23% |
| Model metrics | ROC-AUC ~0.916 (`outputs/metrics.json`) |
| BI predictions CSV | `outputs/bi_train_predictions.csv` (gitignored) |

If ML columns are blank in Power BI → re-run `python -m src.export_predictions` and refresh.
