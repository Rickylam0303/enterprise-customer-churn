# Power BI — Retention & Churn Analytics

Two-page executive dashboard in **`Executive Overview.pbix`**, built on the same Kaggle churn data and ML predictions as the Python/Streamlit project.

| File | Purpose |
|------|---------|
| [`Executive Overview.pbix`](Executive%20Overview.pbix) | **Published artifact** — Page 1 Executive Overview + Page 2 Retention Playbook |
| [`DASHBOARD_REVIEW.md`](DASHBOARD_REVIEW.md) | Review, polish checklist, publish readiness |
| [`queries/LoadTrain.m`](queries/LoadTrain.m) | Power Query: load data, clean `TotalCharges`, join ML predictions |
| [`measures/ChurnMeasures.dax`](measures/ChurnMeasures.dax) | DAX measures (churn KPIs + predicted risk) |
| [`PUBLISH.md`](PUBLISH.md) | Publish to Power BI Service and add link to README/CV |

## Pages

1. **Executive Overview** — KPI cards, slicers, churn by contract/internet/payment/tenure, ML risk overlay.
2. **Retention Playbook** — contract × tenure matrix, monthly charge sensitivity, service bundle chart, top-risk segments table.

## Rebuild from source (optional)

```bash
python -m src.train
python -m src.export_predictions   # outputs/bi_train_predictions.csv
```

In Power BI Desktop: paste `queries/LoadTrain.m` (set `ProjectRoot`), add measures from `ChurnMeasures.dax`, build pages per `DASHBOARD_REVIEW.md`.

## Publish

See [`PUBLISH.md`](PUBLISH.md) → set `portfolio.power_bi` in `config.yaml` → update root `README.md`.
