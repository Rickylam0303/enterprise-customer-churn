# Publish — Power BI Dashboard (final)

Your report is already built: **`Executive Overview.pbix`** (2 pages). These steps get a **public link** on your README and CV.

---

## What you do (≈15 minutes)

### 1. Quick sanity check in Power BI Desktop

1. Open **`Executive Overview.pbix`**.
2. Confirm **Page 1** churn rate ≈ **23%** and slicers filter charts.
3. Confirm **Page 2** matrix shows highest churn for **Month-to-month × short tenure**.
4. **File → Save** (if you applied any polish).

### 2. Publish to Power BI Service

1. Sign in: **File → Options → Account** (Microsoft / work / school account).
2. **Home → Publish** → select **My workspace** (free tier is fine).
3. When prompted, click **Open … in Power BI**.

### 3. Create a portfolio link

**Recommended for CV/GitHub** (synthetic Kaggle data):

1. In [Power BI Service](https://app.powerbi.com), open the report.
2. **File → Embed report → Publish to web (public)**.
3. Confirm → copy the **link** (looks like `https://app.powerbi.com/view?r=...`).

**Alternative** (viewers must sign in): **Share → Copy link**.

### 4. Update the repo

Paste your URL in **`config.yaml`**:

```yaml
portfolio:
  power_bi: https://app.powerbi.com/view?r=YOUR_ID_HERE
```

Update the **first line** of root **`README.md`** — replace the Power BI placeholder with your link.

Update **`preserve/portfolio/CV_PORTFOLIO_SNIPPET.md`** (local CV bullets) with the same URL.

### 5. Push to GitHub

From project root:

```bash
git add README.md config.yaml .gitignore src/export_predictions.py power-bi/
git commit -m "Add Power BI dashboard and ML predictions export for portfolio."
git push origin master
```

**Note:** `Executive Overview.pbix` is ~14 MB. GitHub allows it; if push is slow, that is normal.

---

## Already live (no action needed)

| Asset | URL |
|-------|-----|
| GitHub | https://github.com/Rickylam0303/enterprise-customer-churn |
| Streamlit | https://enterprise-customer-churn-9wmrp4azyunsxvecvfpdxa.streamlit.app |

Ensure Streamlit Cloud still has the **model artifact** (`models/churn_xgb_pipeline.joblib`). If the app errors, re-run `python -m src.train` locally and redeploy or commit the model if your hosting setup requires it.

---

## Optional polish before publish

See [`DASHBOARD_REVIEW.md`](DASHBOARD_REVIEW.md). Highest impact:

- **View → Sync slicers** across both pages
- Fix Page 2 title encoding (retype em dash)
- Matrix conditional formatting (white → red)

---

## Refresh after retraining

```bash
python -m src.train
python -m src.export_predictions
```

Power BI Desktop → **Transform data → Refresh** → **Publish** again (overwrite).
