# Power BI refresh checklist

The portfolio PBIX renders its historical cached dataset. Refreshing it with the current Python holdout scores remains pending after a Power Query Formula.Firewall error. The separate staging-query approach has not yet been successfully refreshed in Desktop.

## Connect the local project

1. Open the retained local portfolio PBIX and save a working copy, or create a report using the published M/DAX sources. PBIX files are not published because they embed machine-specific source metadata.
2. Set `ProjectRoot` in all three M files to your clone directory. The checked-in path is an example, not a working data source.
3. Create a blank query named `ProjectFilesSource` using `power-bi/queries/ProjectFilesSource.m`; disable its load. Replace the existing `train` query with `LoadTrain.m`.
4. Optionally create an aggregate segment query using `LoadActionSegments.m`. It contains one row per segment; avoid a many-to-many relationship that duplicates customer counts.
5. Keep existing privacy settings. Refresh with the saved holdout predictions and matching `bi_kpis.json`; do not substitute final-model in-sample scores.
6. Apply the measures in `power-bi/measures/ChurnMeasures.dax` individually. Keep historical churn and predicted high-risk KPIs distinct. Filter segments to at least 1,000 customers before ranking.

## Verify the refreshed report

Clear all slicers and compare with `outputs/bi_kpis.json` from run `20261027T175000746489Z-71d14d34`:

| 項目 | 預期 |
|---|---:|
| 客戶列數／不同 id 數 | 118,839／118,839 |
| prediction_scope 唯一值 | holdout |
| model_version 唯一值 | c05a64379a94 |
| Historical Churned Customers | 26,763 |
| Predicted High-Risk Customers（≥0.6） | 21,901 |
| Historical Churn Rate | 22.5204% |
| Mean Calibrated Churn Score | 22.6008% |
| 非空分群／合資格分群 | 44／22 |
| 行動分群最低列數 | ≥1,000 |

Check tenure=0, SeniorCitizen labels, slicer-dependent denominators and the minimum segment count. Save the refreshed PBIX and two page screenshots, recording the Desktop version, refresh time and run ID. Only then update the README's refresh status. For a new run, use its KPI values rather than the example above.

The proposed staging design follows the [Microsoft Data Privacy Firewall documentation](https://learn.microsoft.com/en-us/power-query/dataprivacyfirewall); this does not establish that the saved PBIX has passed refresh validation.
