# Customer segment findings

[Back to project overview](../README.md)

Population: all 118,839 outer-holdout customers from run `20261027T175000746489Z-71d14d34`. Churn rate means labeled churned customers divided by customers in the group. These summaries do not use final-model in-sample predictions.

Source: [bi_segments.csv](../outputs/bi_segments.csv). The [analysis-summary.csv](analysis-summary.csv) aggregates customer_count and churned_count, then divides the totals; it does not average segment rates. Its Contract and tenure groups each cover the complete holdout population. Model thresholds were not chosen from these findings.

## Counts behind the README

| Group | Customers | Churned | Observed churn |
|---|---:|---:|---:|
| Month-to-month | 59,853 | 25,174 | 42.1% |
| One year | 21,483 | 1,238 | 5.8% |
| Two year | 37,503 | 351 | 0.9% |
| 0-6 | 17,343 | 9,461 | 54.6% |
| 12-24 | 17,159 | 4,889 | 28.5% |
| 24-60 | 39,346 | 5,612 | 14.3% |
| 6-12 | 13,248 | 5,658 | 42.7% |
| 60+ | 31,743 | 1,143 | 3.6% |
| Month-to-month / 0–6 months / DSL | 4,972 | 1,931 | 38.8% |
| Month-to-month / 0–6 months / Fiber optic | 9,546 | 7,345 | 76.9% |
| Month-to-month / 0–6 months / No | 2,482 | 174 | 7.0% |

## Interpretation

Contract length, tenure and internet service identify groups worth investigating. They do not show that changing a contract or adding a service would reduce churn: customer selection, pricing and other differences may contribute. The dataset does not observe reasons for leaving, intervention outcomes, or a future prediction horizon.

The proposed next steps are onboarding review, service/billing investigation and a controlled offer test with suitable comparison groups. No uplift, savings or ROI is claimed. All customer groups highlighted in the README exceed the 1,000-customer action-policy threshold.

The dashboard image uses historical data and may differ from these current holdout counts. The README distinguishes the preview from the latest Python results.
