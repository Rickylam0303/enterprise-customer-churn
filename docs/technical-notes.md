# Setup and technical notes

[Back to project overview](../README.md)

## Environment

Use Python 3.11.15 and a virtual environment. The [dependency lock](../requirements.lock.txt) pins the current Windows pip environment; [requirements-tested.txt](../requirements-tested.txt) is the smaller direct-dependency snapshot. A short Windows venv path avoids Jupyter extension installation hitting path-length limits.

The isolated-environment check passed all 23 tests, `pip check`, and all 139 locked versions. This reused the existing Python interpreter on the same Windows host; other operating systems and a second full-data training run were not validated. The check used a fresh isolated venv, `python -m pip check`, the regression suite and the README/run integrity check.

## Configuration and generated files

Training and export use the selected YAML file. Relative paths resolve from the project root; absolute paths are supported. Output paths are configurable. Examples:

```bash
python -m src.train --config config.yaml
python -m src.export_predictions --config config.yaml --scope holdout
python -m src.reporting --config config.yaml
python -m src.reporting --config config.yaml --check
```

`--scope holdout` exports saved, calibrated evaluation scores for BI. `--scope final` is an optional, explicitly in-sample diagnostic export and must not replace holdout evaluation. The app uses the separately refitted final model.

Run folders under `outputs/runs/<run_id>/` preserve metrics, configuration, environment, split/fold assignments and model/output identities. `outputs/metrics.json` describes the current completed run. A failed or partial run is not a completed evaluation. The original evaluation run retains its original 137-package snapshot; the current lock additionally fixes pip and setuptools.

## Cleaning and feature behavior

Rules are defined in [schema.py](../src/schema.py) and [features.py](../src/features.py):

- SeniorCitizen accepts 0/1 and is explicitly included in numeric preprocessing.
- Tenure accepts whole months in [0,100]. Shared bins include 0: [0,6], (6,12], (12,24], (24,60], (60,100].
- Missing TotalCharges becomes zero only for tenure=0; invalid/negative/infinite charges are rejected. Other columns are not filled with zero.
- Categories, target labels, IDs and phone/internet service combinations are validated.
- FeatureEngineer learns the high_value medians during fit and reuses them at inference, so scoring does not depend on the other customers in a batch.

## Notebook and app roles

Notebook 01 covers descriptive EDA; notebook 02's exploratory feature fit is descriptive, not evaluation. Notebook 03 reads the authoritative comparison outputs. Notebook 04 uses the shared training/export workflow. Historical notebook scores are not substituted for current results.

The Streamlit demo accepts raw customer profiles. Its profile-based suggestions are hand-written rules, not local model attribution or causal explanations. The earlier online deployment was not updated as part of this local refactor.

## Power BI status

The saved PBIX and screenshots are historical artifacts. The review copy opens in Desktop, but refreshing the latest holdout population remains unfinished after a Formula.Firewall error. A separate ProjectFilesSource staging query is proposed; it has not been successfully refreshed and verified. Keep existing privacy settings. [Desktop integration checklist](power-bi-final-validation.md).

Historical churn counts and predicted high-risk counts are distinct KPIs. Action segments require at least 1,000 customers before ranking. The latest Python summaries are [bi_kpis.json](../outputs/bi_kpis.json), [all segments](../outputs/bi_segments.csv), and [eligible action segments](../outputs/bi_action_segments.csv). The M files use an example project root (`C:\Projects\enterprise-customer-churn`); replace it with your clone path in all three queries. M paths must also be adjusted when using custom export locations.

Public metrics use project-relative input/config paths. This publication redaction does not change evaluation values or recorded source/artifact checksums. Local environment logs and machine-specific setup notes are not included.
