"""Business-facing churn risk demo — loads a pre-trained sklearn pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path
from src.data import clean_total_charges
from src.features import engineer_features
from src.preprocessing import prepare_xy

# ---------------------------------------------------------------------------
# Feature-name → plain-language labels (for importance / driver display)
# ---------------------------------------------------------------------------
FEATURE_LABELS: dict[str, str] = {
    "cat__Contract_Month-to-month": "Month-to-month contract",
    "cat__Contract_One year": "One-year contract",
    "cat__Contract_Two year": "Two-year contract",
    "cat__InternetService_Fiber optic": "Fiber optic internet",
    "cat__InternetService_DSL": "DSL internet",
    "cat__InternetService_No": "No internet service",
    "cat__PaymentMethod_Electronic check": "Electronic check payment",
    "cat__PaymentMethod_Mailed check": "Mailed check payment",
    "cat__PaymentMethod_Bank transfer (automatic)": "Automatic bank transfer",
    "cat__PaymentMethod_Credit card (automatic)": "Automatic credit card",
    "cat__OnlineSecurity_No": "No online security",
    "cat__OnlineSecurity_Yes": "Online security enabled",
    "num__tenure": "Customer tenure (months)",
    "num__MonthlyCharges": "Monthly charges",
    "num__TotalCharges": "Total charges to date",
    "num__avg_monthly_spend": "Average monthly spend",
    "num__service_count": "Number of add-on services",
    "num__mtm_early": "Month-to-month in first year",
    "num__electronic_check": "Pays by electronic check",
    "num__high_value": "High-value loyal customer",
    "num__missing_online_sec": "Internet without online security",
    "num__tenure_x_monthly": "Tenure × monthly charges",
}

INTERNET_DEPENDENT = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

SERVICE_OPTIONS = ["No", "Yes", "No internet service"]
YES_NO = ["No", "Yes"]
YES_NO_PHONE = ["No", "No phone service", "Yes"]


@st.cache_resource
def load_pipeline_and_config():
    config = load_config()
    model_path = resolve_path(config, "model")
    if not model_path.exists():
        st.error(
            f"Model not found at `{model_path}`. "
            "Run `python -m src.train` from the project root first."
        )
        st.stop()
    pipeline = joblib.load(model_path)
    metrics = {}
    metrics_path = resolve_path(config, "metrics")
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as f:
            metrics = json.load(f)
    return pipeline, config, metrics


def risk_band(probability: float, thresholds: dict) -> tuple[str, str]:
    low = thresholds.get("low", 0.30)
    high = thresholds.get("high", 0.60)
    if probability < low:
        return "Low", "🟢"
    if probability < high:
        return "Medium", "🟡"
    return "High", "🔴"


def profile_drivers(row: pd.Series) -> list[str]:
    """Rule-based retention drivers from the customer profile."""
    drivers: list[str] = []

    if row.get("Contract") == "Month-to-month":
        drivers.append("Month-to-month contract — highest churn segment in the portfolio")
    if row.get("InternetService") == "Fiber optic":
        drivers.append("Fiber optic subscriber — competitive market with elevated churn risk")
    if row.get("PaymentMethod") == "Electronic check":
        drivers.append("Electronic check payment — often linked to lower engagement and higher churn")
    if row.get("tenure", 0) < 12:
        drivers.append("Early tenure (< 12 months) — critical onboarding window")
    if (
        row.get("InternetService") != "No"
        and row.get("OnlineSecurity") == "No"
    ):
        drivers.append("No online security with internet — bundle security to increase stickiness")
    if row.get("mtm_early") == 1:
        drivers.append("Month-to-month customer in first year — priority for contract upgrade offer")
    if row.get("service_count", 0) <= 1 and row.get("InternetService") != "No":
        drivers.append("Few add-on services — cross-sell opportunity to deepen relationship")

    if not drivers:
        drivers.append("Profile shows moderate risk factors — monitor usage and satisfaction")

    return drivers[:5]


def recommended_actions(band: str, row: pd.Series) -> list[str]:
    actions: list[str] = []

    if band == "High":
        actions.append("Assign to retention specialist within 48 hours")
        actions.append("Offer loyalty discount or contract upgrade to annual plan")
    elif band == "Medium":
        actions.append("Send proactive check-in email or call within two weeks")
        actions.append("Highlight value of bundled services (security, support, streaming)")
    else:
        actions.append("Maintain standard nurture cadence — quarterly satisfaction survey")
        actions.append("Upsell add-ons that fit usage profile")

    if row.get("Contract") == "Month-to-month":
        actions.append("Incentivize switch to one- or two-year contract (e.g. first-month discount)")
    if row.get("PaymentMethod") == "Electronic check":
        actions.append("Promote automatic payment setup with small bill credit")
    if (
        row.get("InternetService") != "No"
        and row.get("OnlineSecurity") == "No"
    ):
        actions.append("Bundle Online Security at introductory rate")

    return list(dict.fromkeys(actions))[:5]


def build_customer_row(inputs: dict) -> pd.DataFrame:
    df = pd.DataFrame([inputs])
    df = clean_total_charges(df)
    return engineer_features(df)


def render_sidebar() -> dict:
    st.sidebar.header("Customer profile")

    with st.sidebar.form("customer_form"):
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
        total = st.number_input("Total charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=0.1)

        st.subheader("Account")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        paperless = st.selectbox("Paperless billing", YES_NO)

        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", YES_NO)
        dependents = st.selectbox("Dependents", YES_NO)

        st.subheader("Services")
        phone = st.selectbox("Phone service", YES_NO)
        multiple_lines = st.selectbox("Multiple lines", YES_NO_PHONE)
        internet = st.selectbox("Internet service", ["No", "DSL", "Fiber optic"])

        if internet == "No":
            sec_default = "No internet service"
        else:
            sec_default = "No"

        online_sec = st.selectbox("Online security", SERVICE_OPTIONS, index=SERVICE_OPTIONS.index(sec_default))
        online_bak = st.selectbox("Online backup", SERVICE_OPTIONS, index=SERVICE_OPTIONS.index(sec_default))
        device_prot = st.selectbox("Device protection", SERVICE_OPTIONS, index=SERVICE_OPTIONS.index(sec_default))
        tech_sup = st.selectbox("Tech support", SERVICE_OPTIONS, index=SERVICE_OPTIONS.index(sec_default))
        stream_tv = st.selectbox("Streaming TV", SERVICE_OPTIONS, index=SERVICE_OPTIONS.index(sec_default))
        stream_mov = st.selectbox("Streaming movies", SERVICE_OPTIONS, index=SERVICE_OPTIONS.index(sec_default))

        submitted = st.form_submit_button("Estimate churn risk", type="primary")

    return {
        "submitted": submitted,
        "inputs": {
            "id": 0,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bak,
            "DeviceProtection": device_prot,
            "TechSupport": tech_sup,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_mov,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        },
    }


def main() -> None:
    st.set_page_config(
        page_title="Telecom Churn Risk Estimator",
        page_icon="📡",
        layout="wide",
    )

    pipeline, config, metrics = load_pipeline_and_config()
    thresholds = config.get("risk_thresholds", {"low": 0.30, "high": 0.60})

    st.title("Estimate churn risk for a telecom customer")
    st.markdown(
        "Enter a customer profile to get an instant churn probability, risk band, "
        "and retention recommendations — powered by a tuned XGBoost pipeline."
    )

    form = render_sidebar()

    col_main, col_side = st.columns([2, 1])

    if form["submitted"]:
        df = build_customer_row(form["inputs"])
        X, _ = prepare_xy(df)
        proba = float(pipeline.predict_proba(X)[0, 1])
        band, icon = risk_band(proba, thresholds)
        row = df.iloc[0]

        with col_main:
            st.subheader("Risk assessment")
            m1, m2, m3 = st.columns(3)
            m1.metric("Churn probability", f"{proba:.1%}")
            m2.metric("Risk band", f"{icon} {band}")
            m3.metric(
                "Tenure",
                f"{int(row['tenure'])} mo",
            )

            st.progress(min(proba, 1.0))

            st.subheader("Key drivers")
            for driver in profile_drivers(row):
                st.markdown(f"- {driver}")

            st.subheader("Recommended actions")
            for action in recommended_actions(band, row):
                st.markdown(f"- {action}")

        with col_side:
            st.info(
                "**How to read this:** Higher scores mean the customer is more likely "
                "to cancel service. Focus retention spend on High and Medium bands."
            )
    else:
        with col_main:
            st.info("Use the sidebar form and click **Estimate churn risk** to see results.")

    with st.expander("Model details"):
        holdout = metrics.get("holdout_metrics", {})
        roc = holdout.get("roc_auc")
        if roc is not None:
            st.metric("Holdout ROC-AUC", f"{roc:.3f}")
        st.caption(
            "Production pipeline: preprocessing → SelectFromModel (XGBoost) → tuned XGBoost. "
            "Trained on Kaggle Playground Series S6E3 data."
        )


if __name__ == "__main__":
    main()
