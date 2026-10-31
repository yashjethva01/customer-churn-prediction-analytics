from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard_data import (
    add_tenure_group,
    calculate_kpis,
    churn_rate_by,
    filter_customers,
    load_dashboard_data,
)
from src.predict import build_prediction_input, load_compatible_model, predict_customer


st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_dashboard_data()


@st.cache_resource
def get_model():
    return load_compatible_model()


def render_analytics(data: pd.DataFrame) -> None:
    st.header("Customer Churn Analytics")
    st.caption("Historical dataset analysis; this is not live customer data.")

    with st.sidebar:
        st.subheader("Analytics filters")
        contracts = st.multiselect("Contract type", sorted(data["Contract"].unique()), default=sorted(data["Contract"].unique()))
        internet = st.multiselect("Internet service", sorted(data["InternetService"].unique()), default=sorted(data["InternetService"].unique()))
        payments = st.multiselect("Payment method", sorted(data["PaymentMethod"].unique()), default=sorted(data["PaymentMethod"].unique()))
        tenure = st.slider("Tenure range (months)", 0, int(data["tenure"].max()), (0, int(data["tenure"].max())))

    filtered = filter_customers(data, contracts, internet, payments, tenure)
    kpis = calculate_kpis(filtered)
    columns = st.columns(4)
    columns[0].metric("Customers", f"{kpis['total_customers']:,}")
    columns[1].metric("Churned", f"{kpis['churned_customers']:,}")
    columns[2].metric("Churn rate", f"{kpis['churn_rate_percent']:.2f}%")
    columns[3].metric("Avg. monthly charges", f"${kpis['average_monthly_charges']:.2f}")

    if filtered.empty:
        st.warning("No customers match the selected filters. Adjust the sidebar filters.")
        return

    left, right = st.columns(2)
    churn_counts = filtered["Churn"].value_counts().rename_axis("status").reset_index(name="customers")
    left.plotly_chart(px.bar(churn_counts, x="status", y="customers", title="Customers by churn status"), use_container_width=True)
    contract_rates = churn_rate_by(filtered, "Contract")
    right.plotly_chart(px.bar(contract_rates, x="Contract", y="churn_rate_percent", title="Churn rate by contract", labels={"churn_rate_percent": "Churn rate (%)"}), use_container_width=True)

    grouped = add_tenure_group(filtered)
    left, right = st.columns(2)
    tenure_rates = churn_rate_by(grouped, "tenure_group")
    left.plotly_chart(px.bar(tenure_rates, x="tenure_group", y="churn_rate_percent", title="Churn rate by tenure group", labels={"churn_rate_percent": "Churn rate (%)"}), use_container_width=True)
    payment_rates = churn_rate_by(filtered, "PaymentMethod")
    right.plotly_chart(px.bar(payment_rates, x="PaymentMethod", y="churn_rate_percent", title="Churn rate by payment method", labels={"churn_rate_percent": "Churn rate (%)"}), use_container_width=True)

    left, right = st.columns(2)
    internet_rates = churn_rate_by(filtered, "InternetService")
    left.plotly_chart(px.bar(internet_rates, x="InternetService", y="churn_rate_percent", title="Churn rate by internet service", labels={"churn_rate_percent": "Churn rate (%)"}), use_container_width=True)
    right.plotly_chart(px.box(filtered, x="Churn", y="MonthlyCharges", title="Monthly charges by churn status", labels={"MonthlyCharges": "Monthly charges"}), use_container_width=True)

    with st.expander("Historical customer sample"):
        preview_columns = [column for column in ["gender", "SeniorCitizen", "tenure", "Contract", "InternetService", "MonthlyCharges", "Churn"] if column in filtered.columns]
        st.dataframe(filtered[preview_columns].head(100), use_container_width=True, hide_index=True)


def render_prediction(data: pd.DataFrame) -> None:
    st.header("Customer Churn Prediction")
    try:
        model, model_label = get_model()
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(str(error))
        return

    st.info(f"Using: {model_label}. Predictions use the fitted historical pipeline.")
    feature_names = list(model.named_steps["preprocessor"].feature_names_in_)
    values: dict[str, object] = {}
    with st.form("prediction_form"):
        profile, service, billing = st.columns(3)
        with profile:
            values["gender"] = st.selectbox("Gender", sorted(data["gender"].dropna().unique())) if "gender" in feature_names else None
            values["SeniorCitizen"] = int(st.checkbox("Senior citizen")) if "SeniorCitizen" in feature_names else None
            values["Partner"] = st.selectbox("Partner", sorted(data["Partner"].dropna().unique())) if "Partner" in feature_names else None
            values["Dependents"] = st.selectbox("Dependents", sorted(data["Dependents"].dropna().unique())) if "Dependents" in feature_names else None
            values["tenure"] = st.slider("Tenure (months)", 0, int(data["tenure"].max()), 12) if "tenure" in feature_names else None
        with service:
            for name in ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
                if name in feature_names:
                    values[name] = st.selectbox(name, sorted(data[name].dropna().unique()))
        with billing:
            values["Contract"] = st.selectbox("Contract", sorted(data["Contract"].dropna().unique())) if "Contract" in feature_names else None
            values["PaperlessBilling"] = st.selectbox("Paperless billing", sorted(data["PaperlessBilling"].dropna().unique())) if "PaperlessBilling" in feature_names else None
            values["PaymentMethod"] = st.selectbox("Payment method", sorted(data["PaymentMethod"].dropna().unique())) if "PaymentMethod" in feature_names else None
            values["MonthlyCharges"] = st.number_input("Monthly charges", min_value=0.0, value=float(data["MonthlyCharges"].median()), step=1.0) if "MonthlyCharges" in feature_names else None
            values["TotalCharges"] = st.number_input("Total charges", min_value=0.0, value=float(pd.to_numeric(data["TotalCharges"], errors="coerce").median()), step=10.0) if "TotalCharges" in feature_names else None
        submitted = st.form_submit_button("Predict churn")

    if submitted:
        try:
            prediction, probability = predict_customer(model, {key: value for key, value in values.items() if value is not None})
            if prediction == 1:
                st.error("Prediction: Likely to churn")
            else:
                st.success("Prediction: Likely to stay")
            if probability is not None:
                st.metric("Estimated churn probability", f"{probability:.1%}")
            st.caption("This is an estimate based on historical patterns, not a certainty or causal explanation.")
        except (ValueError, TypeError, KeyError) as error:
            st.error(f"Could not create a prediction: {error}")


def render_model_info() -> None:
    try:
        model, label = get_model()
        st.sidebar.caption(f"Loaded artifact: {label}")
        st.sidebar.caption(f"Probability estimates: {'available' if hasattr(model, 'predict_proba') else 'unavailable'}")
    except (FileNotFoundError, ValueError, OSError) as error:
        st.sidebar.warning(str(error))


try:
    dataset = get_data()
    render_model_info()
    section = st.sidebar.radio("Section", ["Analytics", "Predict"])
    if section == "Analytics":
        render_analytics(dataset)
    else:
        render_prediction(dataset)
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Dashboard could not load its data: {error}")
