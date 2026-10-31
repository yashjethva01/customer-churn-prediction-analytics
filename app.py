import streamlit as st


st.set_page_config(
    page_title="Customer Churn Prediction & Analytics",
    page_icon="📊",
    layout="centered",
)

st.title("Customer Churn Prediction & Analytics Platform")
st.write(
    "An end-to-end machine learning application for analyzing telecom "
    "customer data and predicting potential churn."
)

st.subheader("Current development status")
st.info("Milestone 1 complete: project foundation and Git setup.")
st.write(
    "Dataset acquisition, exploratory analysis, preprocessing, model training, "
    "and dashboard features will be added in later milestones."
)
