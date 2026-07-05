import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load('model.joblib')
feature_columns = joblib.load('feature_columns.joblib')

st.title("SBA Loan Default Risk Dashboard")
st.markdown("Predicts probability of default on SBA 7(a) loans, with SHAP-based explanation and RDD context at the $150k guarantee-rate threshold.")

st.header("Score a loan")
col1, col2 = st.columns(2)
with col1:
    gross_approval = st.number_input("Loan amount ($)", min_value=1000, max_value=5000000, value=150000, step=1000)
    guarantee_pct = st.slider("Guarantee %", 0.5, 0.9, 0.75)
    loan_term_years = st.slider("Loan term (years)", 1, 25, 7)
with col2:
    business_age = st.selectbox("Business age", ["New (<2 yrs)", "Existing (2+ yrs)"])
    jobs_supported = st.number_input("Jobs supported", 0, 500, 5)
    fixed_rate = st.selectbox("Rate type", ["Variable", "Fixed"])
    interest_rate = st.slider("Interest rate (%)", 3.0, 12.0, 6.5)

input_row = pd.DataFrame([{
    'log_grossapproval': np.log1p(gross_approval),
    'guarantee_pct': guarantee_pct,
    'loan_term_years': loan_term_years,
    'businessage_num': 1 if business_age.startswith("Existing") else 0,
    'jobssupported': jobs_supported,
    'fixed_rate': 1 if fixed_rate == "Fixed" else 0,
    'initialinterestrate': interest_rate
}])
for col in feature_columns:
    if col not in input_row.columns:
        input_row[col] = 0
input_row = input_row[feature_columns]

prob = model.predict_proba(input_row)[0, 1]
st.metric("Predicted default probability", f"{prob:.2%}")

st.subheader("Why this prediction — SHAP")
explainer = shap.TreeExplainer(model)
shap_values = explainer(input_row)
st.caption("SHAP values shown in log-odds space (model margin output), not probability directly.")
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)

st.header("The $150k guarantee threshold: causal finding")
st.markdown("""
Regression discontinuity design around SBA's $150,000 guarantee-rate cutoff (85% below, 75% above).
After correcting for bunching (donut RDD) and adjusting for loan term, loans just above the 
threshold show a lower default rate, consistent with reduced moral hazard when lenders retain 
more risk. Effect: p=0.035 (conventional), p=0.067 (robust bias-corrected).
""")