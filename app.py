import streamlit as st
import pickle
import numpy as np

# Load the model
with open('logistic_regression_model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Streamlit app setup
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🧠 Loan Approval Prediction App")
st.markdown("Enter your details to see if your loan gets approved!")

# Input fields - MATCH your training features
loan_term = st.number_input("📅 Loan Term (in years)", min_value=1, max_value=30, value=15, step=1)
cibil_score = st.number_input("📊 CIBIL Score", min_value=300, max_value=900, value=700, step=1)
total_assets_lakh = st.number_input("🏘️ Total Assets (in Lakhs)", min_value=0.0, value=10.0, step=0.1)
income_annual_lakh = st.number_input("💰 Annual Income (in Lakhs)", min_value=1.0, value=5.0, step=0.1)
loan_amount_lakh = st.number_input("🏦 Loan Amount (in Lakhs)", min_value=1.0, value=2.0, step=0.1)

# Predict
if st.button("Predict Loan Approval"):
    features = np.array([[loan_term, cibil_score, total_assets_lakh, income_annual_lakh, loan_amount_lakh]])

    try:
        prediction = pipeline.predict(features)[0]

        if prediction == 1:
            st.success("✅ Loan Approved!")
        else:
            st.error("❌ Loan Rejected.")
    
    except Exception as e:
        st.error(f"Oops! Error: {e}")
