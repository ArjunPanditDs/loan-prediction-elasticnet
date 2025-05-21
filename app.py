import streamlit as st
import pickle
import numpy as np
import base64
import requests

# =========================
# 🎨 Set Simple Background Image with 50% opacity overlay
# =========================

def set_background(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img_data = response.content
        b64_encoded = base64.b64encode(img_data).decode()
        style = f"""
            <style>
            .stApp {{
                position: relative;
                background-image: url("data:image/png;base64,{b64_encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                color: #fff;  /* text color */
                min-height: 100vh;
                z-index: 0;
            }}
            /* Overlay to reduce background opacity */
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background-color: rgba(0, 0, 0, 0.3); /* black overlay at 50% opacity */
                z-index: -1;
                pointer-events: none;
            }}
            /* Ensure all Streamlit content appears above the overlay */
            .block-container {{
                position: relative;
                z-index: 1;
            }}
            </style>
        """
        st.markdown(style, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading background image: {str(e)}")

# Put your background image URL here (must be a direct image link)
bg_url = "https://t4.ftcdn.net/jpg/05/21/95/85/360_F_521958580_kNDeJSIB0VUVqJ0n9fUwwubwHTRkn2VS.jpg"
set_background(bg_url)

# =========================
# 🔍 Load Model
# =========================

@st.cache_resource
def load_model():
    with open("logistic_regression_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# =========================
# 🧾 Heading and Inputs (No container)
# =========================

st.markdown("""
    <h1 style='
        text-align: center;
        color: #fff;
        font-size: 3rem;
        margin-top: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);'>
        🏦 XYZ Loan Approval Predictor
    </h1>
    <p style='
        text-align: center;
        color: #f2f6ff;
        font-size: 1.5rem;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.6);'>
        Enter your financial details to check loan eligibility
    </p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    loan_term = st.number_input("📅 Loan Term (years)", 1, 30, 15)
    cibil_score = st.number_input("📊 CIBIL Score", 300, 900, 750)

with col2:
    total_assets = st.number_input("🏘️ Total Assets (₹ Lakhs)", 0.0, 1000.0, 15.0, step=0.5)
    income = st.number_input("💰 Annual Income (₹ Lakhs)", 1.0, 1000.0, 8.0, step=0.5)

loan_amount = st.slider("💵 Loan Amount (₹ Lakhs)", 1.0, 50.0, 10.0, step=0.5)

# =========================
# 🚀 Prediction Button
# =========================

if st.button("🔍 Check Approval Status"):
    try:
        input_data = np.array([[loan_term, cibil_score, total_assets, income, loan_amount]])
        with st.spinner("Analyzing your financial profile..."):
            prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.balloons()
            st.success("✅ **Loan Approved!**\n\nCongratulations! You meet the eligibility criteria.")
        else:
            st.error("❌ **Loan Rejected**\n\nTip: Improve your CIBIL score or reduce the loan amount.")

    except Exception as e:
        st.error(f"⚠️ Prediction Error: {str(e)}\n\nMake sure your model is trained with the same 5 input features.")

# =========================
# 📎 Footer
# =========================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #eee; font-size: 0.9rem; text-shadow: 1px 1px 3px rgba(0,0,0,0.7);'>
        <p>This app is for educational and demo purposes only.</p>
        <p>© 2025 XYZ Loan Predictor System</p>
    </div>
""", unsafe_allow_html=True)
