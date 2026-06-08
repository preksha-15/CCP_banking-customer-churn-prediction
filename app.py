# ============================================================
#         CUSTOMER CHURN PREDICTION - STREAMLIT APP
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── LOAD MODELS ─────────────────────────────────────────────
rf_model  = joblib.load('random_forest.pkl')
xgb_model = joblib.load('xgboost.pkl')
lr_model  = joblib.load('logistic_regression.pkl')
scaler    = joblib.load('scaler.pkl')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(page_title="Customer Churn Prediction",
                   page_icon="🏦", layout="centered")

# ── CUSTOM CSS ──────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Force light mode */
    .stApp {
        background-color: #f0f4f8 !important;
    }

    .main {
        background-color: #f0f4f8;
    }

    p, label, div, span {
        color: #1a1a2e !important;
    }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div {
        background-color: white !important;
        color: #1a1a2e !important;
    }

    .block-container {
        padding: 2rem 3rem;
    }

    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e !important;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1rem;
        color: #666 !important;
        margin-bottom: 2rem;
    }

    .card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e !important;
        margin-bottom: 1rem;
        border-left: 4px solid #4f8ef7;
        padding-left: 0.75rem;
    }

    .predict-btn button {
        background: linear-gradient(135deg, #4f8ef7, #1a1a2e) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }

    .result-box {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 1.5rem;
    }

    .churn {
        background: linear-gradient(135deg, #ff6b6b, #ee0979);
        color: white !important;
    }

    .stay {
        background: linear-gradient(135deg, #56ab2f, #a8e063);
        color: white !important;
    }

    .metric-card {
        background: #f8faff;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e8ff;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #888 !important;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a2e !important;
    }

    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        font-weight: 600;
        color: #444 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────────
st.markdown('<div class="title">🏦 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter customer details below to predict churn probability</div>', unsafe_allow_html=True)

# ── INPUT CARD ──────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    credit_score = st.number_input("Credit Score",         300, 850,    650)
    age          = st.number_input("Age",                  18,  100,    35)
    tenure       = st.number_input("Tenure (years)",       0,   10,     5)
    gender       = st.selectbox("Gender", ["Male", "Female"])

with col2:
    balance          = st.number_input("Balance ($)",          0, 250000, 50000)
    estimated_salary = st.number_input("Estimated Salary ($)", 0, 200000, 60000)
    num_of_products  = st.number_input("Number of Products",   1, 4,      2)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">💳 Account Information</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    has_cr_card = st.selectbox("Has Credit Card",  ["Yes", "No"])
with col4:
    is_active   = st.selectbox("Is Active Member", ["Yes", "No"])

st.markdown('</div>', unsafe_allow_html=True)

# ── PREDICT BUTTON ──────────────────────────────────────────
st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
predict = st.button("🔍 Predict Churn Probability")
st.markdown('</div>', unsafe_allow_html=True)

# ── PREDICTION ──────────────────────────────────────────────
if predict:
    customer = {
        'CreditScore'    : credit_score,
        'Gender'         : 1 if gender == "Male" else 0,
        'Age'            : age,
        'Tenure'         : tenure,
        'Balance'        : balance,
        'NumOfProducts'  : num_of_products,
        'HasCrCard'      : 1 if has_cr_card == "Yes" else 0,
        'IsActiveMember' : 1 if is_active == "Yes" else 0,
        'EstimatedSalary': estimated_salary
    }

    customer_scaled = scaler.transform(pd.DataFrame([customer]))
    rf_prob  = rf_model.predict_proba(customer_scaled)[:, 1][0]
    xgb_prob = xgb_model.predict_proba(customer_scaled)[:, 1][0]
    lr_prob  = lr_model.predict_proba(np.column_stack([rf_prob, xgb_prob]))[:, 1][0]
    final_prob = lr_prob * 100

    # result box
    if final_prob >= 50:
        st.markdown(f'<div class="result-box churn">⚠️ {final_prob:.2f}% — Likely to CHURN</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box stay">✅ {final_prob:.2f}% — Likely to STAY</div>',
                    unsafe_allow_html=True)

    # individual model metrics
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Individual Model Predictions</div>',
                unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">Random Forest</div>
            <div class="metric-value">{rf_prob*100:.2f}%</div>
        </div>''', unsafe_allow_html=True)
    with m2:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">XGBoost</div>
            <div class="metric-value">{xgb_prob*100:.2f}%</div>
        </div>''', unsafe_allow_html=True)
    with m3:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">Final (Stacking)</div>
            <div class="metric-value">{final_prob:.2f}%</div>
        </div>''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)