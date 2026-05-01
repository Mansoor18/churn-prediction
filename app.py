import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import encode_features, engineer_features

# ── Page config ──
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Load model artifacts ──
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# ── Header ──
st.title("📊 Customer Churn Predictor")
st.markdown("*Predict the likelihood of a telecom customer churning using ML*")
st.divider()

# ── Sidebar inputs ──
st.sidebar.header("Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
total_charges = monthly_charges * tenure

# ── Predict button ──
predict_btn = st.sidebar.button("🔮 Predict Churn", use_container_width=True)

# ── Prediction logic ──
def build_input():
    return {
        'gender': gender,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': str(total_charges)
    }

def predict(input_dict):
    df = pd.DataFrame([input_dict])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = encode_features(df, is_training=False)
    df = engineer_features(df)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    scaled = scaler.transform(df.to_numpy())
    prob = model.predict_proba(scaled)[:, 1][0]
    return prob, scaled, df

# ── Main area ──
if predict_btn:
    input_dict = build_input()
    prob, scaled, df_aligned = predict(input_dict)

    # Risk category
    if prob < 0.3:
        risk = "🟢 Low Risk"
        color = "green"
    elif prob < 0.6:
        risk = "🟡 Medium Risk"
        color = "orange"
    else:
        risk = "🔴 High Risk"
        color = "red"

    # ── Results row ──
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Churn Probability", f"{prob:.1%}")

    with col2:
        st.metric("Risk Category", risk)

    with col3:
        st.metric("Monthly Charges", f"${monthly_charges:.2f}")

    st.divider()

    # ── Probability bar ──
    st.subheader("Churn Probability Meter")
    st.progress(float(prob))
    st.caption(f"This customer has a **{prob:.1%}** probability of churning")

    st.divider()

    # ── SHAP explanation ──
    st.subheader("🔍 Why did the model make this prediction?")
    try:
        explainer = shap.LinearExplainer(model, scaled)
        shap_values = explainer.shap_values(scaled)

        feature_names = feature_columns
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values[0]
        }).sort_values('SHAP Value', key=abs, ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_df['SHAP Value']]
        ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel("SHAP Value (Red = increases churn risk, Green = decreases)")
        ax.set_title("Top 10 Features Driving This Prediction")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        st.caption("🔴 Red bars push towards churn | 🟢 Green bars push away from churn")
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

    st.divider()

    # ── Customer summary ──
    st.subheader("📋 Customer Profile Summary")
    summary = {
        "Contract Type": contract,
        "Tenure": f"{tenure} months",
        "Monthly Charges": f"${monthly_charges:.2f}",
        "Internet Service": internet_service,
        "Payment Method": payment
    }
    st.table(pd.DataFrame(summary.items(), columns=["Attribute", "Value"]))

else:
    # Default landing state
    st.info("👈 Configure the customer profile in the sidebar and click **Predict Churn**")

    st.subheader("About this App")
    st.markdown("""
    This app uses a **Logistic Regression model** trained on the IBM Telco Customer Churn dataset.

    **Model Performance:**
    - AUC-ROC: 0.84
    - Recall (Churned): 77%
    - Training samples: 5,625 (after SMOTE resampling)

    **Top Churn Drivers identified by SHAP:**
    1. Monthly Charges
    2. Internet Service (Fiber optic)
    3. Tenure
    4. Contract Type
    5. Total Charges
    """)