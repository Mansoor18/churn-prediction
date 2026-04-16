import pandas as pd
import numpy as np
import joblib

def predict_churn(input_dict):
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')

    df = pd.DataFrame([input_dict])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # --- Encode binary columns ---
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    df[binary_cols] = df[binary_cols].replace(binary_map)

    # --- One-hot encode multi-category columns ---
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # --- Engineer features ---
    df['tenure_group'] = pd.cut(df['tenure'],
                                bins=[0, 12, 24, 48, 60, 72],
                                labels=[1, 2, 3, 4, 5])
    df['tenure_group'] = df['tenure_group'].astype(int)
    df['charge_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

    # --- Align to training columns ---
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    # --- Scale using numpy to avoid name mismatch ---
    df_scaled = scaler.transform(df.to_numpy())

    prob = model.predict_proba(df_scaled)[:, 1][0]
    prediction = "Will Churn" if prob >= 0.5 else "Will Not Churn"

    print(f"Churn Probability : {prob:.2%}")
    print(f"Prediction        : {prediction}")
    return prob, prediction

if __name__ == "__main__":
    sample = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.0,
        'TotalCharges': '170.0'
    }
    predict_churn(sample)