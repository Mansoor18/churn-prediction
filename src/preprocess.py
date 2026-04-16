import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_and_clean(path):
    df = pd.read_csv(path)
    df.drop(columns=['customerID'], inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    return df

def encode_features(df, is_training=True):
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    if is_training and 'Churn' in df.columns:
        binary_cols.append('Churn')

    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    df[binary_cols] = df[binary_cols].replace(binary_map)

    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract',
                  'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    return df

def engineer_features(df):
    df['tenure_group'] = pd.cut(df['tenure'],
                                bins=[0, 12, 24, 48, 60, 72],
                                labels=[1, 2, 3, 4, 5])
    df['tenure_group'] = df['tenure_group'].astype(int)
    df['charge_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    return df

def split_and_scale(df):
    X = df.drop(columns=['Churn'])
    y = df['Churn'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()  # ← must be here
    X = df.drop(columns=['Churn'])
    y = df['Churn'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')