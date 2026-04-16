import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from preprocess import load_and_clean, encode_features, engineer_features, split_and_scale

def apply_smote(X_train, y_train):
    y_train = y_train.astype(int)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE — Class distribution: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res

def evaluate_model(name, model, X_test, y_test):
    y_test = y_test.astype(int)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))
    return auc

def train_all(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1,
                                  use_label_encoder=False,
                                  eval_metric='logloss', random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        auc = evaluate_model(name, model, X_test, y_test)
        results[name] = (auc, model)

    # Pick best model
    best_name = max(results, key=lambda k: results[k][0])
    best_model = results[best_name][1]
    print(f"\n✅ Best Model: {best_name} with AUC-ROC: {results[best_name][0]:.4f}")

    # Save best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    print("✅ Model saved to models/best_model.pkl")

    return best_model, best_name

if __name__ == "__main__":
    df = load_and_clean('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = encode_features(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, features = split_and_scale(df)

    print("Applying SMOTE...")
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    print("\nTraining models...")
    best_model, best_name = train_all(X_train_res, X_test, y_train_res, y_test)