import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
from preprocess import load_and_clean, encode_features, engineer_features, split_and_scale

def explain_model():
    # Load data
    df = load_and_clean('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = encode_features(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, features = split_and_scale(df)

    # Load model
    model = joblib.load('models/best_model.pkl')

    # SHAP explainer
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    plt.figure()
    shap.summary_plot(
        shap_values, X_test,
        feature_names=features,
        show=False
    )
    plt.tight_layout()
    plt.savefig('models/shap_summary.png', dpi=150, bbox_inches='tight')
    print("✅ SHAP summary saved to models/shap_summary.png")

    # Top 10 features
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_shap)[::-1][:10]
    print("\nTop 10 Most Important Features:")
    for i in top_idx:
        print(f"  {features[i]}: {mean_shap[i]:.4f}")

if __name__ == "__main__":
    explain_model()