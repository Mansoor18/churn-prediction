## 🚀 Live Demo
👉 https://churn-prediction-vi6wq3r4k58haxnitydf8z.streamlit.app/
# Customer Churn Prediction

Binary classification model to predict telecom customer churn.

## Results
| Model | AUC-ROC | F1 (Churned) |
|---|---|---|
| Logistic Regression | 0.8356 | 0.61 |
| XGBoost | 0.8300 | 0.59 |
| Random Forest | 0.8222 | 0.58 |

## Key Findings
- Contract type, tenure, and monthly charges are top churn drivers
- Engineered feature `charge_per_tenure` ranked in top 10 by SHAP
- SMOTE improved churned customer recall from 61% to 77%

## Project Structure

churn-prediction/
├── data/               # Raw dataset (not tracked)
├── notebooks/          # EDA notebook
├── src/
│   ├── preprocess.py   # Cleaning & feature engineering
│   ├── train.py        # Model training & evaluation
│   ├── explain.py      # SHAP explainability
│   └── predict.py      # Inference on new customers
├── models/             # Saved model files (not tracked)
└── requirements.txt

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/train.py
python src/predict.py
```

## Tech Stack
Python · Scikit-learn · XGBoost · SHAP · SMOTE · Pandas · Matplotlib