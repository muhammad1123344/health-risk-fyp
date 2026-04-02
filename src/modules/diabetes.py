import joblib
import pandas as pd

MODEL_PATH = "results/diabetes_model.pkl"
FEATURES = joblib.load("results/diabetes_features.pkl")

LABELS = {
    "age": "Age (standardised)",
    "sex": "Sex Indicator (dataset encoded / standardised)",
    "bmi": "BMI (standardised)",
    "bp": "Blood Pressure (standardised)",
    "s1": "Total Serum Cholesterol (standardised)",
    "s2": "LDL Cholesterol (standardised)",
    "s3": "HDL Cholesterol (standardised)",
    "s4": "Total Cholesterol / HDL Ratio (standardised)",
    "s5": "Triglycerides (standardised)",
    "s6": "Blood Sugar (standardised)",
}


def load_model():
    return joblib.load(MODEL_PATH)


def predict_risk(model, user: dict) -> float:
    X = pd.DataFrame([user], columns=FEATURES)
    X = X.astype(float)
    return float(model.predict_proba(X)[:, 1][0])


def pretty_feature_name(name: str) -> str:
    return LABELS.get(name, name)