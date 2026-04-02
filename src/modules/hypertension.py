import joblib
import pandas as pd

MODEL_PATH = "results/hypertension_model.pkl"

FEATURES = [
    "age", "sex", "trestbps", "chol", "fbs", "thalach", "oldpeak", "exang"
]

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "thalach": "Maximum Heart Rate Achieved",
    "oldpeak": "ST Depression (Oldpeak)",
    "exang": "Exercise-Induced Angina",
}

def load_model():
    return joblib.load(MODEL_PATH)

def build_input_df(user: dict) -> pd.DataFrame:
    X = pd.DataFrame([user], columns=FEATURES)
    return X.astype(float)

def predict_risk(model, user: dict) -> float:
    X = build_input_df(user)
    return float(model.predict_proba(X)[:, 1][0])

def get_feature_names(model):
    try:
        pipe = model.calibrated_classifiers_[0].estimator
    except Exception:
        pipe = model.estimators_[0]

    prep = pipe.named_steps["prep"]

    feature_names = []
    for _, _, cols in prep.transformers_:
        feature_names.extend(cols)

    return feature_names

def pretty_feature_name(name: str) -> str:
    return FEATURE_LABELS.get(name, name)