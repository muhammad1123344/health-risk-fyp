import joblib
import pandas as pd

MODEL_PATH = "results/heart_model.pkl"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Maximum Heart Rate Achieved",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "slope": "Slope of Peak Exercise ST Segment",
    "ca": "Number of Major Vessels",
    "thal": "Thalassemia Status",
}

CP_MAP = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3,
}

RESTECG_MAP = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2,
}

SLOPE_MAP = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2,
}

THAL_MAP = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2,
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