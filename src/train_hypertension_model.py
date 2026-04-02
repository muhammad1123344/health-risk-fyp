import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

# Use heart dataset and derive a hypertension screening label
df = pd.read_csv("data/raw/heart.csv").drop_duplicates()

# Proxy label: elevated resting BP
df["hypertension_risk"] = (df["trestbps"] >= 140).astype(int)

FEATURES = ["age", "sex", "trestbps", "chol", "fbs", "thalach", "oldpeak", "exang"]

X = df[FEATURES]
y = df["hypertension_risk"]

num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
cat_cols = ["sex", "fbs", "exang"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

prep = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
])

base_model = Pipeline([
    ("prep", prep),
    ("logreg", LogisticRegression(max_iter=2000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]

print("Hypertension Risk Model")
print("ROC-AUC:", roc_auc_score(y_test, proba))
print("Brier:", brier_score_loss(y_test, proba))

joblib.dump(model, "results/hypertension_model.pkl")
joblib.dump(FEATURES, "results/hypertension_features.pkl")

print("Saved: results/hypertension_model.pkl")
print("Saved: results/hypertension_features.pkl")
