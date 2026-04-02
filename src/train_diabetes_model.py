import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

df = pd.read_csv("data/raw/diabetes_sklearn.csv")

y = df["risk"]
X = df.drop(columns=["risk"])

num_cols = list(X.columns)

base = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=2000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = CalibratedClassifierCV(base, method="isotonic", cv=5)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]

print("Diabetes Risk Model (calibrated LR)")
print("ROC-AUC:", roc_auc_score(y_test, proba))
print("Brier:", brier_score_loss(y_test, proba))

joblib.dump(model, "results/diabetes_model.pkl")
joblib.dump(list(X.columns), "results/diabetes_features.pkl")
print("Saved results/diabetes_model.pkl and results/diabetes_features.pkl")