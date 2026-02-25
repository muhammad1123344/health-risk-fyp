import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

df = pd.read_csv("data/raw/heart.csv")

y = df["target"]
X = df.drop(columns=["target"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Base model
base_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=2000))
])

# Calibrated model (isotonic regression)
calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
calibrated_model.fit(X_train, y_train)

proba = calibrated_model.predict_proba(X_test)[:, 1]

print("Calibrated ROC-AUC:", roc_auc_score(y_test, proba))
print("Brier Score:", brier_score_loss(y_test, proba))

# Calibration curve plot
prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curve")
plt.savefig("results/calibration_plot.png")
print("Calibration plot saved in results/")