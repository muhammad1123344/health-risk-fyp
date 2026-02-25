import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

df = pd.read_csv("data/raw/heart.csv")

# 🔥 Remove duplicates
df = df.drop_duplicates()

print("New shape after removing duplicates:", df.shape)

y = df["target"]
X = df.drop(columns=["target"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=2000))
])

lr_cal = CalibratedClassifierCV(lr, method="isotonic", cv=5)
lr_cal.fit(X_train, y_train)
lr_proba = lr_cal.predict_proba(X_test)[:, 1]

# Gradient Boosting
gb = HistGradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
gb_proba = gb.predict_proba(X_test)[:, 1]

print("\nLogistic Regression (Calibrated)")
print("ROC-AUC:", roc_auc_score(y_test, lr_proba))
print("Brier Score:", brier_score_loss(y_test, lr_proba))

print("\nGradient Boosting")
print("ROC-AUC:", roc_auc_score(y_test, gb_proba))
print("Brier Score:", brier_score_loss(y_test, gb_proba))