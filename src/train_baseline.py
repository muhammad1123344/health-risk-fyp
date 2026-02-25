import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

df = pd.read_csv("data/raw/heart.csv")

# Features + label
y = df["target"]
X = df.drop(columns=["target"])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model: scale + logistic regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=2000))
])

# Train
model.fit(X_train, y_train)

# Predict
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

# Evaluate
print("ROC-AUC:", roc_auc_score(y_test, proba))
print("\nClassification Report:\n")
print(classification_report(y_test, pred))