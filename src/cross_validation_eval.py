import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

df = pd.read_csv("data/raw/heart.csv")

y = df["target"]
X = df.drop(columns=["target"])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=2000))
])

lr_scores = cross_val_score(lr, X, y, cv=cv, scoring="roc_auc")

# Gradient Boosting
gb = HistGradientBoostingClassifier(random_state=42)
gb_scores = cross_val_score(gb, X, y, cv=cv, scoring="roc_auc")

print("Logistic Regression CV ROC-AUC:", np.mean(lr_scores))
print("Gradient Boosting CV ROC-AUC:", np.mean(gb_scores))