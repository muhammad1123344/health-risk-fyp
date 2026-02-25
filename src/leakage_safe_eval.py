import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv("data/raw/heart.csv")

y = df["target"]
X = df.drop(columns=["target"])

# Stronger split check: different random seeds
for seed in [1, 7, 21, 42, 99]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    gb = HistGradientBoostingClassifier(random_state=seed)
    gb.fit(X_train, y_train)
    proba = gb.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"Seed {seed}: ROC-AUC = {auc:.4f}")