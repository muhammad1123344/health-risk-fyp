import pandas as pd

df = pd.read_csv("data/raw/heart.csv")

print("Shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

# Check duplicates ignoring target (same features but maybe repeated samples)
X = df.drop(columns=["target"])
print("Duplicate feature rows (ignoring target):", X.duplicated().sum())