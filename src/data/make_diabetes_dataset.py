import pandas as pd
from sklearn.datasets import load_diabetes

def main():
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()

    # Convert regression target into binary risk label (top 25% = high risk)
    threshold = df["target"].quantile(0.75)
    df["risk"] = (df["target"] >= threshold).astype(int)
    df = df.drop(columns=["target"])

    df.to_csv("data/raw/diabetes_sklearn.csv", index=False)
    print("Saved: data/raw/diabetes_sklearn.csv")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()