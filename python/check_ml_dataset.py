
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

path = ROOT / "data" / "processed" / "ecg_ml_features.csv"

df = pd.read_csv(path, dtype={"record_id": str})

feature_cols = [
    col for col in df.columns
    if col not in ["record_id", "sample", "beat_type"]
]

X = df[feature_cols]

print("\n1. DATASET SHAPE")
print(df.shape)

print("\n2. MISSING VALUES BY FEATURE")
missing = X.isna().sum()
print(missing[missing > 0])

print("\n3. INFINITE VALUES")
print(np.isinf(X.to_numpy()).sum())

print("\n4. CLASS DISTRIBUTION")
print(df["beat_type"].value_counts())

print("\n5. RECORDS PER CLASS")
distribution = pd.crosstab(
    df["record_id"],
    df["beat_type"]
)
print(distribution)

print("\n6. DUPLICATE RECORD/SAMPLE PAIRS")
print(df.duplicated(["record_id", "sample"]).sum())

print("\n7. FEATURES")
print(len(feature_cols))
