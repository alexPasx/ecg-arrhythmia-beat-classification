
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "ecg_ml_features.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "ecg_cv_folds.csv"

df = pd.read_csv(DATA_PATH, dtype={"record_id": str})

# Grouping by record, not by beat.
# Records 201 and 202 belong to the same patient.
groups = df["record_id"].replace({"202": "201"})

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

df["fold"] = -1

for fold, (train_idx, val_idx) in enumerate(
    cv.split(
        df,
        df["beat_type"],
        groups
    ),
    start=1
):
    train_groups = set(groups.iloc[train_idx])
    val_groups = set(groups.iloc[val_idx])

    assert train_groups.isdisjoint(val_groups)

    df.loc[val_idx, "fold"] = fold

    print(f"\nFOLD {fold}")
    print("-" * 45)

    print("Training beats:", len(train_idx))
    print("Validation beats:", len(val_idx))

    print(
        "Validation records:",
        sorted(df.iloc[val_idx]["record_id"].unique())
    )

    print("\nValidation class distribution:")
    print(df.iloc[val_idx]["beat_type"].value_counts())

assert (df["fold"] > 0).all()

# Save fold assignments separately from the features.
folds = df[["record_id", "sample", "beat_type", "fold"]]
folds.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)
