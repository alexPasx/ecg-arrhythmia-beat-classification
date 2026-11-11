
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "ecg_ml_features.csv"
FOLDS_PATH = ROOT / "data" / "processed" / "ecg_cv_folds.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CLASSES = ["N", "V", "F"]
TEST_FOLD = 1

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    dtype={"record_id": str}
)

folds = pd.read_csv(
    FOLDS_PATH,
    dtype={"record_id": str}
)

# Verify that rows match the saved fold assignments
assert len(df) == len(folds)

assert (
    df[["record_id", "sample", "beat_type"]]
    .reset_index(drop=True)
    .equals(
        folds[["record_id", "sample", "beat_type"]]
        .reset_index(drop=True)
    )
), "Dataset and folds do not match."

feature_cols = [
    col for col in df.columns
    if col not in ["record_id", "sample", "beat_type"]
]

X = df[feature_cols]
y = df["beat_type"]

train_mask = folds["fold"] != TEST_FOLD
test_mask = folds["fold"] == TEST_FOLD

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

# Records 201 and 202 represent the same patient
groups = df["record_id"].replace({"202": "201"})

assert set(groups[train_mask]).isdisjoint(
    set(groups[test_mask])
)

print("Training beats:", len(X_train))
print("Validation beats:", len(X_test))
print("Features:", X.shape[1])

# --------------------------------------------------
# 2. Define classifiers
# --------------------------------------------------

models = {
    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            n_jobs=-1
        ))
    ]),

    "Linear_SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10000,
            random_state=42
        ))
    ]),

    "RBF_SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            cache_size=1000
        ))
    ]),

    "Random_Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# --------------------------------------------------
# 3. Train and evaluate
# --------------------------------------------------

results = []

for name, model in models.items():

    print(f"\n{'=' * 55}")
    print(f"MODEL: {name}")
    print(f"{'=' * 55}", flush=True)

    start = perf_counter()

    model.fit(X_train, y_train)

    training_time = perf_counter() - start

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=CLASSES,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        labels=CLASSES,
        average="weighted",
        zero_division=0
    )

    recalls = recall_score(
        y_test,
        predictions,
        labels=CLASSES,
        average=None,
        zero_division=0
    )

    print(f"Training time: {training_time:.2f} sec")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nRecall by class:")
    for label, recall in zip(CLASSES, recalls):
        print(f"{label}: {recall:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=CLASSES,
        zero_division=0
    ))

    print("\nConfusion matrix:")
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=CLASSES
    )
    print(cm)

    results.append({
        "model": name,
        "fold": TEST_FOLD,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "recall_N": recalls[0],
        "recall_V": recalls[1],
        "recall_F": recalls[2],
        "training_seconds": training_time
    })

    pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES
    ).to_csv(
        RESULTS_DIR / f"cm_{name}_fold{TEST_FOLD}.csv"
    )

# --------------------------------------------------
# 4. Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS_DIR / "ml_baseline_fold1.csv",
    index=False
)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))
