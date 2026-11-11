
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "ecg_ml_features.csv"
RESULTS = ROOT / "results" / "group_holdout"
RESULTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

labels = ["N", "V", "F"]
metadata = ["record_id", "sample", "beat_type"]
features = [c for c in df.columns if c not in metadata]

X = df[features]
y = df["beat_type"]

# Records 201 and 202 belong to the same patient.
groups = df["record_id"].astype(str).str.replace(
    r"^202$", "201", regex=True
)

splitter = GroupShuffleSplit(
    n_splits=1,
    train_size=0.75,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups))

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

assert train_groups.isdisjoint(test_groups)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("TRAIN groups:", len(train_groups))
print("TEST groups:", len(test_groups))
print("TRAIN beats:", len(X_train))
print("TEST beats:", len(X_test))

print("\nTRAIN class distribution:")
print(y_train.value_counts().reindex(labels, fill_value=0))

print("\nTEST class distribution:")
print(y_test.value_counts().reindex(labels, fill_value=0))

# Do not silently change the split if a class is absent.
if set(y_train.unique()) != set(labels):
    raise ValueError("Training set is missing at least one class.")

if set(y_test.unique()) != set(labels):
    raise ValueError("Test set is missing at least one class.")

pd.DataFrame({
    "record_id": df["record_id"],
    "patient_group": groups,
    "split": np.where(
        np.isin(np.arange(len(df)), train_idx),
        "train",
        "test"
    )
}).to_csv(RESULTS / "split.csv", index=False)

models = {
    "KNN": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    "Linear_SVM": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearSVC(
            C=1,
            class_weight="balanced",
            max_iter=10000,
            random_state=42
        )
    ),
    "RBF_SVM": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1,
            gamma="scale",
            class_weight="balanced",
            cache_size=1000
        )
    ),
    "Random_Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
}

results = []
predictions = []

for name, model in models.items():
    print(f"\nTraining {name}...", flush=True)

    start = perf_counter()
    model.fit(X_train, y_train)
    train_seconds = perf_counter() - start

    pred = model.predict(X_test)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_test,
            pred,
            labels=labels,
            zero_division=0
        )
    )

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(
            y_test, pred, labels=labels, average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_test, pred, labels=labels, average="weighted",
            zero_division=0
        ),
        "training_seconds": train_seconds,
    }

    for i, label in enumerate(labels):
        result[f"precision_{label}"] = precision[i]
        result[f"recall_{label}"] = recall[i]
        result[f"f1_{label}"] = f1[i]

    results.append(result)

    cm = confusion_matrix(y_test, pred, labels=labels)
    pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    ).to_csv(RESULTS / f"cm_{name}.csv")

    predictions.append(pd.DataFrame({
        "model": name,
        "record_id": df.iloc[test_idx]["record_id"].values,
        "true_label": y_test.values,
        "predicted_label": pred
    }))

    print(
        f"Accuracy: {result['accuracy']:.4f} | "
        f"Macro F1: {result['macro_f1']:.4f} | "
        f"Recall N: {recall[0]:.4f} | "
        f"V: {recall[1]:.4f} | "
        f"F: {recall[2]:.4f}",
        flush=True
    )

    pd.DataFrame(results).to_csv(
        RESULTS / "holdout_results.csv", index=False
    )

pd.concat(predictions, ignore_index=True).to_csv(
    RESULTS / "holdout_predictions.csv", index=False
)

print("\nFINAL HOLDOUT RESULTS")
print(pd.DataFrame(results).to_string(index=False))
