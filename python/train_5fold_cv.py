
from pathlib import Path
from time import perf_counter
import warnings

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
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/processed/ecg_ml_features.csv"
FOLDS_PATH = ROOT / "data/processed/ecg_cv_folds.csv"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["N", "V", "F"]
N_FOLDS = 5

RESULTS_PATH = RESULTS_DIR / "ml_5fold_results.csv"
PREDICTIONS_PATH = RESULTS_DIR / "ml_5fold_predictions.csv"

# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading ECG dataset...", flush=True)

df = pd.read_csv(
    DATA_PATH,
    dtype={"record_id": str}
)

folds = pd.read_csv(
    FOLDS_PATH,
    dtype={"record_id": str}
)

keys = ["record_id", "sample", "beat_type"]

assert len(df) == len(folds), "Dataset size mismatch"

assert (
    df[keys].reset_index(drop=True)
    .equals(folds[keys].reset_index(drop=True))
), "Dataset and folds are not aligned"

feature_cols = [
    c for c in df.columns
    if c not in keys
]

X = df[feature_cols]
y = df["beat_type"]

groups = df["record_id"].replace({"202": "201"})

print(f"Total beats: {len(df):,}")
print(f"Features: {len(feature_cols)}")
print(f"Folds: {N_FOLDS}")
print("\nClass distribution:")
print(y.value_counts())

assert len(feature_cols) == 260
assert set(folds["fold"].unique()) == set(range(1, N_FOLDS + 1))

# ==========================================
# 3. DEFINE MODELS
# ==========================================

def create_models():
    return {
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

# ==========================================
# 4. CROSS-VALIDATION
# ==========================================

all_results = []
all_predictions = []

total_start = perf_counter()

for fold in range(1, N_FOLDS + 1):

    fold_start = perf_counter()

    print("\n" + "=" * 65, flush=True)
    print(f"STARTING FOLD {fold}/{N_FOLDS}", flush=True)
    print("=" * 65, flush=True)

    train_mask = folds["fold"] != fold
    val_mask = folds["fold"] == fold

    assert set(groups[train_mask]).isdisjoint(
        set(groups[val_mask])
    ), "Patient/record leakage detected"

    X_train = X.loc[train_mask]
    X_val = X.loc[val_mask]

    y_train = y.loc[train_mask]
    y_val = y.loc[val_mask]

    print(f"Training beats: {len(X_train):,}")
    print(f"Validation beats: {len(X_val):,}")
    print("Validation class distribution:")
    print(y_val.value_counts().reindex(CLASSES, fill_value=0))
    print(flush=True)

    models = create_models()

    for model_index, (name, model) in enumerate(
        models.items(), start=1
    ):

        print(
            f"[Fold {fold}/5] "
            f"Model {model_index}/4: {name} — training...",
            flush=True
        )

        start = perf_counter()

        model.fit(X_train, y_train)

        training_seconds = perf_counter() - start

        prediction_start = perf_counter()
        y_pred = model.predict(X_val)
        prediction_seconds = perf_counter() - prediction_start

        accuracy = accuracy_score(y_val, y_pred)

        macro_f1 = f1_score(
            y_val, y_pred,
            labels=CLASSES,
            average="macro",
            zero_division=0
        )

        weighted_f1 = f1_score(
            y_val, y_pred,
            labels=CLASSES,
            average="weighted",
            zero_division=0
        )

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_val, y_pred,
                labels=CLASSES,
                zero_division=0
            )
        )

        result = {
            "fold": fold,
            "model": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "precision_N": precision[0],
            "precision_V": precision[1],
            "precision_F": precision[2],
            "recall_N": recall[0],
            "recall_V": recall[1],
            "recall_F": recall[2],
            "f1_N": f1[0],
            "f1_V": f1[1],
            "f1_F": f1[2],
            "training_seconds": training_seconds,
            "prediction_seconds": prediction_seconds
        }

        all_results.append(result)

        val_rows = df.loc[
            val_mask, ["record_id", "sample", "beat_type"]
        ].copy()

        val_rows["fold"] = fold
        val_rows["model"] = name
        val_rows["prediction"] = y_pred

        all_predictions.append(val_rows)

        # Save immediately after each model
        pd.DataFrame(all_results).to_csv(
            RESULTS_PATH,
            index=False
        )

        pd.concat(
            all_predictions,
            ignore_index=True
        ).to_csv(
            PREDICTIONS_PATH,
            index=False
        )

        cm = confusion_matrix(
            y_val, y_pred,
            labels=CLASSES
        )

        pd.DataFrame(
            cm,
            index=CLASSES,
            columns=CLASSES
        ).to_csv(
            RESULTS_DIR / f"cm_{name}_fold{fold}.csv"
        )

        print(
            f"  Accuracy: {accuracy:.4f} | "
            f"Macro F1: {macro_f1:.4f} | "
            f"Recall N: {recall[0]:.4f} | "
            f"V: {recall[1]:.4f} | "
            f"F: {recall[2]:.4f}",
            flush=True
        )

        print(
            f"  Training: {training_seconds:.1f}s | "
            f"Prediction: {prediction_seconds:.1f}s",
            flush=True
        )

    # ======================================
    # 5. PROGRESS AFTER EACH FOLD
    # ======================================

    fold_results = pd.DataFrame(all_results)

    current = fold_results[
        fold_results["fold"] == fold
    ][[
        "model",
        "accuracy",
        "macro_f1",
        "recall_N",
        "recall_V",
        "recall_F"
    ]]

    print("\n" + "-" * 65, flush=True)
    print(f"FOLD {fold} COMPLETED", flush=True)
    print("-" * 65, flush=True)

    print(
        current.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        ),
        flush=True
    )

    print(
        f"\nFold duration: "
        f"{perf_counter() - fold_start:.1f}s",
        flush=True
    )

    print(
        f"Overall progress: {fold}/{N_FOLDS} "
        f"folds ({fold / N_FOLDS * 100:.0f}%)",
        flush=True
    )

    print("\nMean scores so far:")
    print(
        fold_results.groupby("model")[
            ["accuracy", "macro_f1", "recall_V", "recall_F"]
        ].mean().round(4).to_string(),
        flush=True
    )

# ==========================================
# 6. FINAL SUMMARY
# ==========================================

results_df = pd.DataFrame(all_results)

summary = results_df.groupby("model").agg(
    mean_accuracy=("accuracy", "mean"),
    std_accuracy=("accuracy", "std"),
    mean_macro_f1=("macro_f1", "mean"),
    std_macro_f1=("macro_f1", "std"),
    mean_recall_N=("recall_N", "mean"),
    mean_recall_V=("recall_V", "mean"),
    mean_recall_F=("recall_F", "mean"),
    total_training_seconds=("training_seconds", "sum")
)

summary = summary.sort_values(
    "mean_macro_f1",
    ascending=False
)

summary.to_csv(
    RESULTS_DIR / "ml_5fold_summary.csv"
)

# ==========================================
# 7. POOLED OUT-OF-FOLD EVALUATION
# ==========================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

pooled_results = []

for name in create_models():

    model_preds = predictions_df[
        predictions_df["model"] == name
    ]

    assert len(model_preds) == len(df)

    pooled_results.append({
        "model": name,
        "accuracy": accuracy_score(
            model_preds["beat_type"],
            model_preds["prediction"]
        ),
        "macro_f1": f1_score(
            model_preds["beat_type"],
            model_preds["prediction"],
            labels=CLASSES,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            model_preds["beat_type"],
            model_preds["prediction"],
            labels=CLASSES,
            average="weighted",
            zero_division=0
        )
    })

    cm = confusion_matrix(
        model_preds["beat_type"],
        model_preds["prediction"],
        labels=CLASSES
    )

    pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES
    ).to_csv(
        RESULTS_DIR / f"cm_{name}_pooled.csv"
    )

pooled_df = pd.DataFrame(pooled_results)

pooled_df.to_csv(
    RESULTS_DIR / "ml_5fold_pooled.csv",
    index=False
)

print("\n" + "=" * 65)
print("FINAL 5-FOLD RESULTS")
print("=" * 65)

print(summary.round(4).to_string())

print("\nPOOLED OUT-OF-FOLD RESULTS")
print(pooled_df.round(4).to_string(index=False))

print(
    f"\nTotal runtime: "
    f"{(perf_counter() - total_start) / 60:.2f} minutes"
)

print(f"\nResults saved to: {RESULTS_DIR}")
