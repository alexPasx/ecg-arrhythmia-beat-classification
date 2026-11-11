
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RECORDS = wfdb.get_record_list("mitdb")
CLASSES = {"N", "V", "F"}

# All annotated beat types used for RR calculations
BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S",
    "V", "r", "F", "e", "j", "n", "E",
    "/", "f", "Q", "?"
}

PRE_SECONDS = 0.250
POST_SECONDS = 0.450

rows = []

for record_id in RECORDS:
    record = wfdb.rdrecord(str(RAW_DIR / record_id))
    ann = wfdb.rdann(str(RAW_DIR / record_id), "atr")

    fs = record.fs
    ecg = record.p_signal[:, 0]

    sos = butter(
        3, [5, 18],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    filtered = sosfiltfilt(sos, ecg)

    pre = int(PRE_SECONDS * fs)
    post = int(POST_SECONDS * fs)

    beat_samples = np.array([
        sample
        for sample, symbol in zip(
            ann.sample, ann.symbol
        )
        if symbol in BEAT_SYMBOLS
    ])

    record_count = 0

    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol not in CLASSES:
            continue

        start = sample - pre
        end = sample + post

        if start < 0 or end > len(filtered):
            continue

        segment = filtered[start:end]

        prev_idx = np.searchsorted(
            beat_samples, sample
        ) - 1

        next_idx = np.searchsorted(
            beat_samples, sample, side="right"
        )

        rr_prev = (
            (sample - beat_samples[prev_idx]) / fs
            if prev_idx >= 0
            else np.nan
        )

        rr_next = (
            (beat_samples[next_idx] - sample) / fs
            if next_idx < len(beat_samples)
            else np.nan
        )

        slope = np.diff(segment) * fs

        features = {
            "record_id": record_id,
            "sample": int(sample),
            "beat_type": symbol,
            "rr_prev": rr_prev,
            "rr_next": rr_next,
            "amplitude_range": np.ptp(segment),
            "rms": np.sqrt(np.mean(segment ** 2)),
            "std": np.std(segment),
            "energy": np.sum(segment ** 2),
            "max_slope": np.max(np.abs(slope)),
            "mean_abs_slope": np.mean(np.abs(slope))
        }

        # Add waveform morphology samples
        for i, value in enumerate(segment):
            features[f"segment_{i}"] = float(value)

        rows.append(features)
        record_count += 1

    print(
        f"Record {record_id}: "
        f"{record_count} beats extracted"
    )

df = pd.DataFrame(rows)

output_path = OUTPUT_DIR / "ecg_ml_features.csv"
df.to_csv(output_path, index=False)

feature_cols = [
    col for col in df.columns
    if col not in {"record_id", "sample", "beat_type"}
]

print("\nFEATURE EXTRACTION SUMMARY")
print("-" * 50)
print("Dataset shape:", df.shape)
print("Number of features:", len(feature_cols))

print("\nClass distribution:")
print(df["beat_type"].value_counts())

print("\nRecords:", df["record_id"].nunique())

print("\nMissing values:")
print(df[feature_cols].isna().sum().sum())

print("\nSaved to:", output_path)
