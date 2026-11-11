
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RECORDS = ["100", "106", "119", "200", "208"]
CLASSES = {"N", "V", "F"}

# 250 ms before and 450 ms after each beat
PRE_SECONDS = 0.250
POST_SECONDS = 0.450

segments = []
metadata = []

for record_id in RECORDS:
    record = wfdb.rdrecord(
        str(RAW_DIR / record_id)
    )
    ann = wfdb.rdann(
        str(RAW_DIR / record_id), "atr"
    )

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
        if symbol in {
            "N", "L", "R", "B", "A", "a",
            "J", "S", "V", "r", "F",
            "e", "j", "n", "E", "/",
            "f", "Q", "?"
        }
    ])

    for sample, symbol in zip(
        ann.sample, ann.symbol
    ):
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
            beat_samples, sample,
            side="right"
        )

        rr_prev = (
            (sample - beat_samples[prev_idx]) / fs
            if prev_idx >= 0 else np.nan
        )
        rr_next = (
            (beat_samples[next_idx] - sample) / fs
            if next_idx < len(beat_samples)
            else np.nan
        )

        segments.append(segment)

        metadata.append({
            "record_id": record_id,
            "sample": sample,
            "beat_type": symbol,
            "rr_prev": rr_prev,
            "rr_next": rr_next
        })

    print(f"Processed record {record_id}")

X = np.asarray(segments, dtype=np.float32)
df = pd.DataFrame(metadata)

np.save(OUTPUT_DIR / "beat_segments.npy", X)
df.to_csv(
    OUTPUT_DIR / "beat_metadata.csv",
    index=False
)

print("\nDATASET SUMMARY")
print("Shape:", X.shape)
print("\nClass distribution:")
print(df["beat_type"].value_counts())
print("\nRecords:")
print(df.groupby(
    ["record_id", "beat_type"]
).size().unstack(fill_value=0))
