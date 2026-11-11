
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from scipy.signal import butter, sosfiltfilt, find_peaks

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RECORD = "208"

record = wfdb.rdrecord(str(RAW_DIR / RECORD))
annotation = wfdb.rdann(str(RAW_DIR / RECORD), "atr")

fs = record.fs
ecg = record.p_signal[:, 0]

# Same preprocessing as baseline
sos = butter(
    3, [5, 18],
    btype="bandpass",
    fs=fs,
    output="sos"
)

filtered = sosfiltfilt(sos, ecg)
derivative = np.gradient(filtered) * fs
squared = derivative ** 2

window = int(0.150 * fs)
integrated = np.convolve(
    squared,
    np.ones(window) / window,
    mode="same"
)

BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S",
    "V", "r", "F", "e", "j", "n", "E",
    "/", "f", "Q", "?"
}

reference_samples = np.array([
    sample
    for sample, symbol in zip(
        annotation.sample, annotation.symbol
    )
    if symbol in BEAT_SYMBOLS
])

reference_symbols = np.array([
    symbol
    for symbol in annotation.symbol
    if symbol in BEAT_SYMBOLS
])

reference_times = reference_samples / fs

def evaluate(threshold):
    peaks, _ = find_peaks(
        integrated,
        height=threshold,
        distance=int(0.20 * fs)
    )

    detected_times = peaks / fs
    candidates = []

    for i, d in enumerate(detected_times):
        left = np.searchsorted(
            reference_times, d - 0.15, side="left"
        )
        right = np.searchsorted(
            reference_times, d + 0.15, side="right"
        )

        for j in range(left, right):
            candidates.append((
                abs(d - reference_times[j]), i, j
            ))

    candidates.sort()

    matched_detection = set()
    matched_reference = set()

    for error, i, j in candidates:
        if (
            i not in matched_detection
            and j not in matched_reference
        ):
            matched_detection.add(i)
            matched_reference.add(j)

    tp = len(matched_detection)
    fp = len(peaks) - tp
    fn = len(reference_samples) - tp

    sensitivity = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0

    f1 = (
        2 * sensitivity * precision
        / (sensitivity + precision)
        if sensitivity + precision else 0
    )

    pvc_indices = np.where(reference_symbols == "V")[0]
    pvc_total = len(pvc_indices)
    pvc_tp = sum(
        j in matched_reference for j in pvc_indices
    )

    return {
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "sensitivity": sensitivity,
        "precision": precision,
        "F1": f1,
        "PVC_detected": pvc_tp,
        "PVC_missed": pvc_total - pvc_tp,
        "PVC_sensitivity": (
            pvc_tp / pvc_total if pvc_total else 0
        )
    }


baseline = 0.10 * np.max(integrated)

thresholds = [
    baseline,
    150,
    140,
    130,
    120,
    100
]

results = pd.DataFrame([
    evaluate(th) for th in thresholds
])

print("\nTHRESHOLD OPTIMIZATION")
print("-" * 80)
print(results.round(4).to_string(index=False))

results.to_csv(
    RESULTS_DIR / "threshold_optimization.csv",
    index=False
)
