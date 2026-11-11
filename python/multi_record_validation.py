
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from scipy.signal import butter, sosfiltfilt, find_peaks

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"

RAW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RECORDS = ["100", "106", "119", "200"]

BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S",
    "V", "r", "F", "e", "j", "n", "E",
    "/", "f", "Q", "?"
}


def load_record(record_id):
    path = RAW_DIR / record_id

    # Download only when files are missing
    required = [
        path.with_suffix(".hea"),
        path.with_suffix(".dat"),
        path.with_suffix(".atr")
    ]

    if not all(p.exists() for p in required):
        print(f"Downloading record {record_id}...")

        wfdb.dl_database(
            "mitdb",
            dl_dir=str(RAW_DIR),
            records=[record_id],
            annotators=["atr"]
        )

    record = wfdb.rdrecord(str(path))
    annotation = wfdb.rdann(str(path), "atr")

    return record, annotation


def preprocess(ecg, fs):
    sos = butter(
        3,
        [5, 18],
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

    return integrated


def evaluate(integrated, annotation, fs, threshold):
    peaks, _ = find_peaks(
        integrated,
        height=threshold,
        distance=int(0.20 * fs)
    )

    detected_times = peaks / fs

    reference = [
        (sample, symbol)
        for sample, symbol in zip(
            annotation.sample,
            annotation.symbol
        )
        if symbol in BEAT_SYMBOLS
    ]

    reference_samples = np.array(
        [sample for sample, _ in reference]
    )

    reference_symbols = np.array(
        [symbol for _, symbol in reference]
    )

    reference_times = reference_samples / fs

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
                abs(d - reference_times[j]),
                i,
                j
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

    pvc_indices = np.where(
        reference_symbols == "V"
    )[0]

    pvc_total = len(pvc_indices)

    pvc_detected = sum(
        j in matched_reference
        for j in pvc_indices
    )

    return {
        "threshold": threshold,
        "reference_beats": len(reference_samples),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "sensitivity": sensitivity,
        "precision": precision,
        "F1": f1,
        "PVC_total": pvc_total,
        "PVC_detected": pvc_detected,
        "PVC_missed": pvc_total - pvc_detected,
        "PVC_sensitivity": (
            pvc_detected / pvc_total
            if pvc_total else np.nan
        )
    }


results = []

for record_id in RECORDS:
    print(f"\nProcessing record {record_id}...")

    record, annotation = load_record(record_id)

    fs = record.fs
    ecg = record.p_signal[:, 0]

    integrated = preprocess(ecg, fs)

    baseline_threshold = (
        0.10 * np.max(integrated)
    )

    for method, threshold in [
        ("Baseline", baseline_threshold),
        ("Candidate_100", 100.0)
    ]:
        metrics = evaluate(
            integrated,
            annotation,
            fs,
            threshold
        )

        results.append({
            "record": record_id,
            "method": method,
            **metrics
        })


df = pd.DataFrame(results)

print("\nMULTI-RECORD VALIDATION")
print("=" * 100)

print(
    df.round(4).to_string(index=False)
)

output = RESULTS_DIR / "multi_record_validation.csv"

df.to_csv(output, index=False)

print(f"\nResults saved to: {output}")
