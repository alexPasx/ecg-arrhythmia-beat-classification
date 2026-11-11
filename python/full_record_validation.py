
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

from scipy.signal import butter, sosfiltfilt, find_peaks

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RECORD = "208"

# 1. Load complete ECG recording
record = wfdb.rdrecord(str(RAW_DIR / RECORD))
annotation = wfdb.rdann(str(RAW_DIR / RECORD), "atr")

fs = int(record.fs)
ecg = record.p_signal[:, 0]
duration = len(ecg) / fs

print(f"Record: {RECORD}")
print(f"Sampling frequency: {fs} Hz")
print(f"Duration: {duration / 60:.2f} minutes")

# 2. Bandpass filtering
sos = butter(
    3,
    [5, 18],
    btype="bandpass",
    fs=fs,
    output="sos"
)

filtered = sosfiltfilt(sos, ecg)

# 3. QRS signal transformation
derivative = np.gradient(filtered) * fs
squared = derivative ** 2

window = int(0.150 * fs)
integrated = np.convolve(
    squared,
    np.ones(window) / window,
    mode="same"
)

# 4. Initial detection
threshold = 0.10 * np.max(integrated)

peaks, properties = find_peaks(
    integrated,
    height=threshold,
    distance=int(0.20 * fs)
)

detected_times = peaks / fs

# 5. Reference beat annotations
BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S",
    "V", "r", "F", "e", "j", "n", "E",
    "/", "f", "Q", "?"
}

reference_samples = np.array([
    sample
    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol
    )
    if symbol in BEAT_SYMBOLS
])

reference_times = reference_samples / fs

# 6. One-to-one temporal matching
tolerance = 0.15

matched_reference = set()
matched_detection = set()

candidates = []

for i, d in enumerate(detected_times):
    left = np.searchsorted(
        reference_times, d - tolerance, side="left"
    )
    right = np.searchsorted(
        reference_times, d + tolerance, side="right"
    )

    for j in range(left, right):
        candidates.append((
            abs(d - reference_times[j]), i, j
        ))

candidates.sort()

for error, i, j in candidates:
    if i not in matched_detection and j not in matched_reference:
        matched_detection.add(i)
        matched_reference.add(j)

tp = len(matched_detection)
fp = len(peaks) - tp
fn = len(reference_samples) - tp

sensitivity = tp / (tp + fn) if tp + fn else 0
precision = tp / (tp + fp) if tp + fp else 0
f1 = (
    2 * sensitivity * precision / (sensitivity + precision)
    if sensitivity + precision else 0
)


# Analyze missed beats by annotation type

reference_symbols = np.array([
    symbol
    for symbol in annotation.symbol
    if symbol in BEAT_SYMBOLS
])

missed_indices = sorted(
    set(range(len(reference_samples))) - matched_reference
)

missed_symbols = reference_symbols[missed_indices]

missed_df = pd.DataFrame({
    "sample": reference_samples[missed_indices],
    "time_s": reference_times[missed_indices],
    "symbol": missed_symbols
})

print("\nMISSED BEATS BY TYPE")
print("-" * 35)
print(missed_df["symbol"].value_counts())

missed_df.to_csv(
    RESULTS_DIR / "missed_beats.csv",
    index=False
)

# Calculate sensitivity separately for each beat type
print("\nSENSITIVITY BY BEAT TYPE")
print("-" * 35)

for symbol in sorted(set(reference_symbols)):
    indices = np.where(reference_symbols == symbol)[0]
    matched = sum(i in matched_reference for i in indices)

    sensitivity_type = matched / len(indices)

    print(
        f"{symbol}: {matched}/{len(indices)} "
        f"({sensitivity_type:.2%})"
    )

# Phase 3E: PVC energy analysis

rows = []
radius = int(0.15 * fs)

for i, (sample, symbol) in enumerate(
    zip(reference_samples, reference_symbols)
):
    if symbol != "V":
        continue

    start = max(0, int(sample) - radius)
    end = min(len(integrated), int(sample) + radius + 1)

    local_energy = float(np.max(integrated[start:end]))

    rows.append({
        "time_s": sample / fs,
        "detected": i in matched_reference,
        "local_energy": local_energy
    })

pvc_df = pd.DataFrame(rows)

print("\nPVC ENERGY ANALYSIS")
print("-" * 35)

print(
    pvc_df.groupby("detected")["local_energy"]
    .agg(["count", "median", "mean", "min", "max"])
)

print(f"\nGlobal threshold: {threshold:.2f}")

# Plot energy distributions
plt.figure(figsize=(10, 5))

plt.hist(
    pvc_df.loc[pvc_df["detected"], "local_energy"],
    bins=35,
    alpha=0.6,
    label="Detected PVCs"
)

plt.hist(
    pvc_df.loc[~pvc_df["detected"], "local_energy"],
    bins=20,
    alpha=0.7,
    label="Missed PVCs"
)

plt.axvline(
    threshold,
    color="red",
    linestyle="--",
    label="Global threshold"
)

plt.xlabel("Maximum integrated energy near PVC")
plt.ylabel("Number of PVCs")
plt.title("PVC Detection - Energy Distribution")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "pvc_energy_analysis.png",
    dpi=200
)

plt.show()

# 7. Results
print("\nFULL RECORD VALIDATION")
print("-" * 35)
print(f"Reference beats: {len(reference_samples)}")
print(f"Detected candidates: {len(peaks)}")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score: {f1:.4f}")

# 8. Save metrics
metrics = pd.DataFrame([{
    "record": RECORD,
    "reference_beats": len(reference_samples),
    "detected_candidates": len(peaks),
    "TP": tp,
    "FP": fp,
    "FN": fn,
    "sensitivity": sensitivity,
    "precision": precision,
    "f1_score": f1
}])

metrics.to_csv(
    RESULTS_DIR / "full_record_metrics.csv",
    index=False
)

# 9. Plot first 10 seconds
n = 10 * fs
t = np.arange(n) / fs

plt.figure(figsize=(14, 5))
plt.plot(t, integrated[:n], label="Integrated ECG")

mask = peaks < n

plt.scatter(
    peaks[mask] / fs,
    integrated[peaks[mask]],
    color="red",
    label="Detected QRS candidates"
)

plt.axhline(
    threshold,
    linestyle="--",
    color="blue",
    label="Threshold"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Integrated energy")
plt.title("Full Record Detector - First 10 Seconds")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "full_record_qrs_detection.png",
    dpi=200
)

plt.show()
