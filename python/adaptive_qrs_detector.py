
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt, find_peaks

ROOT = Path(__file__).resolve().parents[1]
RECORD_ID = "208"

record = wfdb.rdrecord(
    str(ROOT / "data" / "raw" / RECORD_ID)
)

fs = record.fs
ecg = record.p_signal[:, 0]

# 1. Preprocessing
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

# 2. Candidate peaks
candidate_peaks, _ = find_peaks(
    integrated,
    distance=int(0.20 * fs)
)

# 3. Initialize adaptive levels using first 2 seconds
initial_samples = int(2 * fs)
initial_values = integrated[:initial_samples]

signal_level = float(np.percentile(initial_values, 95))
noise_level = float(np.percentile(initial_values, 50))

# 4. Adaptive detection
detected_peaks = []
threshold_history = []

for peak in candidate_peaks:
    amplitude = integrated[peak]

    threshold = noise_level + 0.25 * (
        signal_level - noise_level
    )

    threshold_history.append((peak / fs, threshold))

    if amplitude >= threshold:
        detected_peaks.append(peak)

        signal_level = (
            0.125 * amplitude
            + 0.875 * signal_level
        )
    else:
        noise_level = (
            0.125 * amplitude
            + 0.875 * noise_level
        )

detected_peaks = np.array(
    detected_peaks, dtype=int
)

# 4B. Search-back for potentially missed QRS peaks
# Uses only ECG-derived information, not annotations.

primary_peaks = detected_peaks.copy()

# Estimate a typical RR interval from primary detections.
rr_intervals = np.diff(primary_peaks) / fs

if len(rr_intervals) > 0:
    median_rr = np.median(rr_intervals)
    long_rr_limit = 1.66 * median_rr
else:
    long_rr_limit = np.inf

# Lower threshold for search-back candidates
threshold_values = np.array([
    value for _, value in threshold_history
])

rejected_mask = ~np.isin(
    candidate_peaks,
    primary_peaks
)

rejected_peaks = candidate_peaks[rejected_mask]
rejected_thresholds = threshold_values[rejected_mask]

recovered_peaks = []

for left_peak, right_peak in zip(
    primary_peaks[:-1],
    primary_peaks[1:]
):
    rr_gap = (right_peak - left_peak) / fs

    if rr_gap <= long_rr_limit:
        continue

    # Candidate peaks inside the unusually long RR gap
    mask = (
        (rejected_peaks > left_peak + int(0.20 * fs))
        & (rejected_peaks < right_peak - int(0.20 * fs))
    )

    indices = np.where(mask)[0]

    if len(indices) == 0:
        continue

    # Use half of the adaptive threshold recorded
    # at each candidate's original detection time.
    eligible = [
        idx for idx in indices
        if integrated[rejected_peaks[idx]]
        >= 0.5 * rejected_thresholds[idx]
    ]

    if not eligible:
        continue

    best_idx = max(
        eligible,
        key=lambda idx: integrated[rejected_peaks[idx]]
    )

    recovered_peaks.append(
        rejected_peaks[best_idx]
    )

recovered_peaks = np.array(
    recovered_peaks,
    dtype=int
)

detected_peaks = np.sort(
    np.concatenate([
        primary_peaks,
        recovered_peaks
    ])
)

print("\nSEARCH-BACK RESULTS")
print(f"Primary detections: {len(primary_peaks)}")
print(f"Recovered candidates: {len(recovered_peaks)}")
print(f"Total detections: {len(detected_peaks)}")


print(f"Record: {RECORD_ID}")
print(f"Reference-independent detections: {len(detected_peaks)}")
print(f"Initial signal level: {np.percentile(initial_values, 95):.2f}")
print(f"Initial noise level: {np.percentile(initial_values, 50):.2f}")

# 5. Reference annotations
annotation = wfdb.rdann(
    str(ROOT / "data" / "raw" / RECORD_ID),
    "atr"
)

BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S",
    "V", "r", "F", "e", "j", "n", "E",
    "/", "f", "Q", "?"
}

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
detected_times = detected_peaks / fs

# 6. One-to-one matching
tolerance = 0.15
candidates = []

for i, d in enumerate(detected_times):
    left = np.searchsorted(
        reference_times,
        d - tolerance,
        side="left"
    )

    right = np.searchsorted(
        reference_times,
        d + tolerance,
        side="right"
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

# 7. Overall metrics
tp = len(matched_detection)
fp = len(detected_peaks) - tp
fn = len(reference_samples) - tp

sensitivity = tp / (tp + fn) if tp + fn else 0
precision = tp / (tp + fp) if tp + fp else 0

f1 = (
    2 * sensitivity * precision
    / (sensitivity + precision)
    if sensitivity + precision else 0
)

# 8. PVC sensitivity
pvc_indices = np.where(
    reference_symbols == "V"
)[0]

pvc_total = len(pvc_indices)
pvc_detected = sum(
    j in matched_reference
    for j in pvc_indices
)

pvc_sensitivity = (
    pvc_detected / pvc_total
    if pvc_total else np.nan
)

print("\nADAPTIVE QRS VALIDATION")
print("-" * 50)
print(f"Reference beats: {len(reference_samples)}")
print(f"Detected peaks: {len(detected_peaks)}")
print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"PVC detected: {pvc_detected}/{pvc_total}")
print(f"PVC sensitivity: {pvc_sensitivity:.4f}")

# 9. False Negative Diagnostics

from collections import Counter
import pandas as pd

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

missed_indices = sorted(
    set(range(len(reference_samples))) - matched_reference
)

# Reconstruct the threshold applied to each candidate.
candidate_times = candidate_peaks / fs
threshold_values = np.array(
    [value for _, value in threshold_history]
)

accepted_mask = np.isin(candidate_peaks, detected_peaks)

diagnostics = []

for j in missed_indices:
    ref_sample = reference_samples[j]
    ref_time = ref_sample / fs
    symbol = reference_symbols[j]

    # Search for candidate peaks within ±150 ms
    nearby = np.where(
        np.abs(candidate_peaks - ref_sample) / fs <= 0.15
    )[0]

    if len(nearby) == 0:
        reason = "no_candidate_within_tolerance"
        candidate_time = np.nan
        candidate_amplitude = np.nan
        applied_threshold = np.nan

    else:
        nearest = nearby[
            np.argmin(
                np.abs(candidate_peaks[nearby] - ref_sample)
            )
        ]

        candidate_time = candidate_times[nearest]
        candidate_amplitude = integrated[
            candidate_peaks[nearest]
        ]
        applied_threshold = threshold_values[nearest]

        if accepted_mask[nearby].any():
            reason = "accepted_but_unmatched"
        else:
            reason = "below_adaptive_threshold"

    diagnostics.append({
        "reference_time": ref_time,
        "beat_type": symbol,
        "reason": reason,
        "candidate_time": candidate_time,
        "candidate_amplitude": candidate_amplitude,
        "applied_threshold": applied_threshold
    })

diagnostics_df = pd.DataFrame(diagnostics)

print("\nFALSE NEGATIVE DIAGNOSTICS")
print("-" * 60)

print("\nMissed beats by reason:")
print(diagnostics_df["reason"].value_counts())

print("\nMissed beats by type:")
print(diagnostics_df["beat_type"].value_counts())

print("\nPVC missed beats by reason:")
print(
    diagnostics_df.loc[
        diagnostics_df["beat_type"] == "V",
        "reason"
    ].value_counts()
)

output = RESULTS_DIR / "adaptive_fn_diagnostics.csv"
diagnostics_df.to_csv(output, index=False)

print(f"\nDiagnostics saved to: {output}")
