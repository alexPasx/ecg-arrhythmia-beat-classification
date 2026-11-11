
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfiltfilt, find_peaks

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RECORDS = ["100", "106", "119", "200"]

BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S",
    "V", "r", "F", "e", "j", "n", "E",
    "/", "f", "Q", "?"
}


def detect_qrs(ecg, fs):
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

    candidate_peaks, _ = find_peaks(
        integrated,
        distance=int(0.20 * fs)
    )

    initial_values = integrated[:int(2 * fs)]
    signal_level = float(
        np.percentile(initial_values, 95)
    )
    noise_level = float(
        np.percentile(initial_values, 50)
    )

    primary_peaks = []
    threshold_values = []

    for peak in candidate_peaks:
        amplitude = integrated[peak]

        threshold = noise_level + 0.25 * (
            signal_level - noise_level
        )
        threshold_values.append(threshold)

        if amplitude >= threshold:
            primary_peaks.append(peak)
            signal_level = (
                0.125 * amplitude
                + 0.875 * signal_level
            )
        else:
            noise_level = (
                0.125 * amplitude
                + 0.875 * noise_level
            )

    primary_peaks = np.array(
        primary_peaks, dtype=int
    )

    # Search-back
    rr_intervals = np.diff(primary_peaks) / fs

    if len(rr_intervals) > 0:
        long_rr_limit = 1.66 * np.median(rr_intervals)
    else:
        long_rr_limit = np.inf

    threshold_values = np.array(threshold_values)

    rejected_mask = ~np.isin(
        candidate_peaks, primary_peaks
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

        mask = (
            (rejected_peaks > left_peak + int(0.20 * fs))
            & (rejected_peaks < right_peak - int(0.20 * fs))
        )

        indices = np.where(mask)[0]

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

    detected_peaks = np.sort(
        np.concatenate([
            primary_peaks,
            np.array(recovered_peaks, dtype=int)
        ])
    )

    return detected_peaks, len(primary_peaks), len(recovered_peaks)


def evaluate(peaks, annotation, fs):
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

    pvc_indices = np.where(
        reference_symbols == "V"
    )[0]

    pvc_total = len(pvc_indices)
    pvc_detected = sum(
        j in matched_reference
        for j in pvc_indices
    )

    return {
        "reference_beats": len(reference_samples),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "sensitivity": sensitivity,
        "precision": precision,
        "F1": f1,
        "PVC_total": pvc_total,
        "PVC_detected": pvc_detected,
        "PVC_sensitivity": (
            pvc_detected / pvc_total
            if pvc_total else np.nan
        )
    }


results = []

for record_id in RECORDS:
    print(f"\nProcessing record {record_id}...")

    path = RAW_DIR / record_id

    required = [
        path.with_suffix(".hea"),
        path.with_suffix(".dat"),
        path.with_suffix(".atr")
    ]

    if not all(p.exists() for p in required):
        wfdb.dl_database(
            "mitdb",
            dl_dir=str(RAW_DIR),
            records=[record_id],
            annotators=["atr"]
        )

    record = wfdb.rdrecord(str(path))
    annotation = wfdb.rdann(str(path), "atr")

    peaks, primary_count, recovered_count = detect_qrs(
        record.p_signal[:, 0],
        record.fs
    )

    metrics = evaluate(
        peaks,
        annotation,
        record.fs
    )

    results.append({
        "record": record_id,
        "primary": primary_count,
        "recovered": recovered_count,
        **metrics
    })


df = pd.DataFrame(results)

print("\nADAPTIVE MULTI-RECORD VALIDATION")
print("=" * 100)
print(df.round(4).to_string(index=False))

df.to_csv(
    RESULTS_DIR / "adaptive_multi_record_validation.csv",
    index=False
)
