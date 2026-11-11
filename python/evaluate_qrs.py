
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

detected = pd.read_csv(DATA / "record_208_qrs_candidates.csv")
annotations = pd.read_csv(DATA / "record_208_annotations.csv")

# MIT-BIH beat annotation symbols.
# Exclude rhythm markers and other non-beat annotations.
BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S",
    "V", "r", "F", "e", "j", "n", "E",
    "/", "f", "Q", "?"
}

reference = annotations[
    annotations["symbol"].isin(BEAT_SYMBOLS)
].copy()

# Initial evaluation window: first 10 seconds
reference = reference[reference["time_s"] < 10]
detected = detected[detected["time_s"] < 10]

ref_times = np.sort(reference["time_s"].to_numpy())
det_times = np.sort(detected["time_s"].to_numpy())

# One-to-one matching with 150 ms tolerance
# This is a preliminary evaluation of integrated-signal
# candidate timing, not yet refined R-peak positions.
tolerance = 0.15

pairs = []
for i, d in enumerate(det_times):
    for j, r in enumerate(ref_times):
        if abs(d - r) <= tolerance:
            pairs.append((abs(d - r), i, j))

pairs.sort()

matched_det = set()
matched_ref = set()

for error, i, j in pairs:
    if i not in matched_det and j not in matched_ref:
        matched_det.add(i)
        matched_ref.add(j)

tp = len(matched_det)
fp = len(det_times) - tp
fn = len(ref_times) - tp

sensitivity = tp / (tp + fn) if tp + fn else 0
precision = tp / (tp + fp) if tp + fp else 0
f1 = (
    2 * precision * sensitivity / (precision + sensitivity)
    if precision + sensitivity else 0
)

print(f"Reference beats: {len(ref_times)}")
print(f"Detected candidates: {len(det_times)}")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Precision: {precision:.3f}")
print(f"F1-score: {f1:.3f}")
