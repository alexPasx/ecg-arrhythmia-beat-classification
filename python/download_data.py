
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

for directory in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RECORD = "208"

# Download ECG and annotations from PhysioNet
wfdb.dl_database(
    "mitdb",
    dl_dir=str(RAW_DIR),
    records=[RECORD],
    annotators=["atr"],
    keep_subdirs=False
)

# Read locally downloaded data
record = wfdb.rdrecord(str(RAW_DIR / RECORD))
annotation = wfdb.rdann(str(RAW_DIR / RECORD), "atr")

fs = record.fs
ecg = record.p_signal[:, 0]
time = np.arange(len(ecg)) / fs

print(f"Record: {RECORD}")
print(f"Sampling frequency: {fs} Hz")
print(f"Channels: {record.sig_name}")
print(f"Duration: {len(ecg) / fs / 60:.2f} minutes")
print(f"Annotations: {len(annotation.sample)}")

# Save first five minutes for MATLAB
n_samples = int(5 * 60 * fs)

ecg_df = pd.DataFrame({
    "time_s": time[:n_samples],
    "ecg_mV": ecg[:n_samples]
})

ecg_df.to_csv(
    PROCESSED_DIR / "record_208_5min.csv",
    index=False
)

# Export annotations within the same window
mask = annotation.sample < n_samples

annotations_df = pd.DataFrame({
    "sample": annotation.sample[mask],
    "time_s": annotation.sample[mask] / fs,
    "symbol": np.asarray(annotation.symbol)[mask]
})

annotations_df.to_csv(
    PROCESSED_DIR / "record_208_annotations.csv",
    index=False
)

# Plot first ten seconds
seconds = 10
n_plot = int(seconds * fs)

plt.figure(figsize=(14, 4))
plt.plot(time[:n_plot], ecg[:n_plot], linewidth=0.8)

plt.title("MIT-BIH Record 208 - First 10 Seconds")
plt.xlabel("Time (seconds)")
plt.ylabel("ECG Amplitude (mV)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "record_208_first_10s.png",
    dpi=200
)

plt.show()
