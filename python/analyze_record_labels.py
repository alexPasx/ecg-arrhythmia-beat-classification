
from pathlib import Path
import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

records = wfdb.get_record_list("mitdb")
rows = []

for record_id in records:
    ann = wfdb.rdann(
        str(RAW_DIR / record_id), "atr"
    )

    counts = pd.Series(ann.symbol).value_counts()

    row = {
        "record_id": record_id,
        "N": int(counts.get("N", 0)),
        "V": int(counts.get("V", 0)),
        "F": int(counts.get("F", 0)),
    }

    row["total_NVF"] = (
        row["N"] + row["V"] + row["F"]
    )

    for label in ["N", "V", "F"]:
        row[f"{label}_pct"] = (
            100 * row[label] / row["total_NVF"]
            if row["total_NVF"] else 0
        )

    rows.append(row)

df = pd.DataFrame(rows)

print("\nLABEL DISTRIBUTION BY RECORD")
print(df.to_string(index=False))

print("\nTOTAL LABEL DISTRIBUTION")
print(df[["N", "V", "F"]].sum())

print("\nRECORDS CONTAINING FUSION BEATS")
print(
    df[df["F"] > 0]
    .sort_values("F", ascending=False)
    [["record_id", "N", "V", "F"]]
    .to_string(index=False)
)

df.to_csv(
    RESULTS_DIR / "record_label_distribution.csv",
    index=False
)
