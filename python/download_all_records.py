
from pathlib import Path
import wfdb

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Retrieve official MIT-BIH record identifiers
records = wfdb.get_record_list("mitdb")

print(f"Total MIT-BIH records: {len(records)}")

for i, record_id in enumerate(records, start=1):
    path = RAW_DIR / record_id

    required = [
        path.with_suffix(".hea"),
        path.with_suffix(".dat"),
        path.with_suffix(".atr")
    ]

    if all(p.exists() for p in required):
        print(f"[{i}/{len(records)}] {record_id}: already exists")
        continue

    print(f"[{i}/{len(records)}] Downloading {record_id}...")

    wfdb.dl_database(
        "mitdb",
        dl_dir=str(RAW_DIR),
        records=[record_id],
        annotators=["atr"]
    )

print("\nDownload completed.")
