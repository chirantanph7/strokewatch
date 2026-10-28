"""
Rebuilds M1's data handoff from the raw Kaggle CSV, using Jyot's own code
(m1_jyot/preprocess.py and m1_jyot/make_splits.py), so a fresh clone can
reproduce data/stroke_typed.parquet and data/splits_REAL.json exactly.

    python make_data.py path/to/healthcare-dataset-stroke-data.csv

The raw CSV is not in the repository (see m1_jyot/LICENCE_NOTE.md). The
checksum check below stops the run if the download differs from the file the
project was built on.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "m1_jyot"))
from make_splits import make_splits  # noqa: E402  (M1's deliverable)
from preprocess import file_checksum, load_and_type  # noqa: E402  (M1's deliverable)

EXPECTED_SHA256 = "644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420"


def main(csv_path: str) -> None:
    digest = file_checksum(csv_path)
    if digest != EXPECTED_SHA256:
        sys.exit(f"Checksum mismatch: got {digest}, expected {EXPECTED_SHA256}. "
                 "This is not the dataset version StrokeWatch was built on.")
    df = load_and_type(csv_path)
    (ROOT / "data").mkdir(exist_ok=True)
    df.to_parquet(ROOT / "data" / "stroke_typed.parquet")
    splits = make_splits(df)
    with open(ROOT / "data" / "splits_REAL.json", "w") as f:
        json.dump(splits, f, indent=2)
    print(f"{len(df)} rows, {int(df['stroke'].sum())} strokes; "
          f"holdout {len(splits['holdout_ids'])}, dev {len(splits['dev_ids'])} in {splits['n_folds']} folds")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python make_data.py path/to/healthcare-dataset-stroke-data.csv")
    main(sys.argv[1])
