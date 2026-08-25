"""One-time script: filter CredData's meta/*.csv annotations down to Python files.

Run after scripts/fetch_creddata.py has populated
data/creddata_raw/CredData/{data,meta}. Writes
data/processed/creddata_python_labels.csv, the single flat file every
evaluation run reads from (see src/credhunter_x/dataset/creddata.py) — this
script does not need to be re-run unless CredData itself is re-fetched.

Usage:
    uv run python scripts/filter_python_creddata.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "creddata_raw" / "CredData"
META_DIR = RAW_ROOT / "meta"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "creddata_python_labels.csv"


def load_all_meta() -> pd.DataFrame:
    csv_files = sorted(META_DIR.glob("*.csv"))
    if not csv_files:
        sys.exit(
            f"No meta/*.csv files found under {META_DIR} — run scripts/fetch_creddata.py first."
        )
    # dtype=str: several columns (RepoName, FileID) are hex strings that can
    # coincidentally parse as valid numbers/scientific-notation (e.g. a repo
    # ID of "55031e17" reads as 5.5031e17) — pandas infers dtype per source
    # file, and a column that's a single repeated value (RepoName, within
    # one repo's CSV) is especially prone to this. Forcing str avoids any
    # numeric misparsing/precision loss across the board.
    frames = [pd.read_csv(f, dtype=str) for f in csv_files]
    return pd.concat(frames, ignore_index=True)


def filter_to_python(meta: pd.DataFrame) -> pd.DataFrame:
    python_rows = meta[meta["FilePath"].str.endswith(".py")].copy()

    exists_mask = python_rows["FilePath"].apply(lambda p: (RAW_ROOT / p).is_file())
    dropped = int((~exists_mask).sum())
    if dropped:
        print(
            f"Dropping {dropped} meta rows whose referenced file is missing on disk (stale entries)."
        )
    return python_rows[exists_mask]


def main() -> None:
    meta = load_all_meta()
    print(f"Loaded {len(meta)} total annotation rows from {META_DIR}")

    python_rows = filter_to_python(meta)
    print(f"{len(python_rows)} rows reference a .py file that exists on disk")

    true_count = int((python_rows["GroundTruth"] == "T").sum())
    print(f"  of which {true_count} are labelled True (real credential)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    python_rows.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
