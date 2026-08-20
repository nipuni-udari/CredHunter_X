from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_LABELS_PATH = Path("data/processed/creddata_python_labels.csv")


@dataclass(frozen=True)
class GroundTruthRow:
    """One row of CredData's own ground truth, already filtered to Python
    files by scripts/filter_python_creddata.py. value_start/value_end use
    -1 as CredData's own "no value" sentinel (see evaluation/labeler.py's
    matching algorithm, ported from CredData's real scanner source) rather
    than None — the raw dataset leaves these blank for non-True rows, but
    the matching algorithm expects the same sentinel CredData itself uses
    internally."""

    id: str
    file_id: str
    repo_id: str
    file_path: str
    line_start: int
    line_end: int
    ground_truth: bool
    value_start: int
    value_end: int
    category: str


def load_creddata_labels(path: Path = _DEFAULT_LABELS_PATH) -> list[GroundTruthRow]:
    """Pure loader: reads the already-filtered CredData Python labels CSV
    into ground-truth records. No network calls, no CredData-specific
    parsing beyond this one flat file."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/fetch_creddata.py and "
            "scripts/filter_python_creddata.py first"
        )
    rows: list[GroundTruthRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        for record in csv.DictReader(f):
            rows.append(
                GroundTruthRow(
                    id=record["Id"],
                    file_id=record["FileID"],
                    repo_id=record["RepoName"],
                    file_path=record["FilePath"],
                    line_start=int(record["LineStart"]),
                    line_end=int(record["LineEnd"]),
                    ground_truth=record["GroundTruth"] == "T",
                    value_start=int(record["ValueStart"]) if record["ValueStart"] else -1,
                    value_end=int(record["ValueEnd"]) if record["ValueEnd"] else -1,
                    category=record["Category"],
                )
            )
    return rows
