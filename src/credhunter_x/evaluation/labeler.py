from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from credhunter_x.dataset.creddata import GroundTruthRow
from credhunter_x.models.candidate import Candidate


class MatchOutcome(StrEnum):
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    LOST = "lost"


@dataclass(frozen=True)
class MetaKey:
    file_path: str
    line_start: int
    line_end: int


def index_ground_truth(rows: list[GroundTruthRow]) -> dict[MetaKey, list[GroundTruthRow]]:
    index: dict[MetaKey, list[GroundTruthRow]] = defaultdict(list)
    for row in rows:
        index[MetaKey(row.file_path, row.line_start, row.line_end)].append(row)
    return index


def match_candidate(
    candidate: Candidate, gt_index: dict[MetaKey, list[GroundTruthRow]]
) -> MatchOutcome:
    """Matches on file+line only, matching CredData's own GitLeaks
    integration (verified against benchmark/scanner/gitleaks.py -- an
    earlier rule/value-aware version got 0% matches against real data).
    Ties go to the first row loaded from the CSV, same as CredData's own
    scan order."""
    key = MetaKey(candidate.file_path, candidate.line_start, candidate.line_end)
    rows = gt_index.get(key)
    if not rows:
        return MatchOutcome.LOST

    row = rows[0]
    return MatchOutcome.TRUE_POSITIVE if row.ground_truth else MatchOutcome.FALSE_POSITIVE
