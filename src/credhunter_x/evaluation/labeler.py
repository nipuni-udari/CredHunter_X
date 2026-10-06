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


def _covers_same_characters(candidate: Candidate, row: GroundTruthRow) -> bool:
    """Do the candidate and the ground-truth row cover overlapping characters
    of the line? -1 means the position is unknown on either side, which never
    counts as a match."""
    if min(candidate.value_start, candidate.value_end, row.value_start, row.value_end) < 0:
        return False
    return candidate.value_start < row.value_end and row.value_start < candidate.value_end


def _row_deciding_the_outcome(candidate: Candidate, rows: list[GroundTruthRow]) -> GroundTruthRow:
    """Picks which ground-truth row at a file and line this candidate answers for.

    One line can hold two labelled items (an id and a key, say) with different
    labels, so character offsets break the tie. With no usable offsets the
    first row is used."""
    if len(rows) == 1:
        return rows[0]
    overlapping = [row for row in rows if _covers_same_characters(candidate, row)]
    return overlapping[0] if overlapping else rows[0]


def match_candidate(
    candidate: Candidate, gt_index: dict[MetaKey, list[GroundTruthRow]]
) -> MatchOutcome:
    """Matches on file and line only, as CredData's own GitLeaks integration
    does (benchmark/scanner/gitleaks.py). When several rows share a line,
    character offsets choose between them; otherwise the first row in the CSV
    is used."""
    key = MetaKey(candidate.file_path, candidate.line_start, candidate.line_end)
    rows = gt_index.get(key)
    if not rows:
        return MatchOutcome.LOST

    row = _row_deciding_the_outcome(candidate, rows)
    return MatchOutcome.TRUE_POSITIVE if row.ground_truth else MatchOutcome.FALSE_POSITIVE
