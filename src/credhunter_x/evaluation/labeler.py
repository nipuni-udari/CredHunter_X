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
    """Do the candidate and the ground-truth row point at overlapping
    characters of their shared line? Both sides use -1 for "position
    unknown" (the scanner reported a value we couldn't locate verbatim, or
    CredData recorded no offsets), and an unknown position can't be
    evidence of anything -- so it never claims a match."""
    if min(candidate.value_start, candidate.value_end, row.value_start, row.value_end) < 0:
        return False
    return candidate.value_start < row.value_end and row.value_start < candidate.value_end


def _row_deciding_the_outcome(candidate: Candidate, rows: list[GroundTruthRow]) -> GroundTruthRow:
    """Which of several ground-truth rows at one file+line this candidate
    is actually answering for.

    A single line commonly carries more than one labelled item -- an id and
    a key, a username and a password -- and they can disagree, one labelled
    a real secret and the other not. file+line alone cannot tell them
    apart, so taking the first row scores a candidate against whichever
    item happened to be listed first in the CSV: the same finding scores
    differently if the ground-truth file is reordered. Character offsets
    are the only thing that distinguishes them, so they break the tie.

    Position is a tie-break, never the primary key: the file+line lookup
    above stays byte-for-byte compatible with CredData's own GitLeaks
    integration, and anything with no usable offsets falls back to first-
    row-wins exactly as before."""
    if len(rows) == 1:
        return rows[0]
    overlapping = [row for row in rows if _covers_same_characters(candidate, row)]
    return overlapping[0] if overlapping else rows[0]


def match_candidate(
    candidate: Candidate, gt_index: dict[MetaKey, list[GroundTruthRow]]
) -> MatchOutcome:
    """Matches on file+line only, matching CredData's own GitLeaks
    integration (verified against benchmark/scanner/gitleaks.py -- an
    earlier rule/value-aware version got 0% matches against real data).
    Where several rows share that file+line, character offsets pick which
    one the candidate is answering for; ties and missing offsets go to the
    first row loaded from the CSV, same as CredData's own scan order."""
    key = MetaKey(candidate.file_path, candidate.line_start, candidate.line_end)
    rows = gt_index.get(key)
    if not rows:
        return MatchOutcome.LOST

    row = _row_deciding_the_outcome(candidate, rows)
    return MatchOutcome.TRUE_POSITIVE if row.ground_truth else MatchOutcome.FALSE_POSITIVE
