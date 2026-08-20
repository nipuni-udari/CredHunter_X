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
    """Matches on file+line only — verified directly against CredData's
    own GitLeaks integration (benchmark/scanner/gitleaks.py), not assumed.
    check_line_from_meta's general signature supports rule/value-position
    matching too, but CredData's real GitLeaks caller passes neither
    (`check_line_from_meta(file, line, line)`, no rule, no value_start/
    value_end) — so their published 0.526/0.244/0.334 baseline is pure
    file+line matching. An earlier version of this function ported the
    general (rule/value-aware) algorithm and got 0% matches against real
    data as a result: gitleaks' own rule_id never appears as a literal
    element of CredData's category strings, because CredData never checks
    it for GitLeaks in the first place.

    Since every candidate in every arm of this project originates from
    GitLeaks (the LLM only ever re-judges GitLeaks' own candidates), this
    is the correct matching mode everywhere in this project, not just for
    a GitLeaks-only baseline run.

    When multiple ground-truth rows exist at the same file+line, the
    first one (in the order loaded from the CSV) wins — matching
    CredData's own iteration order, since their loop returns on the first
    matching row it finds."""
    key = MetaKey(candidate.file_path, candidate.line_start, candidate.line_end)
    rows = gt_index.get(key)
    if not rows:
        return MatchOutcome.LOST

    row = rows[0]
    return MatchOutcome.TRUE_POSITIVE if row.ground_truth else MatchOutcome.FALSE_POSITIVE
