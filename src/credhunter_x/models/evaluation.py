from __future__ import annotations

from dataclasses import dataclass, field

from credhunter_x.models.candidate import Candidate
from credhunter_x.models.classification import ClassificationResult


@dataclass(frozen=True)
class EvalRow:
    """One candidate joined against ground truth, plus results per arm.
    A dict, not fixed fields, so partial runs don't need a schema change."""

    candidate: Candidate
    ground_truth_is_secret: bool
    results: dict[str, ClassificationResult] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricReport:
    arm: str
    treatment: str
    n_candidates: int
    precision: float
    recall: float
    f1: float
    lost_count: int = 0  # no matching ground-truth row; excluded from precision/recall/F1
    mcnemar_stat: float | None = None
    mcnemar_p_value: float | None = None
