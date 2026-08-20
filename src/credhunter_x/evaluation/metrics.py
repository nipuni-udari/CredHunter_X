from __future__ import annotations

from collections import Counter

from scipy.stats import chi2

from credhunter_x.evaluation.labeler import MatchOutcome
from credhunter_x.models.evaluation import MetricReport


def compute_metrics(
    outcomes: list[MatchOutcome], *, total_true_count: int, arm: str, treatment: str
) -> MetricReport:
    """precision/recall/F1 from matched outcomes, following CredData's own
    scoring convention (verified against their real
    benchmark/common/result.py and benchmark/scanner/scanner.py): LOST is
    excluded entirely from both precision's and recall's numerator and
    denominator — it's a diagnostic count, not a false positive.

    total_true_count is the count of GroundTruth=='T' rows across the
    WHOLE labelled dataset, independent of what became a candidate at all
    — this is what makes recall capture secrets the scanner never
    generated a candidate for in the first place, not just wrong
    downstream judgments on candidates it did generate."""
    counts = Counter(outcomes)
    true_positive = counts[MatchOutcome.TRUE_POSITIVE]
    false_positive = counts[MatchOutcome.FALSE_POSITIVE]
    lost = counts[MatchOutcome.LOST]

    total_flagged = true_positive + false_positive
    precision = true_positive / total_flagged if total_flagged else 0.0
    recall = true_positive / total_true_count if total_true_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    return MetricReport(
        arm=arm,
        treatment=treatment,
        n_candidates=len(outcomes),
        precision=precision,
        recall=recall,
        f1=f1,
        lost_count=lost,
    )


def mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> tuple[float, float]:
    """Continuity-corrected McNemar's test comparing two arms' per-candidate
    correctness on the SAME paired candidates (same order, same length —
    e.g. gitleaks-only vs Arm A over identical candidates). scipy has no
    built-in mcnemar() function — the statistic is computed by hand and
    cross-checked against scipy.stats.chi2 for the p-value, per the plan.

    Returns (0.0, 1.0) when there's no discordant pair — the test is
    undefined in that case (both arms agree on every candidate), not
    evidence of "no difference" in the usual statistical sense."""
    if len(correct_a) != len(correct_b):
        raise ValueError("correct_a and correct_b must be the same length (paired candidates)")

    only_a_correct = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if a and not b)
    only_b_correct = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if not a and b)
    discordant = only_a_correct + only_b_correct

    if discordant == 0:
        return 0.0, 1.0

    statistic = (abs(only_a_correct - only_b_correct) - 1) ** 2 / discordant
    p_value = float(chi2.sf(statistic, df=1))
    return statistic, p_value
