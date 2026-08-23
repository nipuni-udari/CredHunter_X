from __future__ import annotations

from collections import Counter

from scipy.stats import chi2, norm

from credhunter_x.evaluation.labeler import MatchOutcome
from credhunter_x.models.evaluation import MetricReport


def compute_metrics(
    outcomes: list[MatchOutcome], *, total_true_count: int, arm: str, treatment: str
) -> MetricReport:
    """precision/recall/F1 from matched outcomes, matching CredData's own
    scoring: LOST is excluded entirely, not counted as a false positive.
    total_true_count is every GroundTruth=='T' row in the dataset, not
    just what became a candidate -- so recall reflects secrets the
    scanner never even flagged."""
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
    """Continuity-corrected McNemar's test on paired per-candidate
    correctness (same order, same length). scipy has no built-in
    mcnemar(), so the statistic's hand-computed and only the p-value
    lookup uses scipy.stats.chi2. Returns (0.0, 1.0) if there's no
    discordant pair -- undefined, not "no difference"."""
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


def cohens_kappa(rater_a: list[object], rater_b: list[object]) -> float:
    """Inter-rater agreement, corrected for chance. po = observed
    agreement, pe = agreement expected by chance from each rater's own
    category distribution. Returns 1.0 for the degenerate 0/0 case
    (everyone agrees on one category)."""
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must be the same length (paired items)")
    if not rater_a:
        raise ValueError("cannot compute kappa over an empty rating set")

    n = len(rater_a)
    po = sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a == b) / n

    categories = set(rater_a) | set(rater_b)
    pe = sum((rater_a.count(c) / n) * (rater_b.count(c) / n) for c in categories)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings: list[list[object]]) -> float:
    """Agreement among a fixed number of raters per item (unlike
    cohens_kappa's two named raters) -- used for the model's own
    stability across repeated calls. ratings[i] is every rater's category
    for item i; every item needs the same rater count (>= 2). Returns 1.0
    for the degenerate 0/0 case."""
    if not ratings:
        raise ValueError("cannot compute kappa over an empty rating set")

    rater_count = len(ratings[0])
    if rater_count < 2:
        raise ValueError("fleiss_kappa needs at least 2 raters per item")
    if any(len(item_ratings) != rater_count for item_ratings in ratings):
        raise ValueError("every item must have the same number of raters")

    n_items = len(ratings)
    categories = {c for item_ratings in ratings for c in item_ratings}
    category_counts = [Counter(item_ratings) for item_ratings in ratings]

    p_j = {
        c: sum(counts[c] for counts in category_counts) / (n_items * rater_count)
        for c in categories
    }
    p_e_bar = sum(p**2 for p in p_j.values())

    p_i = [
        (sum(counts[c] ** 2 for c in categories) - rater_count) / (rater_count * (rater_count - 1))
        for counts in category_counts
    ]
    p_bar = sum(p_i) / n_items

    if p_e_bar == 1.0:
        return 1.0
    return (p_bar - p_e_bar) / (1 - p_e_bar)


def wilson_score_interval(
    successes: int, n: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Binomial-proportion CI for a pass rate. Wilson, not a naive Wald
    interval -- stays within [0,1] and holds up better at small n. z comes
    from scipy.stats.norm.ppf rather than hardcoding 1.96."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")

    p_hat = successes / n
    z = norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * ((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5) / denominator
    return float(center - margin), float(center + margin)
