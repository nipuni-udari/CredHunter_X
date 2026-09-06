from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

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


def agreement_vectors(
    outcomes: list[MatchOutcome], flagged: list[bool]
) -> tuple[list[bool], list[bool]]:
    """Paired correctness for a detector-vs-LLM McNemar, where "correct"
    means the prediction agrees with ground truth.

    The other definition -- correct = flagged AND really a secret -- makes
    the LLM's correct set a strict subset of the detector's, since the
    detector flags every candidate it generated. One McNemar cell is then
    structurally zero and the test can't favour the LLM however well it
    does: a correctly suppressed false positive counts as wrong for both.
    Both are reported; this is the pair that answers RQ1.

    LOST is dropped -- with no ground-truth row there's nothing to agree
    with, matching compute_metrics, which excludes it from precision."""
    if len(outcomes) != len(flagged):
        raise ValueError("outcomes and flagged must be the same length (paired candidates)")

    detector_agrees: list[bool] = []
    llm_agrees: list[bool] = []
    for outcome, is_flagged in zip(outcomes, flagged, strict=True):
        if outcome is MatchOutcome.LOST:
            continue
        is_secret = outcome is MatchOutcome.TRUE_POSITIVE
        detector_agrees.append(is_secret)  # the detector flagged all of them
        llm_agrees.append(is_flagged == is_secret)
    return detector_agrees, llm_agrees


@dataclass(frozen=True)
class ConfusionCounts:
    """Candidate-level confusion. Distinct from compute_metrics, whose
    recall denominator is every secret in the corpus, not just candidates."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        flagged = self.true_positive + self.false_positive
        return self.true_positive / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        real = self.true_positive + self.false_negative
        return self.true_positive / real if real else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def candidate_confusion(
    outcomes: Sequence[MatchOutcome], flagged: Sequence[bool]
) -> ConfusionCounts:
    """LOST is dropped -- with no ground-truth row there's nothing to be
    right or wrong about, the same exclusion compute_metrics applies."""
    if len(outcomes) != len(flagged):
        raise ValueError("outcomes and flagged must be the same length (paired candidates)")

    tp = fp = fn = tn = 0
    for outcome, is_flagged in zip(outcomes, flagged, strict=True):
        if outcome is MatchOutcome.LOST:
            continue
        is_secret = outcome is MatchOutcome.TRUE_POSITIVE
        if is_flagged and is_secret:
            tp += 1
        elif is_flagged:
            fp += 1
        elif is_secret:
            fn += 1
        else:
            tn += 1
    return ConfusionCounts(tp, fp, fn, tn)


def matthews_corrcoef(counts: ConfusionCounts) -> float:
    """MCC stays honest on unbalanced classes where F1 flatters. Returns
    0.0 when a row or column is empty -- undefined, conventionally
    reported as no correlation. A flag-everything detector scores 0.0."""
    tp, fp = counts.true_positive, counts.false_positive
    fn, tn = counts.false_negative, counts.true_negative
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0


def false_positive_rate(counts: ConfusionCounts) -> float:
    """Share of non-secret candidates that still got flagged."""
    negatives = counts.false_positive + counts.true_negative
    return counts.false_positive / negatives if negatives else 0.0


def bootstrap_ci(
    outcomes: Sequence[MatchOutcome],
    flagged: Sequence[bool],
    statistic: Callable[[ConfusionCounts], float],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap over candidates. Captures sampling variation --
    which candidates the scanners happened to produce -- and NOT run-to-run
    LLM variation, which needs repeated runs instead. Seeded, so a rerun
    reproduces the interval exactly."""
    if len(outcomes) != len(flagged):
        raise ValueError("outcomes and flagged must be the same length (paired candidates)")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    paired = [(o, f) for o, f in zip(outcomes, flagged, strict=True) if o is not MatchOutcome.LOST]
    if not paired:
        return 0.0, 0.0

    rng = random.Random(seed)
    n = len(paired)
    values = []
    for _ in range(resamples):
        drawn = [paired[rng.randrange(n)] for _ in range(n)]
        values.append(statistic(candidate_confusion([o for o, _ in drawn], [f for _, f in drawn])))
    values.sort()

    tail = (1.0 - confidence) / 2.0
    low = values[min(resamples - 1, int(tail * resamples))]
    high = values[min(resamples - 1, int((1.0 - tail) * resamples))]
    return low, high


def stratify_by_rule(
    rule_ids: Sequence[str], outcomes: Sequence[MatchOutcome], flagged: Sequence[bool]
) -> dict[str, ConfusionCounts]:
    """Per-rule confusion, so "where does the LLM actually help" is
    answerable rather than asserted."""
    if not len(rule_ids) == len(outcomes) == len(flagged):
        raise ValueError("rule_ids, outcomes and flagged must be the same length")

    grouped: dict[str, list[tuple[MatchOutcome, bool]]] = {}
    for rule_id, outcome, is_flagged in zip(rule_ids, outcomes, flagged, strict=True):
        grouped.setdefault(rule_id, []).append((outcome, is_flagged))
    return {
        rule_id: candidate_confusion([o for o, _ in rows], [f for _, f in rows])
        for rule_id, rows in grouped.items()
    }


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
