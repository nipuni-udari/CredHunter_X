from __future__ import annotations

import math

import pytest

from credhunter_x.evaluation.labeler import MatchOutcome
from credhunter_x.evaluation.metrics import (
    ConfusionCounts,
    agreement_vectors,
    bootstrap_ci,
    candidate_confusion,
    cohens_kappa,
    compute_metrics,
    false_positive_rate,
    fleiss_kappa,
    matthews_corrcoef,
    mcnemar_test,
    stratify_by_rule,
    wilson_score_interval,
)


def test_compute_metrics_from_hand_computed_confusion_counts():
    outcomes = [MatchOutcome.TRUE_POSITIVE] * 6 + [MatchOutcome.FALSE_POSITIVE] * 4
    report = compute_metrics(outcomes, total_true_count=10, arm="single", treatment="masked")

    assert report.precision == 0.6  # 6 / (6 + 4)
    assert report.recall == 0.6  # 6 / 10
    assert math.isclose(report.f1, 0.6)
    assert report.n_candidates == 10
    assert report.lost_count == 0


def test_lost_outcomes_are_excluded_from_precision_recall_but_still_counted():
    outcomes = (
        [MatchOutcome.TRUE_POSITIVE] * 4
        + [MatchOutcome.FALSE_POSITIVE] * 1
        + [MatchOutcome.LOST] * 5
    )
    report = compute_metrics(outcomes, total_true_count=4, arm="gitleaks_only", treatment="raw")

    assert report.precision == 0.8  # 4 / (4 + 1) — LOST not in the denominator
    assert report.recall == 1.0  # 4 / 4
    assert report.n_candidates == 10
    assert report.lost_count == 5


def test_recall_reflects_secrets_never_generated_as_candidates_at_all():
    # only 3 of 10 real secrets in the dataset even became a candidate
    outcomes = [MatchOutcome.TRUE_POSITIVE] * 3
    report = compute_metrics(outcomes, total_true_count=10, arm="gitleaks_only", treatment="raw")

    assert report.recall == 0.3


def test_zero_candidates_gives_zero_metrics_not_a_crash():
    report = compute_metrics([], total_true_count=10, arm="single", treatment="masked")

    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1 == 0.0


def test_mcnemar_statistic_matches_a_known_worked_example():
    # 10 pairs where only A is correct, 2 where only B is correct, 10 concordant
    correct_a = [True] * 10 + [False] * 2 + [True] * 5 + [False] * 5
    correct_b = [False] * 10 + [True] * 2 + [True] * 5 + [False] * 5

    stat, p_value = mcnemar_test(correct_a, correct_b)

    # (|10 - 2| - 1)^2 / (10 + 2) = 49 / 12
    assert math.isclose(stat, 49 / 12)
    assert 0.0 <= p_value <= 1.0


def test_mcnemar_returns_no_significance_when_no_discordant_pairs():
    correct_a = [True, False, True, False]
    correct_b = [True, False, True, False]

    stat, p_value = mcnemar_test(correct_a, correct_b)

    assert stat == 0.0
    assert p_value == 1.0


def test_mcnemar_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        mcnemar_test([True, False], [True])


def test_cohens_kappa_matches_a_known_worked_example():
    rater_a = ["yes", "yes", "no", "no", "yes"]
    rater_b = ["yes", "no", "no", "no", "yes"]

    kappa = cohens_kappa(rater_a, rater_b)

    # po = 4/5 = 0.8; pe = (3/5)(2/5) + (2/5)(3/5) = 0.24 + 0.24 = 0.48
    # kappa = (0.8 - 0.48) / (1 - 0.48) = 0.32 / 0.52 = 8/13
    assert math.isclose(kappa, 8 / 13)


def test_cohens_kappa_is_one_for_perfect_agreement():
    assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_cohens_kappa_is_one_for_the_degenerate_single_category_case():
    # every item shares one category for both raters -- po==pe==1, not 0/0
    assert cohens_kappa(["a", "a", "a"], ["a", "a", "a"]) == 1.0


def test_cohens_kappa_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        cohens_kappa(["a", "b"], ["a"])


def test_cohens_kappa_raises_on_empty_input():
    with pytest.raises(ValueError):
        cohens_kappa([], [])


def test_fleiss_kappa_matches_a_known_worked_example():
    # 3 items, 3 raters, 2 categories: item1 unanimous A, item2 unanimous
    # B, item3 split 2A/1B
    ratings = [["A", "A", "A"], ["B", "B", "B"], ["A", "A", "B"]]

    kappa = fleiss_kappa(ratings)

    # p_j(A) = 5/9, p_j(B) = 4/9 -> p_e_bar = 25/81 + 16/81 = 41/81
    # P_i: item1 = (9-3)/6 = 1.0; item2 = (9-3)/6 = 1.0; item3 = (5-3)/6 = 1/3
    # p_bar = (1 + 1 + 1/3) / 3 = 7/9
    # kappa = (7/9 - 41/81) / (1 - 41/81) = (63/81 - 41/81) / (40/81) = 22/40 = 0.55
    assert math.isclose(kappa, 0.55)


def test_fleiss_kappa_is_one_for_unanimous_agreement_on_every_item():
    ratings = [["A", "A", "A"], ["B", "B", "B"]]
    assert fleiss_kappa(ratings) == 1.0


def test_fleiss_kappa_raises_when_fewer_than_two_raters():
    with pytest.raises(ValueError):
        fleiss_kappa([["A"], ["B"]])


def test_fleiss_kappa_raises_on_uneven_rater_counts():
    with pytest.raises(ValueError):
        fleiss_kappa([["A", "A"], ["B", "B", "B"]])


def test_fleiss_kappa_raises_on_empty_input():
    with pytest.raises(ValueError):
        fleiss_kappa([])


def test_wilson_score_interval_matches_a_known_worked_example():
    lower, upper = wilson_score_interval(80, 100, confidence=0.95)

    # verified independently via the closed-form formula, not transcribed
    # from memory: z=norm.ppf(0.975)~=1.95996, p_hat=0.8
    assert math.isclose(lower, 0.7111708344068411)
    assert math.isclose(upper, 0.8666330666689676)


def test_wilson_score_interval_contains_the_point_estimate():
    lower, upper = wilson_score_interval(8, 10)
    assert lower <= 0.8 <= upper


def test_wilson_score_interval_narrows_with_more_data_at_the_same_proportion():
    lower_small, upper_small = wilson_score_interval(8, 10)
    lower_large, upper_large = wilson_score_interval(800, 1000)
    assert (upper_large - lower_large) < (upper_small - lower_small)


def test_wilson_score_interval_raises_on_non_positive_n():
    with pytest.raises(ValueError):
        wilson_score_interval(0, 0)


def test_wilson_score_interval_raises_when_successes_exceeds_n():
    with pytest.raises(ValueError):
        wilson_score_interval(11, 10)


def test_agreement_vectors_drops_lost_candidates():
    outcomes = [
        MatchOutcome.TRUE_POSITIVE,
        MatchOutcome.LOST,
        MatchOutcome.FALSE_POSITIVE,
    ]
    detector, llm = agreement_vectors(outcomes, [True, True, False])
    assert len(detector) == len(llm) == 2


def test_agreement_vectors_detector_is_correct_exactly_on_true_positives():
    outcomes = [MatchOutcome.TRUE_POSITIVE, MatchOutcome.FALSE_POSITIVE]
    detector, _ = agreement_vectors(outcomes, [True, True])
    assert detector == [True, False]


def test_agreement_vectors_credits_the_llm_for_a_correct_suppression():
    """The case the old definition can't express: the candidate isn't a
    real secret, the LLM says so, the detector flagged it anyway."""
    detector, llm = agreement_vectors([MatchOutcome.FALSE_POSITIVE], [False])
    assert detector == [False]
    assert llm == [True]


def test_agreement_vectors_marks_a_discarded_true_positive_wrong():
    detector, llm = agreement_vectors([MatchOutcome.TRUE_POSITIVE], [False])
    assert detector == [True]
    assert llm == [False]


def test_agreement_vectors_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        agreement_vectors([MatchOutcome.TRUE_POSITIVE], [True, False])


def test_agreement_mcnemar_can_favour_the_llm_where_the_old_definition_cannot():
    """Two suppressed false positives, one discarded true positive. Under
    'flagged AND real' the LLM can only lose; under agreement it wins 2-1."""
    outcomes = [
        MatchOutcome.FALSE_POSITIVE,
        MatchOutcome.FALSE_POSITIVE,
        MatchOutcome.TRUE_POSITIVE,
    ]
    flagged = [False, False, False]

    old_detector = [o == MatchOutcome.TRUE_POSITIVE for o in outcomes]
    old_llm = [
        f and o == MatchOutcome.TRUE_POSITIVE for f, o in zip(flagged, outcomes, strict=True)
    ]
    # the LLM is never right where the detector isn't -- one cell is empty
    assert not any(m and not d for d, m in zip(old_detector, old_llm, strict=True))
    assert sum(old_llm) == 0

    detector, llm = agreement_vectors(outcomes, flagged)
    only_detector = sum(1 for d, m in zip(detector, llm, strict=True) if d and not m)
    only_llm = sum(1 for d, m in zip(detector, llm, strict=True) if not d and m)
    assert (only_detector, only_llm) == (1, 2)

    statistic, p_value = mcnemar_test(detector, llm)
    assert statistic == pytest.approx((abs(1 - 2) - 1) ** 2 / 3)
    assert 0.0 <= p_value <= 1.0


TP, FP, LOST = MatchOutcome.TRUE_POSITIVE, MatchOutcome.FALSE_POSITIVE, MatchOutcome.LOST


def test_candidate_confusion_counts_all_four_cells():
    outcomes = [TP, TP, FP, FP, TP, FP]
    flagged = [True, False, True, False, True, False]

    counts = candidate_confusion(outcomes, flagged)

    assert (counts.true_positive, counts.false_negative) == (2, 1)
    assert (counts.false_positive, counts.true_negative) == (1, 2)


def test_candidate_confusion_drops_lost():
    counts = candidate_confusion([TP, LOST, FP], [True, True, False])

    assert counts.true_positive + counts.false_positive == 1
    assert counts.true_negative == 1
    assert counts.false_negative == 0


def test_candidate_confusion_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        candidate_confusion([TP], [True, False])


def test_confusion_precision_recall_f1_from_hand_computed_counts():
    counts = ConfusionCounts(true_positive=6, false_positive=4, false_negative=2, true_negative=8)

    assert counts.precision == 0.6  # 6 / 10
    assert counts.recall == 0.75  # 6 / 8
    assert math.isclose(counts.f1, 2 * 0.6 * 0.75 / 1.35)


def test_matthews_corrcoef_matches_a_hand_computed_example():
    counts = ConfusionCounts(true_positive=3, false_positive=1, false_negative=2, true_negative=4)

    # ((3*4) - (1*2)) / sqrt(4 * 5 * 5 * 6) = 10 / sqrt(600)
    assert math.isclose(matthews_corrcoef(counts), 10 / math.sqrt(600))


def test_matthews_corrcoef_is_zero_for_a_flag_everything_detector():
    """The detector flags every candidate it generated, so it has no
    true negatives and no discrimination at all."""
    counts = ConfusionCounts(
        true_positive=105, false_positive=77, false_negative=0, true_negative=0
    )

    assert matthews_corrcoef(counts) == 0.0


def test_matthews_corrcoef_is_one_for_a_perfect_classifier():
    assert matthews_corrcoef(ConfusionCounts(5, 0, 0, 5)) == 1.0


def test_false_positive_rate_is_share_of_non_secrets_flagged():
    counts = ConfusionCounts(true_positive=3, false_positive=6, false_negative=1, true_negative=2)

    assert false_positive_rate(counts) == 0.75  # 6 / (6 + 2)


def test_false_positive_rate_is_zero_when_there_are_no_negatives():
    assert false_positive_rate(ConfusionCounts(4, 0, 1, 0)) == 0.0


def test_bootstrap_ci_brackets_the_point_estimate():
    outcomes = [TP] * 60 + [FP] * 40
    flagged = [True] * 100

    low, high = bootstrap_ci(outcomes, flagged, lambda c: c.precision, resamples=500)

    assert low <= 0.6 <= high


def test_bootstrap_ci_narrows_with_more_data_at_the_same_proportion():
    small = bootstrap_ci([TP] * 6 + [FP] * 4, [True] * 10, lambda c: c.precision, resamples=500)
    large = bootstrap_ci(
        [TP] * 600 + [FP] * 400, [True] * 1000, lambda c: c.precision, resamples=500
    )

    assert (large[1] - large[0]) < (small[1] - small[0])


def test_bootstrap_ci_is_reproducible_for_a_fixed_seed():
    outcomes, flagged = [TP] * 30 + [FP] * 20, [True] * 50
    first = bootstrap_ci(outcomes, flagged, lambda c: c.precision, resamples=200, seed=7)
    second = bootstrap_ci(outcomes, flagged, lambda c: c.precision, resamples=200, seed=7)

    assert first == second


def test_bootstrap_ci_returns_zeros_when_every_candidate_is_lost():
    assert bootstrap_ci([LOST, LOST], [True, False], lambda c: c.precision) == (0.0, 0.0)


def test_bootstrap_ci_rejects_bad_arguments():
    with pytest.raises(ValueError):
        bootstrap_ci([TP], [True], lambda c: c.precision, resamples=0)
    with pytest.raises(ValueError):
        bootstrap_ci([TP], [True], lambda c: c.precision, confidence=1.0)


def test_stratify_by_rule_splits_counts_per_rule():
    rule_ids = ["aws", "aws", "jwt"]
    counts = stratify_by_rule(rule_ids, [TP, FP, TP], [True, True, False])

    assert counts["aws"].true_positive == 1
    assert counts["aws"].false_positive == 1
    assert counts["jwt"].false_negative == 1


def test_stratify_by_rule_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        stratify_by_rule(["aws"], [TP, FP], [True, True])
