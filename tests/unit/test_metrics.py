from __future__ import annotations

import math

import pytest

from credhunter_x.evaluation.labeler import MatchOutcome
from credhunter_x.evaluation.metrics import (
    cohens_kappa,
    compute_metrics,
    fleiss_kappa,
    mcnemar_test,
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
