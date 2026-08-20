from __future__ import annotations

import math

import pytest

from credhunter_x.evaluation.labeler import MatchOutcome
from credhunter_x.evaluation.metrics import compute_metrics, mcnemar_test


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
