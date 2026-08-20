from __future__ import annotations

from credhunter_x.dataset.creddata import GroundTruthRow
from credhunter_x.evaluation.labeler import MatchOutcome, index_ground_truth, match_candidate
from credhunter_x.models.candidate import Candidate


def make_candidate(
    file_path: str = "app.py",
    line_start: int = 10,
    line_end: int = 10,
    rule_id: str = "aws-key",
) -> Candidate:
    return Candidate(
        id="c1",
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        rule_id=rule_id,
        matched_value="AKIAFAKEFAKEFAKEFAKE",
        value_start=5,
        value_end=15,
        entropy=4.0,
        matched_lines=["key = 'AKIAFAKEFAKEFAKEFAKE'"],
        context_before=[],
        context_after=[],
        repo_id="repo1",
    )


def make_gt_row(
    row_id: str = "gt1",
    file_path: str = "app.py",
    line_start: int = 10,
    line_end: int = 10,
    ground_truth: bool = True,
    category: str = "aws-key",
) -> GroundTruthRow:
    return GroundTruthRow(
        id=row_id,
        file_id="f1",
        repo_id="repo1",
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        ground_truth=ground_truth,
        value_start=5,
        value_end=15,
        category=category,
    )


def test_no_ground_truth_at_location_is_lost():
    candidate = make_candidate()
    gt_index = index_ground_truth([])

    assert match_candidate(candidate, gt_index) == MatchOutcome.LOST


def test_matches_on_file_and_line_regardless_of_rule_id():
    """Verified against CredData's real GitLeaks integration
    (benchmark/scanner/gitleaks.py): it never checks rule/category for
    GitLeaks findings, only file+line — so a candidate's rule_id must not
    affect whether it matches."""
    candidate = make_candidate(rule_id="github-pat")
    gt_row = make_gt_row(ground_truth=True, category="totally-unrelated-category")
    gt_index = index_ground_truth([gt_row])

    assert match_candidate(candidate, gt_index) == MatchOutcome.TRUE_POSITIVE


def test_matched_location_with_true_label_is_true_positive():
    candidate = make_candidate()
    gt_row = make_gt_row(ground_truth=True)
    gt_index = index_ground_truth([gt_row])

    assert match_candidate(candidate, gt_index) == MatchOutcome.TRUE_POSITIVE


def test_matched_location_with_false_label_is_false_positive():
    candidate = make_candidate()
    gt_row = make_gt_row(ground_truth=False)
    gt_index = index_ground_truth([gt_row])

    assert match_candidate(candidate, gt_index) == MatchOutcome.FALSE_POSITIVE


def test_different_file_or_line_is_lost():
    candidate = make_candidate(file_path="other.py")
    gt_row = make_gt_row(file_path="app.py", ground_truth=True)
    gt_index = index_ground_truth([gt_row])

    assert match_candidate(candidate, gt_index) == MatchOutcome.LOST


def test_first_row_at_a_location_wins_when_several_exist():
    """CredData's own loop returns on the first matching row at a given
    file+line — index_ground_truth preserves CSV load order, so the first
    row registered for a key must be the one that decides the outcome."""
    candidate = make_candidate()
    first_row = make_gt_row(row_id="first", ground_truth=True)
    second_row = make_gt_row(row_id="second", ground_truth=False)
    gt_index = index_ground_truth([first_row, second_row])

    assert match_candidate(candidate, gt_index) == MatchOutcome.TRUE_POSITIVE
