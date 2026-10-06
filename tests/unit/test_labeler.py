from __future__ import annotations

from dataclasses import replace

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
    """CredData's GitLeaks integration (benchmark/scanner/gitleaks.py) only
    checks file and line, never the rule, so rule_id must not affect matching."""
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
    """CredData returns the first matching row at a file and line, so the
    first row loaded must decide the outcome."""
    candidate = make_candidate()
    first_row = make_gt_row(row_id="first", ground_truth=True)
    second_row = make_gt_row(row_id="second", ground_truth=False)
    gt_index = index_ground_truth([first_row, second_row])

    assert match_candidate(candidate, gt_index) == MatchOutcome.TRUE_POSITIVE


def test_offsets_choose_which_of_two_conflicting_rows_applies():
    """One line often has two labelled items that disagree, e.g. a key id
    labelled false and a key labelled true. The candidate must be scored
    against the row covering its own characters."""
    # candidate sits at 57-101, the same characters as the False row
    candidate = make_candidate()
    candidate = replace(candidate, value_start=57, value_end=101)
    secret_row = replace(
        make_gt_row(row_id="secret", ground_truth=True), value_start=5, value_end=15
    )
    not_secret_row = replace(
        make_gt_row(row_id="not-secret", ground_truth=False), value_start=57, value_end=101
    )
    gt_index = index_ground_truth([secret_row, not_secret_row])

    assert match_candidate(candidate, gt_index) == MatchOutcome.FALSE_POSITIVE


def test_offsets_pick_the_true_row_when_that_is_the_one_matched():
    """The mirror case: the result follows the candidate's position."""
    candidate = replace(make_candidate(), value_start=57, value_end=101)
    not_secret_row = replace(
        make_gt_row(row_id="not-secret", ground_truth=False), value_start=5, value_end=15
    )
    secret_row = replace(
        make_gt_row(row_id="secret", ground_truth=True), value_start=57, value_end=101
    )
    gt_index = index_ground_truth([not_secret_row, secret_row])

    assert match_candidate(candidate, gt_index) == MatchOutcome.TRUE_POSITIVE


def test_falls_back_to_first_row_when_the_candidate_has_no_known_position():
    """value_start is -1 when the value couldn't be found in its line; the
    first row is used."""
    candidate = replace(make_candidate(), value_start=-1, value_end=-1)
    first_row = make_gt_row(row_id="first", ground_truth=True)
    second_row = replace(
        make_gt_row(row_id="second", ground_truth=False), value_start=57, value_end=101
    )
    gt_index = index_ground_truth([first_row, second_row])

    assert match_candidate(candidate, gt_index) == MatchOutcome.TRUE_POSITIVE


def test_falls_back_to_first_row_when_ground_truth_has_no_offsets():
    """Missing offsets (-1 or absent) fall back to the first row."""
    candidate = replace(make_candidate(), value_start=57, value_end=101)
    first_row = replace(
        make_gt_row(row_id="first", ground_truth=True), value_start=-1, value_end=-1
    )
    second_row = replace(
        make_gt_row(row_id="second", ground_truth=False), value_start=-1, value_end=-1
    )
    gt_index = index_ground_truth([first_row, second_row])

    assert match_candidate(candidate, gt_index) == MatchOutcome.TRUE_POSITIVE


def test_falls_back_to_first_row_when_no_row_overlaps_the_candidate():
    """Offsets that don't line up with any row must not pick an unrelated one."""
    candidate = replace(make_candidate(), value_start=200, value_end=220)
    first_row = make_gt_row(row_id="first", ground_truth=True)
    second_row = replace(
        make_gt_row(row_id="second", ground_truth=False), value_start=57, value_end=101
    )
    gt_index = index_ground_truth([first_row, second_row])

    assert match_candidate(candidate, gt_index) == MatchOutcome.TRUE_POSITIVE


def test_adjacent_but_non_overlapping_offsets_do_not_count_as_a_match():
    """Half-open ranges: a row ending where the candidate starts doesn't
    overlap it."""
    candidate = replace(make_candidate(), value_start=15, value_end=30)
    touching_row = replace(
        make_gt_row(row_id="touching", ground_truth=False), value_start=5, value_end=15
    )
    real_row = replace(make_gt_row(row_id="real", ground_truth=True), value_start=15, value_end=30)
    gt_index = index_ground_truth([touching_row, real_row])

    assert match_candidate(candidate, gt_index) == MatchOutcome.TRUE_POSITIVE
