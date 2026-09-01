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


def test_offsets_choose_which_of_two_conflicting_rows_applies():
    """A line commonly carries two labelled items that disagree -- e.g.
    PublicKey(key_id="568...", key="g0y8X95+...") where the id is labelled
    not-a-secret and the key is. file+line cannot tell them apart, so the
    candidate must be scored against the row covering the characters it
    actually matched, not whichever row loaded first."""
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
    """The mirror of the test above -- the tie-break must not simply
    invert the old answer, it must follow the candidate's position."""
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
    """value_start is -1 when the scanner reported a value we could not
    locate verbatim in its line (12 of 517 in the corpus). An unknown
    position is not evidence, so behaviour must be exactly as before."""
    candidate = replace(make_candidate(), value_start=-1, value_end=-1)
    first_row = make_gt_row(row_id="first", ground_truth=True)
    second_row = replace(
        make_gt_row(row_id="second", ground_truth=False), value_start=57, value_end=101
    )
    gt_index = index_ground_truth([first_row, second_row])

    assert match_candidate(candidate, gt_index) == MatchOutcome.TRUE_POSITIVE


def test_falls_back_to_first_row_when_ground_truth_has_no_offsets():
    """CredData writes -1 when a row records no offsets; other datasets may
    omit them entirely. The tie-break must degrade to the old behaviour
    rather than mis-score."""
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
    """Offsets that simply do not line up (seen on 3 corpus candidates)
    must not silently pick an unrelated row."""
    candidate = replace(make_candidate(), value_start=200, value_end=220)
    first_row = make_gt_row(row_id="first", ground_truth=True)
    second_row = replace(
        make_gt_row(row_id="second", ground_truth=False), value_start=57, value_end=101
    )
    gt_index = index_ground_truth([first_row, second_row])

    assert match_candidate(candidate, gt_index) == MatchOutcome.TRUE_POSITIVE


def test_adjacent_but_non_overlapping_offsets_do_not_count_as_a_match():
    """Half-open ranges: a row ending exactly where the candidate begins
    shares no characters with it."""
    candidate = replace(make_candidate(), value_start=15, value_end=30)
    touching_row = replace(
        make_gt_row(row_id="touching", ground_truth=False), value_start=5, value_end=15
    )
    real_row = replace(make_gt_row(row_id="real", ground_truth=True), value_start=15, value_end=30)
    gt_index = index_ground_truth([touching_row, real_row])

    assert match_candidate(candidate, gt_index) == MatchOutcome.TRUE_POSITIVE
