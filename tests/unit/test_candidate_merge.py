from __future__ import annotations

from credhunter_x.models.candidate import Candidate
from credhunter_x.pipeline.candidate_merge import merge_candidates


def _make_candidate(
    *,
    id: str = "c1",
    file_path: str = "app/auth.py",
    line_start: int = 1,
    line_end: int = 1,
    rule_id: str = "aws-key",
    matched_value: str = "AKIAFAKEFAKEFAKEFAKE",
    source: str = "gitleaks",
) -> Candidate:
    return Candidate(
        id=id,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        rule_id=rule_id,
        matched_value=matched_value,
        value_start=0,
        value_end=len(matched_value),
        entropy=4.0,
        matched_lines=[f"key = '{matched_value}'"],
        context_before=[],
        context_after=[],
        repo_id="repo1",
        source=source,
    )


def test_same_file_and_line_and_overlapping_value_merges_into_one():
    gitleaks = [_make_candidate(id="g1", matched_value="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")]
    trufflehog = [
        _make_candidate(
            id="t1", matched_value="wWPw5k4aXcaT4fNP0UcnZwJUVFk6LO0pINUx", source="trufflehog"
        )
    ]

    merged = merge_candidates(gitleaks, trufflehog)

    assert len(merged) == 1
    assert merged[0].id == "g1"
    assert merged[0].source == "gitleaks+trufflehog"


def test_different_files_do_not_merge():
    gitleaks = [_make_candidate(id="g1", file_path="app/a.py")]
    trufflehog = [_make_candidate(id="t1", file_path="app/b.py", source="trufflehog")]

    merged = merge_candidates(gitleaks, trufflehog)

    assert len(merged) == 2
    assert {c.id for c in merged} == {"g1", "t1"}


def test_non_overlapping_lines_do_not_merge():
    gitleaks = [_make_candidate(id="g1", line_start=1, line_end=1)]
    trufflehog = [_make_candidate(id="t1", line_start=5, line_end=5, source="trufflehog")]

    merged = merge_candidates(gitleaks, trufflehog)

    assert len(merged) == 2


def test_same_line_but_unrelated_values_do_not_merge():
    gitleaks = [_make_candidate(id="g1", matched_value="AKIAFAKEFAKEFAKEFAKE")]
    trufflehog = [
        _make_candidate(id="t1", matched_value="totally-different-secret", source="trufflehog")
    ]

    merged = merge_candidates(gitleaks, trufflehog)

    assert len(merged) == 2


def test_unmatched_candidates_from_both_sides_survive_unchanged():
    gitleaks = [_make_candidate(id="g1", file_path="app/a.py")]
    trufflehog = [_make_candidate(id="t1", file_path="app/b.py", source="trufflehog")]

    merged = merge_candidates(gitleaks, trufflehog)

    g1 = next(c for c in merged if c.id == "g1")
    t1 = next(c for c in merged if c.id == "t1")
    assert g1.source == "gitleaks"
    assert t1.source == "trufflehog"


def test_empty_secondary_list_returns_primary_unchanged():
    gitleaks = [_make_candidate(id="g1")]
    assert merge_candidates(gitleaks, []) == gitleaks


def test_empty_primary_list_returns_secondary_unchanged():
    trufflehog = [_make_candidate(id="t1", source="trufflehog")]
    assert merge_candidates([], trufflehog) == trufflehog


def test_one_detector_reporting_the_same_secret_twice_collapses():
    gitleaks = [
        _make_candidate(id="g1", matched_value="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"),
        _make_candidate(id="g2", matched_value="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"),
    ]
    trufflehog = [
        _make_candidate(
            id="t1", matched_value="wWPw5k4aXcaT4fNP0UcnZwJUVFk6LO0pINUx", source="trufflehog"
        )
    ]

    merged = merge_candidates(gitleaks, trufflehog)

    assert [c.id for c in merged] == ["g1"]
    assert merged[0].source == "gitleaks+trufflehog"


def test_one_candidate_absorbs_every_matching_secondary():
    """One value trips several trufflehog rules at once; pairing them off one
    at a time used to leave the surplus behind as separate candidates."""
    gitleaks = [_make_candidate(id="g1", matched_value="ghp_wWPw5k4aXcaT4fNP0UcnZwJUVFk6LO0")]
    trufflehog = [
        _make_candidate(id="t1", matched_value="wWPw5k4aXcaT4fNP0UcnZwJUVFk6LO0", source="th"),
        _make_candidate(id="t2", matched_value="ghp_wWPw5k4aXcaT4fNP0UcnZwJUVFk6LO0", source="th"),
        _make_candidate(id="t3", matched_value="wWPw5k4aXcaT4fNP0Ucn", source="th"),
    ]

    merged = merge_candidates(gitleaks, trufflehog)

    assert [c.id for c in merged] == ["g1"]
    assert merged[0].source == "gitleaks+th"


def test_secondary_repeats_with_no_primary_match_also_collapse():
    trufflehog = [
        _make_candidate(id="t1", matched_value="AKIAFAKEFAKEFAKEFAKE", source="trufflehog"),
        _make_candidate(id="t2", matched_value="AKIAFAKEFAKEFAKEFAKE", source="trufflehog"),
    ]

    merged = merge_candidates([], trufflehog)

    assert [c.id for c in merged] == ["t1"]


def test_different_secrets_on_the_same_line_are_both_kept():
    gitleaks = [
        _make_candidate(id="g1", matched_value="AKIAFAKEFAKEFAKEFAKE"),
        _make_candidate(id="g2", matched_value="ghp_TOTALLYDIFFERENTVALUE0000000000"),
    ]

    merged = merge_candidates(gitleaks, [])

    assert [c.id for c in merged] == ["g1", "g2"]
