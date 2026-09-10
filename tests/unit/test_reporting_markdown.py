from __future__ import annotations

from pathlib import Path

from credhunter_x.models.candidate import Candidate
from credhunter_x.models.classification import ClassificationResult, Label, Severity
from credhunter_x.models.treatment import Treatment
from credhunter_x.pipeline.orchestrator import ScanResult
from credhunter_x.reporting.markdown_summary import (
    build_markdown_summary,
    write_markdown_summary,
)


def _scan_result(
    *,
    label: Label,
    severity: Severity = Severity.LOW,
    explanation: str = "a test explanation",
    file_path: str = "app/config.py",
    rule_id: str = "aws-access-token",
) -> ScanResult:
    candidate = Candidate(
        id=f"{file_path}:{rule_id}:1",
        file_path=file_path,
        line_start=1,
        line_end=1,
        rule_id=rule_id,
        matched_value="irrelevant-for-reporting",
        value_start=0,
        value_end=5,
        entropy=4.0,
        matched_lines=["x = 'irrelevant-for-reporting'"],
        context_before=[],
        context_after=[],
        repo_id="test-repo",
    )
    classification = ClassificationResult(
        candidate_id=candidate.id,
        arm="agentic",
        treatment=Treatment.MASKED,
        label=label,
        confidence=0.9,
        severity=severity,
        explanation=explanation,
        remediation="a test remediation",
        raw_model_output="{}",
    )
    return ScanResult(candidate=candidate, classification=classification)


def test_counts_every_label_in_the_headline():
    summary = build_markdown_summary(
        [
            _scan_result(label=Label.TRUE_SECRET),
            _scan_result(label=Label.UNCERTAIN),
            _scan_result(label=Label.FALSE_POSITIVE),
        ]
    )

    assert "**1 secret(s) found**" in summary
    assert "1 need review" in summary
    assert "1 dismissed" in summary
    assert "3 candidate(s) scanned" in summary


def test_dismissed_findings_stay_out_of_the_table():
    summary = build_markdown_summary(
        [
            _scan_result(label=Label.TRUE_SECRET, file_path="app/real.py"),
            _scan_result(label=Label.FALSE_POSITIVE, file_path="tests/fake.py"),
        ]
    )

    table = summary.split("<details>")[0]
    assert "app/real.py" in table
    assert "tests/fake.py" not in table
    # ...but they are still reported, in their own collapsed section.
    assert "tests/fake.py" in summary
    assert "1 dismissed as false positive(s)" in summary


def test_rule_ids_are_shown_as_readable_titles():
    summary = build_markdown_summary(
        [_scan_result(label=Label.TRUE_SECRET, rule_id="generic.password-in-url")]
    )

    assert "Password in a connection URL" in summary


def test_unknown_rule_id_falls_back_to_the_id():
    summary = build_markdown_summary(
        [_scan_result(label=Label.TRUE_SECRET, rule_id="some-made-up-rule")]
    )

    assert "some-made-up-rule" in summary


def test_no_findings_says_so():
    summary = build_markdown_summary([])

    assert "**No secrets found**" in summary


def test_skipped_candidates_are_flagged_as_an_incomplete_scan():
    summary = build_markdown_summary([_scan_result(label=Label.TRUE_SECRET)], skipped_count=2)

    assert "2 candidate(s) could not be classified" in summary
    assert "incomplete" in summary


def test_remediation_is_not_duplicated_into_the_summary():
    summary = build_markdown_summary([_scan_result(label=Label.TRUE_SECRET)])

    assert "a test explanation" in summary
    assert "a test remediation" not in summary


def test_pipes_in_text_do_not_break_the_table():
    summary = build_markdown_summary(
        [_scan_result(label=Label.TRUE_SECRET, file_path="app/a|b.py")]
    )

    assert "a\\|b.py" in summary


def test_write_appends_rather_than_overwriting(tmp_path: Path):
    path = tmp_path / "summary.md"
    path.write_text("## An earlier step wrote this\n", encoding="utf-8")

    write_markdown_summary([_scan_result(label=Label.TRUE_SECRET)], path)

    written = path.read_text(encoding="utf-8")
    assert written.startswith("## An earlier step wrote this")
    assert "## CredHunter-X" in written
