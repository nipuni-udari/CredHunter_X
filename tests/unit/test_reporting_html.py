from __future__ import annotations

from pathlib import Path

from credhunter_x.models.candidate import Candidate
from credhunter_x.models.classification import ClassificationResult, Label, Severity
from credhunter_x.models.treatment import Treatment
from credhunter_x.pipeline.orchestrator import ScanResult
from credhunter_x.reporting.html_report import build_html_report, write_html_report


def _scan_result(
    *,
    label: Label,
    severity: Severity = Severity.LOW,
    explanation: str = "a test explanation",
    file_path: str = "app/config.py",
) -> ScanResult:
    candidate = Candidate(
        id=f"{file_path}:rule:1",
        file_path=file_path,
        line_start=1,
        line_end=1,
        rule_id="aws-access-token",
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


def test_build_html_report_includes_all_three_labels():
    results = [
        _scan_result(label=Label.TRUE_SECRET),
        _scan_result(label=Label.UNCERTAIN),
        _scan_result(label=Label.FALSE_POSITIVE),
    ]

    html = build_html_report(results)

    assert "Secrets found" in html
    assert "Needs manual review" in html
    assert "Reviewed and dismissed as false positives" in html
    assert 'class="empty"' not in html


def test_build_html_report_escapes_script_tags_in_model_generated_text():
    """explanation/remediation are LLM-generated free text over untrusted
    repo content -- must never be interpolated into the report unescaped,
    or a crafted comment/string in a scanned repo could inject a script
    into a report a developer opens in a browser."""
    payload = "<script>alert(1)</script>"
    results = [_scan_result(label=Label.TRUE_SECRET, explanation=payload)]

    html = build_html_report(results)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_build_html_report_escapes_file_path():
    results = [_scan_result(label=Label.TRUE_SECRET, file_path="<injected>.py")]

    html = build_html_report(results)

    assert "<injected>.py" not in html
    assert "&lt;injected&gt;.py" in html


def test_build_html_report_summary_counts_match_findings():
    results = [
        _scan_result(label=Label.TRUE_SECRET),
        _scan_result(label=Label.TRUE_SECRET),
        _scan_result(label=Label.FALSE_POSITIVE),
    ]

    html = build_html_report(results)

    assert '<span class="count">2</span>secrets found' in html
    assert '<span class="count">0</span>need review' in html
    assert '<span class="count">1</span>dismissed' in html


def test_build_html_report_with_no_results_shows_empty_sections():
    html = build_html_report([])

    assert html.count('class="empty"') == 3


def test_write_html_report_writes_a_file(tmp_path: Path):
    out = tmp_path / "report.html"
    write_html_report([_scan_result(label=Label.TRUE_SECRET)], out)

    content = out.read_text(encoding="utf-8")
    assert "<!doctype html>" in content
    assert "Secrets found" in content
