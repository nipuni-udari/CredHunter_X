from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import credhunter_x.cli.main as cli_main
from credhunter_x.models.candidate import Candidate
from credhunter_x.models.classification import ClassificationResult, Label, Severity
from credhunter_x.models.treatment import Treatment
from credhunter_x.pipeline.orchestrator import ScanOutcome, ScanResult

runner = CliRunner()


def _scan_result(label: Label, file_path: str = "app/config.py") -> ScanResult:
    candidate = Candidate(
        id=f"{file_path}:rule:1",
        file_path=file_path,
        line_start=1,
        line_end=1,
        rule_id="aws-access-token",
        matched_value="irrelevant-for-cli-tests",
        value_start=0,
        value_end=5,
        entropy=4.0,
        matched_lines=["x = 'irrelevant-for-cli-tests'"],
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
        severity=Severity.HIGH,
        explanation="a test explanation",
        remediation="a test remediation",
        raw_model_output="{}",
    )
    return ScanResult(candidate=candidate, classification=classification)


def _patch_scan_repository(
    monkeypatch: pytest.MonkeyPatch, results: list[ScanResult], *, skipped_count: int = 0
) -> None:
    outcome = ScanOutcome(results=results, skipped_count=skipped_count)
    monkeypatch.setattr(cli_main, "scan_repository", lambda *args, **kwargs: outcome)


def test_scan_exits_zero_with_no_findings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_scan_repository(monkeypatch, [])

    result = runner.invoke(cli_main.app, [str(tmp_path)])

    assert result.exit_code == 0
    assert "No candidates found" in result.stdout


def test_scan_exits_zero_when_only_false_positives_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _patch_scan_repository(monkeypatch, [_scan_result(Label.FALSE_POSITIVE)])

    result = runner.invoke(cli_main.app, [str(tmp_path)])

    assert result.exit_code == 0


def test_scan_exits_one_when_a_true_secret_is_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _patch_scan_repository(monkeypatch, [_scan_result(Label.TRUE_SECRET)])

    result = runner.invoke(cli_main.app, [str(tmp_path)])

    assert result.exit_code == 1


def test_scan_exits_zero_for_uncertain_findings_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """uncertain means "couldn't decide," not "confirmed" -- it shouldn't
    fail a CI check by itself, only surface in the report for a human."""
    _patch_scan_repository(monkeypatch, [_scan_result(Label.UNCERTAIN)])

    result = runner.invoke(cli_main.app, [str(tmp_path)])

    assert result.exit_code == 0


def test_scan_writes_sarif_and_html_reports_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _patch_scan_repository(monkeypatch, [_scan_result(Label.TRUE_SECRET)])
    sarif_path = tmp_path / "out.sarif"
    html_path = tmp_path / "out.html"

    result = runner.invoke(
        cli_main.app,
        [str(tmp_path), "--sarif", str(sarif_path), "--html", str(html_path)],
    )

    assert result.exit_code == 1
    assert sarif_path.exists()
    assert html_path.exists()
    assert "aws-access-token" in sarif_path.read_text(encoding="utf-8")
    assert "Secrets found" in html_path.read_text(encoding="utf-8")


def test_scan_writes_reports_even_when_no_candidates_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _patch_scan_repository(monkeypatch, [])
    sarif_path = tmp_path / "out.sarif"

    result = runner.invoke(cli_main.app, [str(tmp_path), "--sarif", str(sarif_path)])

    assert result.exit_code == 0
    assert sarif_path.exists()


def test_scan_exits_one_when_a_candidate_was_skipped_even_with_no_true_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The exact bug found live in the GitHub Action demo: a candidate that
    couldn't be classified must never be treated as equivalent to a clean
    scan, even when every candidate that DID get classified came back
    false_positive -- an unresolved candidate might have been the real
    secret."""
    _patch_scan_repository(monkeypatch, [_scan_result(Label.FALSE_POSITIVE)], skipped_count=1)

    result = runner.invoke(cli_main.app, [str(tmp_path)])

    assert result.exit_code == 1
    assert "1 candidate(s) could not be classified" in result.stdout
    assert "incomplete" in result.stdout


def test_scan_does_not_print_no_candidates_found_when_some_were_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Distinguishes "gitleaks found nothing" from "gitleaks found
    something but every candidate errored out" -- the latter must not look
    identical to a clean scan."""
    _patch_scan_repository(monkeypatch, [], skipped_count=2)

    result = runner.invoke(cli_main.app, [str(tmp_path)])

    assert result.exit_code == 1
    assert "No candidates found" not in result.stdout
    assert "2 candidate(s) could not be classified" in result.stdout
