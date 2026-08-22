from __future__ import annotations

import json
from pathlib import Path

from credhunter_x.models.candidate import Candidate
from credhunter_x.models.classification import ClassificationResult, Label, Severity
from credhunter_x.models.treatment import Treatment
from credhunter_x.pipeline.orchestrator import ScanResult
from credhunter_x.reporting.sarif import build_sarif_report, write_sarif_report


def _scan_result(
    *,
    label: Label,
    severity: Severity = Severity.LOW,
    rule_id: str = "aws-access-token",
    file_path: str = "app/config.py",
    line_start: int = 10,
    line_end: int = 10,
) -> ScanResult:
    candidate = Candidate(
        id=f"{file_path}:{rule_id}:{line_start}",
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
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
        explanation="a test explanation",
        remediation="a test remediation",
        raw_model_output="{}",
    )
    return ScanResult(candidate=candidate, classification=classification)


def test_build_sarif_report_excludes_false_positives():
    results = [
        _scan_result(label=Label.TRUE_SECRET, severity=Severity.HIGH),
        _scan_result(label=Label.FALSE_POSITIVE),
        _scan_result(label=Label.UNCERTAIN),
    ]

    report = build_sarif_report(results)
    sarif_results = report["runs"][0]["results"]

    assert len(sarif_results) == 2
    labels_mentioned = {r["message"]["text"].split(",")[0].strip("[") for r in sarif_results}
    assert labels_mentioned == {"true_secret", "uncertain"}


def test_build_sarif_report_maps_severity_to_level():
    results = [
        _scan_result(
            label=Label.TRUE_SECRET, severity=Severity.CRITICAL, rule_id="a", line_start=1
        ),
        _scan_result(label=Label.TRUE_SECRET, severity=Severity.HIGH, rule_id="b", line_start=2),
        _scan_result(label=Label.TRUE_SECRET, severity=Severity.MEDIUM, rule_id="c", line_start=3),
        _scan_result(label=Label.TRUE_SECRET, severity=Severity.LOW, rule_id="d", line_start=4),
    ]

    sarif_results = build_sarif_report(results)["runs"][0]["results"]
    levels_by_rule = {r["ruleId"]: r["level"] for r in sarif_results}

    assert levels_by_rule == {"a": "error", "b": "error", "c": "warning", "d": "note"}


def test_build_sarif_report_uncertain_is_always_note_even_at_high_severity():
    results = [_scan_result(label=Label.UNCERTAIN, severity=Severity.CRITICAL)]

    level = build_sarif_report(results)["runs"][0]["results"][0]["level"]

    assert level == "note"


def test_build_sarif_report_location_uses_file_path_and_line_numbers():
    results = [
        _scan_result(label=Label.TRUE_SECRET, file_path="src/keys.py", line_start=42, line_end=44)
    ]

    location = build_sarif_report(results)["runs"][0]["results"][0]["locations"][0]
    physical = location["physicalLocation"]

    assert physical["artifactLocation"]["uri"] == "src/keys.py"
    assert physical["region"]["startLine"] == 42
    assert physical["region"]["endLine"] == 44


def test_build_sarif_report_rules_list_has_one_entry_per_included_rule_id():
    results = [
        _scan_result(label=Label.TRUE_SECRET, rule_id="aws-access-token", line_start=1),
        _scan_result(label=Label.TRUE_SECRET, rule_id="aws-access-token", line_start=2),
        _scan_result(label=Label.TRUE_SECRET, rule_id="github-pat", line_start=3),
        _scan_result(label=Label.FALSE_POSITIVE, rule_id="jwt", line_start=4),
    ]

    rule_ids = {r["id"] for r in build_sarif_report(results)["runs"][0]["tool"]["driver"]["rules"]}

    assert rule_ids == {"aws-access-token", "github-pat"}


def test_build_sarif_report_with_no_results_is_still_well_formed():
    report = build_sarif_report([])

    assert report["version"] == "2.1.0"
    assert report["runs"][0]["results"] == []
    assert report["runs"][0]["tool"]["driver"]["rules"] == []


def test_write_sarif_report_writes_valid_json(tmp_path: Path):
    out = tmp_path / "report.sarif"
    results = [_scan_result(label=Label.TRUE_SECRET, severity=Severity.HIGH)]

    write_sarif_report(results, out)

    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["version"] == "2.1.0"
    assert len(parsed["runs"][0]["results"]) == 1
