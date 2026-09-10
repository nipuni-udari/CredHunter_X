from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from credhunter_x.classifiers.tools.get_gitleaks_rule import get_gitleaks_rule
from credhunter_x.models.classification import Label, Severity
from credhunter_x.pipeline.orchestrator import ScanResult
from credhunter_x.reporting.rule_titles import rule_title

# Tracks pyproject.toml's [project].version -- SARIF's driver.version is
# informational only, not worth wiring up importlib.metadata for.
_TOOL_VERSION = "0.1.0"

_SCHEMA_URI = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
)

_SEVERITY_TO_LEVEL: dict[Severity, str] = {
    Severity.CRITICAL: "error",
    Severity.HIGH: "error",
    Severity.MEDIUM: "warning",
    Severity.LOW: "note",
}


def build_sarif_report(results: list[ScanResult]) -> dict[str, Any]:
    """Builds a SARIF 2.1.0 document for GitHub code scanning. Only
    true_secret and uncertain become results -- false_positive is
    excluded, or this would just reproduce GitLeaks' own alert-fatigue
    problem. uncertain is always "note" level regardless of severity."""
    included = [
        r for r in results if r.classification.label in (Label.TRUE_SECRET, Label.UNCERTAIN)
    ]
    rule_ids = sorted({r.candidate.rule_id for r in included})

    return {
        "$schema": _SCHEMA_URI,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "CredHunter-X",
                        "version": _TOOL_VERSION,
                        "informationUri": "https://github.com/nipuni-udari/CredHunter_X",
                        "rules": [_build_rule(rule_id) for rule_id in rule_ids],
                    }
                },
                "results": [_build_result(r) for r in included],
            }
        ],
    }


def _build_rule(rule_id: str) -> dict[str, Any]:
    return {
        "id": rule_id,
        "name": rule_id,
        "shortDescription": {"text": rule_title(rule_id)},
        "fullDescription": {"text": get_gitleaks_rule(rule_id)},
    }


def _build_result(result: ScanResult) -> dict[str, Any]:
    candidate = result.candidate
    classification = result.classification
    level = (
        "note"
        if classification.label == Label.UNCERTAIN
        else _SEVERITY_TO_LEVEL.get(classification.severity, "warning")
    )
    return {
        "ruleId": candidate.rule_id,
        "level": level,
        "message": {
            "text": (
                f"[{classification.label}, confidence={classification.confidence:.2f}] "
                f"{classification.explanation} Suggested remediation: {classification.remediation}"
            )
        },
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": candidate.file_path},
                    "region": {
                        "startLine": max(candidate.line_start, 1),
                        "endLine": max(candidate.line_end, candidate.line_start, 1),
                    },
                }
            }
        ],
    }


def write_sarif_report(results: list[ScanResult], path: Path) -> None:
    path.write_text(json.dumps(build_sarif_report(results), indent=2), encoding="utf-8")
