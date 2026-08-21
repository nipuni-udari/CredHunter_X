from __future__ import annotations

from pathlib import Path

import pytest

from credhunter_x.evaluation.remediation_reference import load_remediation_reference

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_PATH = FIXTURES_DIR / "remediation_reference_sample.yaml"


def test_loads_all_entries():
    reference = load_remediation_reference(SAMPLE_PATH)
    assert set(reference) == {"aws-access-token", "github-pat"}


def test_fields_map_correctly_for_a_known_entry():
    reference = load_remediation_reference(SAMPLE_PATH)
    entry = reference["aws-access-token"]

    assert entry.rule_id == "aws-access-token"
    assert entry.required_elements == [
        "move to environment variable or secrets manager",
        "rotate or revoke the exposed key",
    ]
    assert entry.source == "OWASP Secrets Management Cheat Sheet"


def test_raises_a_clear_error_when_file_is_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_remediation_reference(tmp_path / "does_not_exist.yaml")
