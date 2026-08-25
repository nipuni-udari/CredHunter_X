from __future__ import annotations

import json
from pathlib import Path

from credhunter_x.trufflehog.parser import parse_trufflehog_report

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_REPO = FIXTURES_DIR / "sample_repo"
REPORT_PATH = FIXTURES_DIR / "trufflehog_report.json"


def load_findings() -> list[dict]:
    return json.loads(REPORT_PATH.read_text())


def test_parses_all_findings_into_candidates():
    candidates = parse_trufflehog_report(load_findings(), SAMPLE_REPO, repo_id="test-repo")
    assert len(candidates) == 3


def test_every_candidate_is_tagged_with_trufflehog_source():
    candidates = parse_trufflehog_report(load_findings(), SAMPLE_REPO, repo_id="test-repo")
    assert all(c.source == "trufflehog" for c in candidates)


def test_maps_fields_correctly_for_github_token_finding():
    candidates = parse_trufflehog_report(
        load_findings(), SAMPLE_REPO, repo_id="test-repo", context_lines=2
    )
    github = next(c for c in candidates if c.line_start == 1)

    assert github.id == "app/auth.py:high-entropy:1"
    assert github.file_path == "app/auth.py"
    assert github.line_start == 1
    assert github.line_end == 1
    assert github.rule_id == "high-entropy"
    # trufflehog3's "secret" is only the high-entropy fragment it matched,
    # not the whole ghp_-prefixed token gitleaks would report.
    assert github.matched_value == "wWPw5k4aXcaT4fNP0UcnZwJUVFk6LO0pINUx"
    assert github.value_start == 20
    assert github.value_end == 56
    assert github.repo_id == "test-repo"
    assert github.matched_lines == ['GITHUB_TOKEN = "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"']
    assert github.context_before == []
    assert github.context_after == [
        'SLACK_BOT_TOKEN = "xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"',
        "",
    ]


def test_maps_fields_correctly_for_password_in_url_finding():
    candidates = parse_trufflehog_report(
        load_findings(), SAMPLE_REPO, repo_id="test-repo", context_lines=2
    )
    db = next(c for c in candidates if c.rule_id == "generic.password-in-url")

    assert db.file_path == "app/db.py"
    assert db.line_start == 5
    assert db.line_end == 5
    assert db.matched_value == 'postgres://user:{DATABASE_PASSWORD}@localhost/db"'
    assert db.context_before == ["", "def connect():"]
    assert db.context_after == []


def test_entropy_is_computed_since_trufflehog_reports_none():
    candidates = parse_trufflehog_report(load_findings(), SAMPLE_REPO, repo_id="test-repo")
    assert all(c.entropy > 0 for c in candidates)


def test_empty_findings_list_returns_empty_candidates():
    assert parse_trufflehog_report([], SAMPLE_REPO) == []


def test_value_start_end_fall_back_to_negative_one_when_value_not_found_in_line(tmp_path):
    (tmp_path / "weird.py").write_text("TOKEN = something_else\n")
    finding = {
        "id": "fake-id",
        "path": "weird.py",
        "rule": {"id": "fake-rule"},
        "context": {"1": "TOKEN = something_else"},
        "secret": "this-value-is-not-actually-on-the-line",
    }

    candidates = parse_trufflehog_report([finding], tmp_path)

    assert candidates[0].value_start == -1
    assert candidates[0].value_end == -1
