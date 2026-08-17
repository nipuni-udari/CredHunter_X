from __future__ import annotations

import json
from pathlib import Path

from credhunter_x.gitleaks.parser import parse_gitleaks_report

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_REPO = FIXTURES_DIR / "sample_repo"
REPORT_PATH = FIXTURES_DIR / "gitleaks_report.json"


def load_findings() -> list[dict]:
    return json.loads(REPORT_PATH.read_text())


def test_parses_all_findings_into_candidates():
    candidates = parse_gitleaks_report(load_findings(), SAMPLE_REPO, repo_id="test-repo")
    assert len(candidates) == 2


def test_maps_fields_correctly_for_github_token_finding():
    candidates = parse_gitleaks_report(
        load_findings(), SAMPLE_REPO, repo_id="test-repo", context_lines=2
    )
    github = next(c for c in candidates if c.rule_id == "github-pat")

    assert github.id == "app/auth.py:github-pat:1"
    assert github.file_path == "app/auth.py"
    assert github.line_start == 1
    assert github.line_end == 1
    assert github.matched_value == "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"
    assert github.value_start == 17
    assert github.value_end == 56
    assert github.entropy == 4.6841836
    assert github.repo_id == "test-repo"
    assert github.matched_lines == ['GITHUB_TOKEN = "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"']
    assert github.context_before == []
    assert github.context_after == [
        'SLACK_BOT_TOKEN = "xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"',
        "",
    ]


def test_maps_fields_correctly_for_slack_token_finding():
    candidates = parse_gitleaks_report(
        load_findings(), SAMPLE_REPO, repo_id="test-repo", context_lines=2
    )
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    assert slack.id == "app/auth.py:slack-bot-token:2"
    assert slack.file_path == "app/auth.py"
    assert slack.line_start == 2
    assert slack.line_end == 2
    assert slack.matched_value == "xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"
    assert slack.matched_lines == [
        'SLACK_BOT_TOKEN = "xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"'
    ]
    assert slack.context_before == ['GITHUB_TOKEN = "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"']
    assert slack.context_after == ["", ""]


def test_empty_findings_list_returns_empty_candidates():
    assert parse_gitleaks_report([], SAMPLE_REPO) == []
