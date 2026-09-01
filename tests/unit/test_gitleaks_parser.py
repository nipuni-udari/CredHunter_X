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

    assert github.id == "app/auth.py:github-pat:1:0"
    assert github.file_path == "app/auth.py"
    assert github.line_start == 1
    assert github.line_end == 1
    assert github.matched_value == "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"
    assert github.value_start == 16
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

    assert slack.id == "app/auth.py:slack-bot-token:2:0"
    assert slack.file_path == "app/auth.py"
    assert slack.line_start == 2
    assert slack.line_end == 2
    assert slack.matched_value == "xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"
    # gitleaks itself reports StartColumn=21, EndColumn=76 for this finding
    # — EndColumn is actually just the line's total length, not the
    # secret's real end (verified empirically). The self-derived values
    # below are the true position, found via exact string search.
    assert slack.value_start == 19
    assert slack.value_end == 75
    assert slack.matched_lines == [
        'SLACK_BOT_TOKEN = "xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"'
    ]
    assert slack.context_before == ['GITHUB_TOKEN = "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"']
    assert slack.context_after == ["", ""]


def test_empty_findings_list_returns_empty_candidates():
    assert parse_gitleaks_report([], SAMPLE_REPO) == []


def test_two_findings_sharing_a_fingerprint_get_distinct_ids(tmp_path):
    """The real bug this closes: gitleaks' decode-pass can emit a second
    finding with the same Fingerprint as the original (same file:rule:line)
    for what is still the same real secret. Without an occurrence index,
    both would get the identical id -- and classify_candidates' id-keyed
    results dict would silently drop one of the two classifications (the
    second overwrites the first)."""
    (tmp_path / "url.py").write_text('URL = "...AKIAEXAMPLE123456789..."\n')
    findings = [
        {
            "Fingerprint": "url.py:aws-access-token:1",
            "File": "url.py",
            "StartLine": 1,
            "EndLine": 1,
            "StartColumn": 8,
            "EndColumn": 28,
            "RuleID": "aws-access-token",
            "Secret": "AKIAEXAMPLE123456789",
            "Entropy": 3.5,
        },
        {
            "Fingerprint": "url.py:aws-access-token:1",
            "File": "url.py",
            "StartLine": 1,
            "EndLine": 1,
            "StartColumn": 8,
            "EndColumn": 512,
            "RuleID": "aws-access-token",
            "Secret": "AKIAEXAMPLE123456789",
            "Entropy": 3.5,
        },
    ]

    candidates = parse_gitleaks_report(findings, tmp_path)

    assert len(candidates) == 2
    ids = {c.id for c in candidates}
    assert len(ids) == 2  # the real assertion: no collision
    assert ids == {"url.py:aws-access-token:1:0", "url.py:aws-access-token:1:1"}


def test_value_start_end_fall_back_to_negative_one_when_value_not_found_in_line(tmp_path):
    (tmp_path / "weird.py").write_text("TOKEN = something_else\n")
    finding = {
        "Fingerprint": "weird.py:fake-rule:1",
        "File": "weird.py",
        "StartLine": 1,
        "EndLine": 1,
        "StartColumn": 999,
        "EndColumn": 999,
        "RuleID": "fake-rule",
        "Secret": "this-value-is-not-actually-on-the-line",
        "Entropy": 3.0,
    }

    candidates = parse_gitleaks_report([finding], tmp_path)

    assert candidates[0].value_start == -1
    assert candidates[0].value_end == -1
