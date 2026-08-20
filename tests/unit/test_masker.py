from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from credhunter_x.gitleaks.parser import parse_gitleaks_report
from credhunter_x.masking.masker import (
    build_raw_context,
    mask_arbitrary_text,
    mask_context_window,
    mask_value,
)
from credhunter_x.models.treatment import Treatment

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_REPO = FIXTURES_DIR / "sample_repo"
REPORT_PATH = FIXTURES_DIR / "gitleaks_report.json"

GITHUB_SECRET = "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"
SLACK_SECRET = "xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"


def load_candidates(context_lines: int = 10):
    findings = json.loads(REPORT_PATH.read_text())
    return parse_gitleaks_report(
        findings, SAMPLE_REPO, repo_id="test-repo", context_lines=context_lines
    )


def test_mask_value_never_reveals_real_characters():
    placeholder, metadata = mask_value(GITHUB_SECRET, type_hint="github-pat")
    assert len(placeholder) == len(GITHUB_SECRET)
    assert set(placeholder) == {"•"}
    assert metadata.length == len(GITHUB_SECRET)
    assert metadata.prefix_hint == "github-pat"
    assert metadata.entropy > 0


def test_mask_value_hint_is_never_sliced_from_the_real_value():
    value = "S0meR4ndomP4ssw0rd!2024xyz"
    _, metadata = mask_value(value, type_hint="generic-password")
    assert metadata.prefix_hint == "generic-password"
    assert value[:4] not in metadata.prefix_hint


def test_mask_context_window_masks_targets_own_secret():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    others = [c for c in candidates if c.id != github.id]

    result = mask_context_window(github, others)

    assert result.treatment == Treatment.MASKED
    assert GITHUB_SECRET not in result.sanitised_snippet
    assert "•" in result.sanitised_snippet


def test_mask_context_window_also_masks_other_secrets_in_the_same_window():
    """The specific bug the design calls out: a naive implementation only
    masks the flagged candidate's own line and leaves other secrets in the
    surrounding context exposed. Both real secrets here are on adjacent
    lines, well within the default context window."""
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    result = mask_context_window(github, [slack])

    assert GITHUB_SECRET not in result.sanitised_snippet
    assert SLACK_SECRET not in result.sanitised_snippet
    assert len(result.masked_spans) == 2


def test_mask_context_window_excludes_candidates_outside_the_window():
    candidates = load_candidates(context_lines=0)
    github = next(c for c in candidates if c.rule_id == "github-pat")
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    # context_lines=0: github's own window is just its own line, so slack's
    # line (the very next one) never enters the window at all
    result = mask_context_window(github, [slack])

    assert len(result.masked_spans) == 1
    assert result.masked_spans[0].start == github.line_start


def test_mask_context_window_ignores_candidates_from_a_different_file():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")
    unrelated = replace(slack, file_path="app/other.py")

    result = mask_context_window(github, [unrelated])
    assert len(result.masked_spans) == 1


def test_build_raw_context_leaves_target_secret_unmasked():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")

    result = build_raw_context(github, [])

    assert result.treatment == Treatment.RAW
    assert GITHUB_SECRET in result.sanitised_snippet
    assert result.masked_spans == []


def test_build_raw_context_still_masks_other_secrets_in_the_same_window():
    """RAW treatment means "reveal this one candidate's real value," not
    "expose every secret nearby" — a bystander secret sharing the window
    (here, Slack's token sits one line below GitHub's) must stay masked
    even when the target itself is sent raw."""
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    result = build_raw_context(github, [slack])

    assert GITHUB_SECRET in result.sanitised_snippet
    assert SLACK_SECRET not in result.sanitised_snippet
    assert len(result.masked_spans) == 1
    assert result.masked_spans[0].start == slack.line_start


def test_mask_arbitrary_text_masks_a_known_secret_found_anywhere():
    """Unlike the window-based masking above, this is for Arm B's tool
    results -- text that can come from anywhere in the repo, not just a
    candidate's own precomputed window."""
    candidates = load_candidates()
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    text = f"some unrelated file content...\n{SLACK_SECRET}\n...more content"
    result = mask_arbitrary_text(text, [slack])

    assert SLACK_SECRET not in result
    assert "some unrelated file content" in result
    assert "more content" in result


def test_mask_arbitrary_text_masks_multiple_occurrences_and_candidates():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    text = f"{GITHUB_SECRET} appears twice: {GITHUB_SECRET}\nand also {SLACK_SECRET}"
    result = mask_arbitrary_text(text, [github, slack])

    assert GITHUB_SECRET not in result
    assert SLACK_SECRET not in result


def test_mask_arbitrary_text_leaves_unrelated_text_untouched_when_no_secret_present():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")

    text = "just some ordinary code with no secrets in it"
    assert mask_arbitrary_text(text, [github]) == text


def test_mask_arbitrary_text_handles_an_empty_candidate_list():
    text = f"contains {GITHUB_SECRET} but nothing is registered"
    assert mask_arbitrary_text(text, []) == text
