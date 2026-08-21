from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from credhunter_x.gitleaks.parser import parse_gitleaks_report
from credhunter_x.masking.masker import (
    build_metadata_only_context,
    build_pseudonymised_context,
    build_raw_context,
    mask_arbitrary_text,
    mask_context_window,
    mask_value,
    pseudonymise_value,
    redact_value,
)
from credhunter_x.models.candidate import Candidate
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


def test_mask_arbitrary_text_masks_a_truncated_snippet_of_a_multiline_secret():
    """The real bug this rewrite fixes: Arm B's search_file tool only ever
    returns a small +/-2-line snippet, which for a multi-line PEM key never
    contains the *entire* matched_value -- a whole-value-only replace would
    never fire, silently leaving real key bytes in a tool result. This
    snippet is an interior line only (no BEGIN/END markers, exactly what
    search_file would return), well short of the full key."""
    private_key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIICWgIBAAKBgQDTj1bqB4WmayWNPB+8jVSYpZYk80Ujvj680pOTh2bORBjbIAyz\n"
        "iiqU+dqBDBxBtiWQZVywgz5TcZcHG95k17GHGlpMIEortIRm22f23U/Hd6VE+7s/\n"
        "-----END RSA PRIVATE KEY-----"
    )
    key_line = "iiqU+dqBDBxBtiWQZVywgz5TcZcHG95k17GHGlpMIEortIRm22f23U/Hd6VE+7s/"
    candidate = Candidate(
        id="c1",
        file_path="app/keys.py",
        line_start=5,
        line_end=8,
        rule_id="private-key",
        matched_value=private_key,
        value_start=-1,
        value_end=-1,
        entropy=4.0,
        matched_lines=private_key.split("\n"),
        context_before=[],
        context_after=[],
        repo_id="test-repo",
    )

    snippet = f"63: {key_line}\n64: -----END RSA PRIVATE KEY-----"
    assert private_key not in snippet  # confirms this is genuinely partial

    result = mask_arbitrary_text(snippet, [candidate])

    assert key_line not in result
    assert "•" in result


def test_mask_arbitrary_text_merges_overlapping_fragment_matches_fully():
    """A leaked run longer than one FRAGMENT_LEN window must be blanked out
    completely, not just its first FRAGMENT_LEN characters -- an early
    implementation attempt (replace-one-fragment-and-stop) would have left
    a tail of real secret characters exposed."""
    candidates = load_candidates()
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    text = f"before {SLACK_SECRET} after"
    result = mask_arbitrary_text(text, [slack])

    assert SLACK_SECRET not in result
    assert SLACK_SECRET[-4:] not in result
    assert "before" in result
    assert "after" in result


def test_pseudonymise_value_replaces_with_a_same_shape_fake():
    placeholder, metadata = pseudonymise_value(GITHUB_SECRET, type_hint="github-pat")
    assert placeholder != GITHUB_SECRET
    assert placeholder.startswith("ghp_")
    assert metadata.length == len(GITHUB_SECRET)
    assert metadata.prefix_hint == "github-pat"


def test_redact_value_uses_a_fixed_marker_not_a_length_matched_one():
    placeholder, metadata = redact_value(GITHUB_SECRET, type_hint="github-pat")
    assert placeholder == "[REDACTED]"
    assert placeholder != "•" * len(GITHUB_SECRET)
    assert metadata.length == len(GITHUB_SECRET)  # length still recorded in metadata


def test_build_pseudonymised_context_masks_target_and_bystanders():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    result = build_pseudonymised_context(github, [slack])

    assert result.treatment == Treatment.PSEUDONYMISED
    assert GITHUB_SECRET not in result.sanitised_snippet
    assert SLACK_SECRET not in result.sanitised_snippet
    assert len(result.masked_spans) == 2


def test_build_pseudonymised_context_injects_a_recognisable_fake_shape():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")

    result = build_pseudonymised_context(github, [])

    assert "ghp_" in result.sanitised_snippet


def test_build_pseudonymised_context_falls_back_to_bullet_masking_for_multiline_secrets():
    """The documented limitation: a multi-line matched_value (private keys)
    can never match via exact single-line substring replace, so this
    silently falls back to the same bullet-masking mask_context_window
    uses, rather than attempting realistic multi-line PEM synthesis."""
    multiline_key = "-----BEGIN RSA PRIVATE KEY-----\nMIIB\n-----END RSA PRIVATE KEY-----"
    candidate = Candidate(
        id="c1",
        file_path="app/keys.py",
        line_start=5,
        line_end=7,
        rule_id="private-key",
        matched_value=multiline_key,
        value_start=-1,
        value_end=-1,
        entropy=4.0,
        matched_lines=multiline_key.split("\n"),
        context_before=["KEY = '''"],
        context_after=["'''"],
        repo_id="test-repo",
    )

    result = build_pseudonymised_context(candidate, [])

    assert "•" in result.sanitised_snippet
    assert "MIIB" not in result.sanitised_snippet


def test_build_metadata_only_context_masks_target_and_bystanders():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    result = build_metadata_only_context(github, [slack])

    assert result.treatment == Treatment.METADATA_ONLY
    assert GITHUB_SECRET not in result.sanitised_snippet
    assert SLACK_SECRET not in result.sanitised_snippet
    assert len(result.masked_spans) == 2


def test_build_metadata_only_context_reveals_no_length_signal():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")

    result = build_metadata_only_context(github, [])

    assert "•" * len(GITHUB_SECRET) not in result.sanitised_snippet
    assert "[REDACTED]" in result.sanitised_snippet
