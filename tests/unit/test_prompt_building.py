from __future__ import annotations

import json
from pathlib import Path

from credhunter_x.gitleaks.parser import parse_gitleaks_report
from credhunter_x.llm.prompts import build_classification_prompt
from credhunter_x.masking.masker import (
    build_metadata_only_context,
    build_pseudonymised_context,
    build_raw_context,
    mask_context_window,
)
from credhunter_x.models.candidate import Candidate

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_REPO = FIXTURES_DIR / "sample_repo"
REPORT_PATH = FIXTURES_DIR / "gitleaks_report.json"

GITHUB_SECRET = "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"


def load_candidates():
    findings = json.loads(REPORT_PATH.read_text())
    return parse_gitleaks_report(findings, SAMPLE_REPO, repo_id="test-repo", context_lines=10)


def test_raw_prompt_contains_the_real_secret():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    others = [c for c in candidates if c.id != github.id]

    context = build_raw_context(github, others)
    prompt = build_classification_prompt(github, context)

    assert GITHUB_SECRET in prompt


def test_masked_prompt_never_contains_the_real_secret_and_explains_metadata():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    others = [c for c in candidates if c.id != github.id]

    context = mask_context_window(github, others)
    prompt = build_classification_prompt(github, context)

    assert GITHUB_SECRET not in prompt
    assert "length=" in prompt
    assert "entropy=" in prompt


def test_prompt_includes_rule_id_and_file_location():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    context = build_raw_context(github, [])

    prompt = build_classification_prompt(github, context)

    assert "github-pat" in prompt
    assert github.file_path in prompt


def test_pseudonymised_prompt_never_contains_the_real_secret_or_length_signal():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    others = [c for c in candidates if c.id != github.id]

    context = build_pseudonymised_context(github, others)
    prompt = build_classification_prompt(github, context)

    assert GITHUB_SECRET not in prompt
    assert "FAKE value" in prompt
    assert "length=" in prompt


def test_metadata_only_prompt_never_contains_the_real_secret_or_a_length_matched_placeholder():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    others = [c for c in candidates if c.id != github.id]

    context = build_metadata_only_context(github, others)
    prompt = build_classification_prompt(github, context)

    assert GITHUB_SECRET not in prompt
    assert "•" * len(GITHUB_SECRET) not in prompt
    assert "[REDACTED]" in prompt
    assert "length=" in prompt


def _same_line_pair():
    """Two genuinely different secrets sharing one line -- the case that
    made line-range labelling ambiguous."""
    aws, slack = "AKIAIOSFODNN7EXAMPLE", "xoxb-1234567890-AbCdEfGh"
    line = f'creds = ("{aws}", "{slack}")'

    def make(cid, value, start, rule):
        return Candidate(
            id=cid,
            file_path="cfg.py",
            line_start=7,
            line_end=7,
            rule_id=rule,
            matched_value=value,
            value_start=start,
            value_end=start + len(value),
            entropy=4.0,
            matched_lines=[line],
            context_before=["import os", ""],
            context_after=["", "use(creds)"],
            repo_id="test-repo",
        )

    return make("a", aws, 9, "aws-access-token"), make("b", slack, 32, "slack-bot-token")


def _span_lines(prompt: str) -> list[str]:
    """Only the per-span metadata lines. The raw treatment note also uses
    the phrase "candidate under review", so counting it across the whole
    prompt would measure the note rather than the labelling."""
    return [line for line in prompt.splitlines() if line.startswith("Sanitised span")]


def test_only_one_span_is_labelled_the_candidate_under_review():
    """Labelling used to be derived from line ranges, so every secret
    sharing the target's line was announced as the candidate under review.
    The model then received several contradictory metadata blocks -- and
    under the masked treatments that block is the ONLY signal about the
    hidden value, so it had nothing reliable to judge from."""
    aws, slack = _same_line_pair()

    spans = _span_lines(build_classification_prompt(aws, mask_context_window(aws, [slack])))

    assert len(spans) == 2
    assert sum("candidate under review" in line for line in spans) == 1
    assert sum("neighbouring candidate" in line for line in spans) == 1


def test_the_review_label_follows_whichever_candidate_is_being_classified():
    """The two spans are indistinguishable by line range, so this pins that
    the label tracks candidate identity rather than position."""
    aws, slack = _same_line_pair()

    aws_spans = _span_lines(build_classification_prompt(aws, mask_context_window(aws, [slack])))
    slack_spans = _span_lines(build_classification_prompt(slack, mask_context_window(slack, [aws])))

    aws_review = next(line for line in aws_spans if "candidate under review" in line)
    slack_review = next(line for line in slack_spans if "candidate under review" in line)

    assert "aws-access-token" in aws_review
    assert "slack-bot-token" in slack_review


def test_raw_treatment_labels_every_span_as_a_neighbour():
    """Raw shows the target's own value, so it produces no span for the
    target -- every masked span present really is someone else's secret."""
    aws, slack = _same_line_pair()

    spans = _span_lines(build_classification_prompt(aws, build_raw_context(aws, [slack])))

    assert len(spans) == 1
    assert "neighbouring candidate" in spans[0]
    assert "candidate under review" not in spans[0]
