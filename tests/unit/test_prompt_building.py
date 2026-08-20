from __future__ import annotations

import json
from pathlib import Path

from credhunter_x.gitleaks.parser import parse_gitleaks_report
from credhunter_x.llm.prompts import build_classification_prompt
from credhunter_x.masking.masker import build_raw_context, mask_context_window

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
