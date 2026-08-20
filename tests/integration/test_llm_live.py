from __future__ import annotations

import json
from pathlib import Path

import pytest

from credhunter_x.config.settings import Settings
from credhunter_x.gitleaks.parser import parse_gitleaks_report
from credhunter_x.guard.leak_guard import LeakGuard
from credhunter_x.llm.guarded_client import GuardedLLMClient
from credhunter_x.llm.litellm_client import LiteLLMClient
from credhunter_x.llm.parsing import parse_classification
from credhunter_x.llm.prompts import build_classification_prompt
from credhunter_x.masking.masker import mask_context_window
from credhunter_x.masking.secret_registry import SecretRegistry
from credhunter_x.models.classification import Label
from credhunter_x.models.treatment import Treatment

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_REPO = FIXTURES_DIR / "sample_repo"
REPORT_PATH = FIXTURES_DIR / "gitleaks_report.json"


@pytest.mark.live
def test_guarded_llm_client_classifies_a_real_secret_end_to_end():
    """Manual, opt-in sanity check against whichever real provider is
    configured in .env (excluded from default CI via the "live" marker).
    Proves the whole chain works together: masking -> prompt -> guarded
    call -> parsing, regardless of which LLM provider is behind it."""
    settings = Settings()
    findings = json.loads(REPORT_PATH.read_text())
    candidates = parse_gitleaks_report(findings, SAMPLE_REPO, repo_id="test-repo", context_lines=10)
    github = next(c for c in candidates if c.rule_id == "github-pat")
    others = [c for c in candidates if c.id != github.id]

    registry = SecretRegistry()
    registry.register_candidates(candidates)
    guard = LeakGuard(registry)
    llm_client = LiteLLMClient(model=settings.llm_model, api_key=settings.llm_api_key)
    client = GuardedLLMClient(llm_client, guard)

    context = mask_context_window(github, others)
    prompt = build_classification_prompt(github, context)

    response = client.generate(prompt, candidate_id=github.id, rule_id=github.rule_id)
    result = parse_classification(
        response, candidate_id=github.id, arm="single", treatment=Treatment.MASKED
    )

    assert result.label in {Label.TRUE_SECRET, Label.FALSE_POSITIVE, Label.UNCERTAIN}
    assert result.raw_model_output == response.text
