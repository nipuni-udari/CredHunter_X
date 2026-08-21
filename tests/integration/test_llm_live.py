from __future__ import annotations

import json
from pathlib import Path

import pytest

from credhunter_x.config.settings import Settings
from credhunter_x.evaluation.remediation_scoring import score_remediation
from credhunter_x.gitleaks.parser import parse_gitleaks_report
from credhunter_x.guard.leak_guard import LeakGuard
from credhunter_x.llm.guarded_client import GuardedLLMClient
from credhunter_x.llm.litellm_client import LiteLLMClient
from credhunter_x.llm.parsing import parse_classification
from credhunter_x.llm.prompts import build_classification_prompt
from credhunter_x.masking.masker import (
    build_metadata_only_context,
    build_pseudonymised_context,
    mask_context_window,
)
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


@pytest.mark.live
def test_guarded_llm_client_classifies_a_real_secret_end_to_end_pseudonymised():
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

    context = build_pseudonymised_context(github, others)
    prompt = build_classification_prompt(github, context)

    response = client.generate(prompt, candidate_id=github.id, rule_id=github.rule_id)
    result = parse_classification(
        response, candidate_id=github.id, arm="single", treatment=Treatment.PSEUDONYMISED
    )

    assert result.label in {Label.TRUE_SECRET, Label.FALSE_POSITIVE, Label.UNCERTAIN}


@pytest.mark.live
def test_guarded_llm_client_classifies_a_real_secret_end_to_end_metadata_only():
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

    context = build_metadata_only_context(github, others)
    prompt = build_classification_prompt(github, context)

    response = client.generate(prompt, candidate_id=github.id, rule_id=github.rule_id)
    result = parse_classification(
        response, candidate_id=github.id, arm="single", treatment=Treatment.METADATA_ONLY
    )

    assert result.label in {Label.TRUE_SECRET, Label.FALSE_POSITIVE, Label.UNCERTAIN}


@pytest.mark.live
def test_guarded_llm_client_scores_remediation_elements_end_to_end():
    """The highest-risk test in the RQ3 build: proves the provider's
    structured-output mode actually honors a SECOND, non-default Pydantic
    schema (ElementCheckSchema) the way ClassificationSchema is honored
    today. If this fails, the schema design itself needs revisiting
    before any more RQ3 code is built on top of it."""
    settings = Settings()
    registry = SecretRegistry()
    registry.register("dummy", "not-a-real-secret-just-for-guard-population")
    guard = LeakGuard(registry)
    llm_client = LiteLLMClient(model=settings.llm_model, api_key=settings.llm_api_key)
    client = GuardedLLMClient(llm_client, guard)

    required_elements = ["rotate the exposed key", "use an environment variable"]
    result = score_remediation(
        client,
        candidate_id="c1",
        rule_id="aws-access-token",
        remediation="Rotate the AWS access key immediately and store the new one in an "
        "environment variable instead of hardcoding it.",
        required_elements=required_elements,
    )

    assert len(result.element_present) == len(required_elements)
    assert 0.0 <= result.pass_rate <= 1.0
