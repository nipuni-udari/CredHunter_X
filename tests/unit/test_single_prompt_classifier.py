from __future__ import annotations

import json
from pathlib import Path

from credhunter_x.classifiers.single_prompt import SinglePromptClassifier
from credhunter_x.gitleaks.parser import parse_gitleaks_report
from credhunter_x.llm.client import LLMResponse
from credhunter_x.masking.masker import build_raw_context, mask_context_window
from credhunter_x.models.classification import Label

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_REPO = FIXTURES_DIR / "sample_repo"
REPORT_PATH = FIXTURES_DIR / "gitleaks_report.json"

FAKE_RESPONSE_JSON = (
    '{"label": "true_secret", "confidence": 0.9, "severity": "high", '
    '"explanation": "looks real", "remediation": "rotate it"}'
)


class _FakeLLMClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        prompt: str,
        *,
        candidate_id: str = "",
        rule_id: str = "",
        raw_permit_candidate_id: str | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "candidate_id": candidate_id,
                "rule_id": rule_id,
                "raw_permit_candidate_id": raw_permit_candidate_id,
            }
        )
        return LLMResponse(text=FAKE_RESPONSE_JSON, input_tokens=1, output_tokens=1, latency_ms=1.0)


def load_candidates():
    findings = json.loads(REPORT_PATH.read_text())
    return parse_gitleaks_report(findings, SAMPLE_REPO, repo_id="test-repo", context_lines=10)


def test_masked_treatment_never_sets_raw_permit():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    others = [c for c in candidates if c.id != github.id]

    client = _FakeLLMClient()
    classifier = SinglePromptClassifier(client)
    context = mask_context_window(github, others)

    result = classifier.classify(github, context)

    assert client.calls[0]["raw_permit_candidate_id"] is None
    assert client.calls[0]["candidate_id"] == github.id
    assert client.calls[0]["rule_id"] == "github-pat"
    assert result.label == Label.TRUE_SECRET
    assert result.arm == "single"


def test_raw_treatment_sets_raw_permit_to_the_candidates_own_id():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")

    client = _FakeLLMClient()
    classifier = SinglePromptClassifier(client)
    context = build_raw_context(github, [])

    classifier.classify(github, context)

    assert client.calls[0]["raw_permit_candidate_id"] == github.id


def test_classification_result_treatment_matches_the_context():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")

    client = _FakeLLMClient()
    classifier = SinglePromptClassifier(client)
    context = mask_context_window(github, [])

    result = classifier.classify(github, context)

    assert result.treatment == context.treatment
