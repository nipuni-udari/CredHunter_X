from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from credhunter_x.config.settings import Mode, ScanConfig, Settings
from credhunter_x.llm import litellm_client as litellm_client_module
from credhunter_x.models.classification import Label
from credhunter_x.models.treatment import Treatment
from credhunter_x.pipeline.orchestrator import scan_repository

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_REPO = FIXTURES_DIR / "sample_repo"

pytestmark = pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks binary not on PATH"
)

FAKE_RESPONSE_JSON = (
    '{"label": "true_secret", "confidence": 0.9, "severity": "high", '
    '"explanation": "looks real", "remediation": "rotate it"}'
)

GITHUB_SECRET = "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"
SLACK_SECRET = "xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"


@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage


class _FakeCompletion:
    def __init__(self) -> None:
        self.received_prompts: list[str] = []

    def __call__(self, *, model, api_key, messages, response_format=None, tools=None):
        self.received_prompts.append(messages[0]["content"])
        return _FakeResponse(
            choices=[_FakeChoice(message=_FakeMessage(content=FAKE_RESPONSE_JSON))],
            usage=_FakeUsage(prompt_tokens=1, completion_tokens=1),
        )


def _install_fake_completion(monkeypatch: pytest.MonkeyPatch) -> _FakeCompletion:
    fake = _FakeCompletion()
    monkeypatch.setattr(litellm_client_module.litellm, "completion", fake)
    return fake


class _FakeCompletionEmptyOnFirstCall:
    """Reproduces the real failure found live in CredHunter-X's own GitHub
    Action demo: a provider returning a completely empty response body for
    one candidate (LLMParsingError: 'EOF while parsing a value'), with
    every other candidate classified normally."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, model, api_key, messages, response_format=None, tools=None):
        self.calls += 1
        content = "" if self.calls == 1 else FAKE_RESPONSE_JSON
        return _FakeResponse(
            choices=[_FakeChoice(message=_FakeMessage(content=content))],
            usage=_FakeUsage(prompt_tokens=1, completion_tokens=1),
        )


def _fake_settings(**overrides) -> Settings:
    return Settings(llm_model="gemini/gemini-flash-latest", llm_api_key="fake-key", **overrides)


def test_scan_repository_classifies_both_fixture_secrets(monkeypatch: pytest.MonkeyPatch):
    _install_fake_completion(monkeypatch)
    settings = _fake_settings()
    scan_config = ScanConfig(mode=Mode.SINGLE)  # single mode, masked treatment

    outcome = scan_repository(SAMPLE_REPO, settings=settings, scan_config=scan_config)

    assert {r.candidate.rule_id for r in outcome.results} == {"github-pat", "slack-bot-token"}
    assert all(r.classification.label == Label.TRUE_SECRET for r in outcome.results)
    assert outcome.skipped_count == 0


def test_scan_repository_masked_treatment_never_sends_raw_secrets(monkeypatch: pytest.MonkeyPatch):
    fake = _install_fake_completion(monkeypatch)
    settings = _fake_settings()
    scan_config = ScanConfig(mode=Mode.SINGLE)

    scan_repository(SAMPLE_REPO, settings=settings, scan_config=scan_config)

    for sent in fake.received_prompts:
        assert GITHUB_SECRET not in sent
        assert SLACK_SECRET not in sent


def test_scan_repository_raw_treatment_sends_the_real_secret_for_its_own_candidate(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _install_fake_completion(monkeypatch)
    settings = _fake_settings()
    scan_config = ScanConfig(mode=Mode.SINGLE, treatment=Treatment.RAW)

    scan_repository(SAMPLE_REPO, settings=settings, scan_config=scan_config)

    assert any(GITHUB_SECRET in sent for sent in fake.received_prompts)


def test_scan_repository_returns_empty_list_for_a_clean_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _install_fake_completion(monkeypatch)
    (tmp_path / "clean.py").write_text("x = 1\n")
    settings = _fake_settings()
    scan_config = ScanConfig(mode=Mode.SINGLE)

    outcome = scan_repository(tmp_path, settings=settings, scan_config=scan_config)
    assert outcome.results == []
    assert outcome.skipped_count == 0


def test_scan_repository_agentic_mode_classifies_without_calling_a_tool(
    monkeypatch: pytest.MonkeyPatch,
):
    # The fake always answers immediately (no tool_calls), so this exercises
    # the "model didn't need a tool" path end-to-end through scan_repository.
    _install_fake_completion(monkeypatch)
    settings = _fake_settings()
    scan_config = ScanConfig(mode=Mode.AGENTIC)

    outcome = scan_repository(SAMPLE_REPO, settings=settings, scan_config=scan_config)

    assert {r.candidate.rule_id for r in outcome.results} == {"github-pat", "slack-bot-token"}
    assert all(r.classification.label == Label.TRUE_SECRET for r in outcome.results)
    assert all(r.classification.arm == "agentic" for r in outcome.results)
    assert outcome.skipped_count == 0


def test_scan_repository_pseudonymised_treatment_classifies_and_never_sends_raw_secrets(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _install_fake_completion(monkeypatch)
    settings = _fake_settings()
    scan_config = ScanConfig(mode=Mode.SINGLE, treatment=Treatment.PSEUDONYMISED)

    outcome = scan_repository(SAMPLE_REPO, settings=settings, scan_config=scan_config)

    assert {r.candidate.rule_id for r in outcome.results} == {"github-pat", "slack-bot-token"}
    assert all(r.classification.label == Label.TRUE_SECRET for r in outcome.results)
    assert outcome.skipped_count == 0
    for sent in fake.received_prompts:
        assert GITHUB_SECRET not in sent
        assert SLACK_SECRET not in sent


def test_scan_repository_metadata_only_never_sends_a_real_or_length_matched_value(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _install_fake_completion(monkeypatch)
    settings = _fake_settings()
    scan_config = ScanConfig(mode=Mode.SINGLE, treatment=Treatment.METADATA_ONLY)

    outcome = scan_repository(SAMPLE_REPO, settings=settings, scan_config=scan_config)

    assert {r.candidate.rule_id for r in outcome.results} == {"github-pat", "slack-bot-token"}
    assert all(r.classification.label == Label.TRUE_SECRET for r in outcome.results)
    assert outcome.skipped_count == 0
    for sent in fake.received_prompts:
        assert GITHUB_SECRET not in sent
        assert SLACK_SECRET not in sent
        assert "•" * len(GITHUB_SECRET) not in sent
        assert "•" * len(SLACK_SECRET) not in sent


def test_scan_repository_survives_one_candidate_with_an_empty_model_response(
    monkeypatch: pytest.MonkeyPatch,
):
    """The real bug found live in the GitHub Action demo: an unhandled
    LLMParsingError from one candidate crashed the whole scan with a
    traceback before any report was ever written. scan_repository() must
    survive it, still classify the other candidate, and truthfully report
    that one was skipped rather than silently returning a "clean" result."""
    fake = _FakeCompletionEmptyOnFirstCall()
    monkeypatch.setattr(litellm_client_module.litellm, "completion", fake)
    settings = _fake_settings()
    scan_config = ScanConfig(mode=Mode.SINGLE)

    outcome = scan_repository(SAMPLE_REPO, settings=settings, scan_config=scan_config)

    assert outcome.skipped_count == 1
    assert len(outcome.results) == 1
    assert outcome.results[0].classification.label == Label.TRUE_SECRET
