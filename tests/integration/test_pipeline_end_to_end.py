from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from credhunter_x.config.settings import Mode, ScanConfig, Settings
from credhunter_x.llm import litellm_client as litellm_client_module
from credhunter_x.models.candidate import Candidate
from credhunter_x.models.classification import Label
from credhunter_x.models.treatment import Treatment
from credhunter_x.pipeline.orchestrator import classify_candidates, scan_repository

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

    def __call__(self, *, model, api_key, messages, response_format=None, tools=None, **_kwargs):
        self.received_prompts.append(messages[0]["content"])
        return _FakeResponse(
            choices=[_FakeChoice(message=_FakeMessage(content=FAKE_RESPONSE_JSON))],
            usage=_FakeUsage(prompt_tokens=1, completion_tokens=1),
        )


def _install_fake_completion(monkeypatch: pytest.MonkeyPatch) -> _FakeCompletion:
    fake = _FakeCompletion()
    monkeypatch.setattr(litellm_client_module.litellm, "completion", fake)
    return fake


class _FakeCompletionFirstCandidateAlwaysTimesOut:
    """Reproduces the real failure found live running Arm B against
    CredData: a provider call timing out on every retry (litellm.Timeout,
    exhausting litellm_client's own retries) used to crash the whole batch
    with an unhandled LiteLLMClientError -- one flaky candidate losing
    every other candidate's already-completed work. Fails every attempt
    for the first candidate's call (all _MAX_ATTEMPTS retries), then
    succeeds for every call after -- a genuinely unrecoverable timeout,
    not one retry away from working."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, model, api_key, messages, response_format=None, tools=None, **_kwargs):
        self.calls += 1
        if self.calls <= litellm_client_module._MAX_ATTEMPTS:
            raise litellm_client_module.litellm.exceptions.Timeout(
                message="Connection timed out", model=model, llm_provider="openrouter"
            )
        return _FakeResponse(
            choices=[_FakeChoice(message=_FakeMessage(content=FAKE_RESPONSE_JSON))],
            usage=_FakeUsage(prompt_tokens=1, completion_tokens=1),
        )


class _FakeCompletionAlwaysEmptyForOneCandidate:
    """Reproduces the real failure found live in CredHunter-X's own GitHub
    Action demo: a provider returning a completely empty response body for
    one candidate (LLMParsingError: 'EOF while parsing a value'), with
    every other candidate classified normally.

    Empty on every attempt for that one candidate, so it outlives
    generate()'s empty-response retry -- this pins the skip path for output
    that is genuinely unusable, not a transient blip (see
    _FakeCompletionEmptyOnFirstCall for that)."""

    def __init__(self) -> None:
        self.calls = 0
        self._cursed_prompt: str | None = None

    def __call__(self, *, model, api_key, messages, response_format=None, tools=None, **_kwargs):
        self.calls += 1
        prompt = messages[-1]["content"]
        if self._cursed_prompt is None:
            self._cursed_prompt = prompt
        content = "" if prompt == self._cursed_prompt else FAKE_RESPONSE_JSON
        return _FakeResponse(
            choices=[_FakeChoice(message=_FakeMessage(content=content))],
            usage=_FakeUsage(prompt_tokens=1, completion_tokens=1),
        )


class _FakeCompletionEmptyOnFirstCall:
    """One empty response, then normal output -- the transient provider blip
    seen once in a 517-candidate run, where the same candidate classified
    fine on every retry."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, model, api_key, messages, response_format=None, tools=None, **_kwargs):
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


def test_scan_repository_survives_one_candidate_whose_call_times_out(
    monkeypatch: pytest.MonkeyPatch,
):
    """The real failure found live running Arm B against CredData: a
    timed-out provider call raised LiteLLMClientError, which
    skip_candidate_on_error didn't catch, crashing the whole batch. Must
    survive it the same way an empty response or a guard block do."""
    fake = _FakeCompletionFirstCandidateAlwaysTimesOut()
    monkeypatch.setattr(litellm_client_module.litellm, "completion", fake)
    monkeypatch.setattr(litellm_client_module.time, "sleep", lambda *a, **k: None)
    settings = _fake_settings()
    scan_config = ScanConfig(mode=Mode.SINGLE)

    outcome = scan_repository(SAMPLE_REPO, settings=settings, scan_config=scan_config)

    assert outcome.skipped_count == 1
    assert len(outcome.results) == 1
    assert outcome.results[0].classification.label == Label.TRUE_SECRET


def test_scan_repository_survives_one_candidate_with_an_empty_model_response(
    monkeypatch: pytest.MonkeyPatch,
):
    """The real bug found live in the GitHub Action demo: an unhandled
    LLMParsingError from one candidate crashed the whole scan with a
    traceback before any report was ever written. scan_repository() must
    survive it, still classify the other candidate, and truthfully report
    that one was skipped rather than silently returning a "clean" result."""
    fake = _FakeCompletionAlwaysEmptyForOneCandidate()
    monkeypatch.setattr(litellm_client_module.litellm, "completion", fake)
    monkeypatch.setattr(litellm_client_module.time, "sleep", lambda *a, **k: None)
    settings = _fake_settings()
    scan_config = ScanConfig(mode=Mode.SINGLE)

    outcome = scan_repository(SAMPLE_REPO, settings=settings, scan_config=scan_config)

    assert outcome.skipped_count == 1
    assert len(outcome.results) == 1
    assert outcome.results[0].classification.label == Label.TRUE_SECRET


def test_scan_repository_recovers_a_candidate_whose_response_was_empty_once(
    monkeypatch: pytest.MonkeyPatch,
):
    """A one-off empty completion used to drop the candidate permanently:
    it is a successful response, so the error-retry path never saw it, and
    it died at the parse instead. Both candidates must now survive."""
    fake = _FakeCompletionEmptyOnFirstCall()
    monkeypatch.setattr(litellm_client_module.litellm, "completion", fake)
    monkeypatch.setattr(litellm_client_module.time, "sleep", lambda *a, **k: None)
    settings = _fake_settings()
    scan_config = ScanConfig(mode=Mode.SINGLE)

    outcome = scan_repository(SAMPLE_REPO, settings=settings, scan_config=scan_config)

    assert outcome.skipped_count == 0
    assert len(outcome.results) == 2


def _make_candidate(
    *,
    id: str,
    repo_id: str,
    matched_value: str,
    line_start: int,
    context_after: list[str] | None = None,
) -> Candidate:
    return Candidate(
        id=id,
        file_path="app/fixtures.py",
        line_start=line_start,
        line_end=line_start,
        rule_id="high-entropy",
        matched_value=matched_value,
        value_start=0,
        value_end=len(matched_value),
        entropy=4.0,
        matched_lines=[f"gravatar_id = '{matched_value}'"],
        context_before=[],
        context_after=context_after or [],
        repo_id=repo_id,
        source="trufflehog",
    )


def test_another_candidates_registered_value_appearing_nearby_does_not_block_this_one(
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression test for the real bug found running trufflehog3 against
    CredData: two candidates in the same repo, each with its own distinct
    "secret" (often not a real secret at all -- e.g. a gravatar hash
    trufflehog3's high-entropy rule mistakes for one). When one candidate's
    context window happens to also contain the other's registered value,
    the guard used to block the first candidate entirely and drop it from
    the report -- even though nothing of its own was leaking. Every
    finding should reach the LLM and show up in the report on its own
    merits, regardless of what other candidates in the same repo look
    like."""
    _install_fake_completion(monkeypatch)
    settings = _fake_settings()
    scan_config = ScanConfig(mode=Mode.SINGLE, treatment=Treatment.RAW)
    own_value = "own-secret-aaaaaaaaaaaaaaaaaaaaaa"
    neighbour_value = "b68de5ae38616c296fa345d2b9df2225"
    candidates = [
        _make_candidate(
            id="c1",
            repo_id="repo-a",
            matched_value=own_value,
            line_start=10,
            # c1's own context window happens to also contain c2's value.
            context_after=[f"gravatar_id = '{neighbour_value}'"],
        ),
        _make_candidate(id="c2", repo_id="repo-a", matched_value=neighbour_value, line_start=11),
    ]

    results = classify_candidates(
        candidates,
        settings=settings,
        scan_config=scan_config,
        source_root=SAMPLE_REPO,
        skip_candidate_on_error=True,
    )

    assert {r.candidate.id for r in results} == {"c1", "c2"}
