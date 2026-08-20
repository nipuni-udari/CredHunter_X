from __future__ import annotations

import pytest

from credhunter_x.guard.errors import LeakError
from credhunter_x.guard.leak_guard import LeakGuard
from credhunter_x.llm.client import LLMResponse
from credhunter_x.llm.guarded_client import GuardedLLMClient
from credhunter_x.masking.secret_registry import SecretRegistry

GITHUB_SECRET = "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"


class _FakeLLMClient:
    def __init__(self, response: LLMResponse) -> None:
        self._response = response
        self.received_prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        candidate_id: str = "",
        rule_id: str = "",
        raw_permit_candidate_id: str | None = None,
    ) -> LLMResponse:
        self.received_prompts.append(prompt)
        return self._response


def make_guard(secrets: dict[str, str]) -> LeakGuard:
    registry = SecretRegistry()
    for candidate_id, value in secrets.items():
        registry.register(candidate_id, value)
    return LeakGuard(registry)


def test_clean_payload_reaches_the_inner_client():
    inner = _FakeLLMClient(LLMResponse(text="{}", input_tokens=1, output_tokens=1, latency_ms=1.0))
    guard = make_guard({"c1": GITHUB_SECRET})
    client = GuardedLLMClient(inner, guard)

    result = client.generate("a clean masked payload with no secrets", candidate_id="c1")

    assert result.text == "{}"
    assert inner.received_prompts == ["a clean masked payload with no secrets"]


def test_leaking_payload_never_reaches_the_inner_client():
    inner = _FakeLLMClient(LLMResponse(text="{}", input_tokens=1, output_tokens=1, latency_ms=1.0))
    guard = make_guard({"c1": GITHUB_SECRET})
    client = GuardedLLMClient(inner, guard)

    with pytest.raises(LeakError):
        client.generate(f"leaking: {GITHUB_SECRET}", candidate_id="c1")

    assert inner.received_prompts == []


def test_raw_permit_allows_the_named_candidates_own_value_through():
    inner = _FakeLLMClient(LLMResponse(text="{}", input_tokens=1, output_tokens=1, latency_ms=1.0))
    guard = make_guard({"c1": GITHUB_SECRET})
    client = GuardedLLMClient(inner, guard)

    client.generate(
        f"sending real value: {GITHUB_SECRET}", candidate_id="c1", raw_permit_candidate_id="c1"
    )

    assert inner.received_prompts == [f"sending real value: {GITHUB_SECRET}"]
