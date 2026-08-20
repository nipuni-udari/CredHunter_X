from __future__ import annotations

from dataclasses import dataclass

import litellm
import pytest

from credhunter_x.llm import litellm_client as litellm_client_module
from credhunter_x.llm.litellm_client import LiteLLMClient, LiteLLMClientError


@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction


@dataclass
class _FakeMessage:
    content: str
    tool_calls: list | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage


def _install_fake_completion(monkeypatch: pytest.MonkeyPatch, response=None, error=None):
    calls: list[dict[str, object]] = []

    def fake_completion(*, model, api_key, messages, response_format=None, tools=None):
        calls.append(
            {
                "model": model,
                "api_key": api_key,
                "messages": messages,
                "response_format": response_format,
                "tools": tools,
            }
        )
        if error:
            raise error
        return response

    monkeypatch.setattr(litellm_client_module.litellm, "completion", fake_completion)
    return calls


def _install_fake_completion_sequence(monkeypatch: pytest.MonkeyPatch, side_effects: list):
    """Each entry in side_effects is consumed in order per call: an
    exception instance is raised, anything else is returned."""
    calls: list[dict[str, object]] = []
    remaining = list(side_effects)

    def fake_completion(*, model, api_key, messages, response_format=None, tools=None):
        calls.append(
            {
                "model": model,
                "api_key": api_key,
                "messages": messages,
                "response_format": response_format,
                "tools": tools,
            }
        )
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(litellm_client_module.litellm, "completion", fake_completion)
    return calls


def test_generate_maps_content_and_token_usage(monkeypatch: pytest.MonkeyPatch):
    fake_response = _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content='{"label": "true_secret"}'))],
        usage=_FakeUsage(prompt_tokens=42, completion_tokens=17),
    )
    calls = _install_fake_completion(monkeypatch, response=fake_response)

    client = LiteLLMClient(model="gemini/gemini-flash-latest", api_key="fake-key")
    result = client.generate("classify this")

    assert result.text == '{"label": "true_secret"}'
    assert result.input_tokens == 42
    assert result.output_tokens == 17
    assert result.latency_ms >= 0
    assert calls[0]["model"] == "gemini/gemini-flash-latest"
    assert calls[0]["api_key"] == "fake-key"
    assert calls[0]["messages"] == [{"role": "user", "content": "classify this"}]


def test_generate_wraps_sdk_errors_in_a_typed_error(monkeypatch: pytest.MonkeyPatch):
    calls = _install_fake_completion(monkeypatch, error=RuntimeError("network exploded"))

    client = LiteLLMClient(model="gemini/gemini-flash-latest", api_key="fake-key")
    with pytest.raises(LiteLLMClientError):
        client.generate("classify this")

    # non-retryable errors fail immediately — no wasted retry attempts
    assert len(calls) == 1


def test_generate_retries_on_rate_limit_error_then_succeeds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm_client_module.time, "sleep", lambda _: None)
    fake_response = _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content='{"label": "true_secret"}'))],
        usage=_FakeUsage(prompt_tokens=5, completion_tokens=5),
    )
    rate_limited = litellm.RateLimitError(
        message="rate limited", llm_provider="test", model="test-model"
    )
    calls = _install_fake_completion_sequence(
        monkeypatch, [rate_limited, rate_limited, fake_response]
    )

    client = LiteLLMClient(model="gemini/gemini-flash-latest", api_key="fake-key")
    result = client.generate("classify this")

    assert result.text == '{"label": "true_secret"}'
    assert len(calls) == 3


def test_generate_retries_on_service_unavailable_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm_client_module.time, "sleep", lambda _: None)
    fake_response = _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content="{}"))],
        usage=_FakeUsage(prompt_tokens=1, completion_tokens=1),
    )
    overloaded = litellm.ServiceUnavailableError(
        message="overloaded", llm_provider="test", model="test-model"
    )
    calls = _install_fake_completion_sequence(monkeypatch, [overloaded, fake_response])

    client = LiteLLMClient(model="gemini/gemini-flash-latest", api_key="fake-key")
    result = client.generate("classify this")

    assert result.text == "{}"
    assert len(calls) == 2


def test_generate_raises_after_exhausting_all_retry_attempts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm_client_module.time, "sleep", lambda _: None)
    rate_limited = litellm.RateLimitError(
        message="rate limited", llm_provider="test", model="test-model"
    )
    calls = _install_fake_completion_sequence(monkeypatch, [rate_limited] * 5)

    client = LiteLLMClient(model="gemini/gemini-flash-latest", api_key="fake-key")
    with pytest.raises(LiteLLMClientError):
        client.generate("classify this")

    assert len(calls) == 5


def test_generate_ignores_guard_only_kwargs(monkeypatch: pytest.MonkeyPatch):
    fake_response = _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content="{}"))],
        usage=_FakeUsage(prompt_tokens=1, completion_tokens=1),
    )
    _install_fake_completion(monkeypatch, response=fake_response)

    client = LiteLLMClient(model="gemini/gemini-flash-latest", api_key="fake-key")
    result = client.generate(
        "x", candidate_id="c1", rule_id="github-pat", raw_permit_candidate_id="c1"
    )

    assert result.text == "{}"


def test_constructor_raises_when_model_is_missing():
    with pytest.raises(LiteLLMClientError):
        LiteLLMClient(model="", api_key="fake-key")


def test_constructor_raises_when_api_key_is_missing():
    with pytest.raises(LiteLLMClientError):
        LiteLLMClient(model="gemini/gemini-flash-latest", api_key="")


def test_generate_with_tools_maps_final_text_answer_when_no_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_response = _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content='{"label": "false_positive"}'))],
        usage=_FakeUsage(prompt_tokens=10, completion_tokens=4),
    )
    calls = _install_fake_completion(monkeypatch, response=fake_response)

    client = LiteLLMClient(model="groq/openai/gpt-oss-120b", api_key="fake-key")
    messages = [{"role": "user", "content": "classify this"}]
    result = client.generate_with_tools(messages, tools=[{"type": "function"}])

    assert result.text == '{"label": "false_positive"}'
    assert result.tool_calls == []
    assert result.input_tokens == 10
    assert result.output_tokens == 4
    # no response_format is sent alongside tools -- see litellm_client.py's
    # generate_with_tools docstring for why
    assert calls[0]["response_format"] is None
    assert calls[0]["tools"] == [{"type": "function"}]


def test_generate_with_tools_parses_tool_calls_and_their_json_arguments(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_response = _FakeResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    content="",
                    tool_calls=[
                        _FakeToolCall(
                            id="call_1",
                            function=_FakeFunction(
                                name="search_file",
                                arguments='{"file_path": "app.py", "query": "TOKEN"}',
                            ),
                        )
                    ],
                )
            )
        ],
        usage=_FakeUsage(prompt_tokens=20, completion_tokens=8),
    )
    _install_fake_completion(monkeypatch, response=fake_response)

    client = LiteLLMClient(model="groq/openai/gpt-oss-120b", api_key="fake-key")
    result = client.generate_with_tools(
        [{"role": "user", "content": "x"}], tools=[{"type": "function"}]
    )

    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "search_file"
    assert call.arguments == {"file_path": "app.py", "query": "TOKEN"}


def test_generate_with_tools_omits_tools_kwarg_when_tools_list_is_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_response = _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content="{}"))],
        usage=_FakeUsage(prompt_tokens=1, completion_tokens=1),
    )
    calls = _install_fake_completion(monkeypatch, response=fake_response)

    client = LiteLLMClient(model="groq/openai/gpt-oss-120b", api_key="fake-key")
    client.generate_with_tools([{"role": "user", "content": "x"}], tools=[])

    # empty tools -> no tools kwarg reaches litellm at all, forcing a
    # plain-text answer instead of offering (zero) functions to call
    assert calls[0]["tools"] is None


def test_generate_with_tools_retries_on_rate_limit_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm_client_module.time, "sleep", lambda _: None)
    fake_response = _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content="{}"))],
        usage=_FakeUsage(prompt_tokens=1, completion_tokens=1),
    )
    rate_limited = litellm.RateLimitError(
        message="rate limited", llm_provider="test", model="test-model"
    )
    calls = _install_fake_completion_sequence(monkeypatch, [rate_limited, fake_response])

    client = LiteLLMClient(model="groq/openai/gpt-oss-120b", api_key="fake-key")
    result = client.generate_with_tools([{"role": "user", "content": "x"}], tools=[])

    assert result.text == "{}"
    assert len(calls) == 2
