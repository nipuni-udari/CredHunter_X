from __future__ import annotations

import json
import time
from typing import Any, cast

import litellm
from litellm.exceptions import RateLimitError, ServiceUnavailableError, Timeout
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from credhunter_x.llm.client import LLMResponse, LLMToolResponse, ToolCall
from credhunter_x.llm.schema import ClassificationSchema

# Turns off litellm's "Provider List" debug print for openrouter models,
# using litellm's documented flag.
litellm.suppress_debug_info = True

_MAX_ATTEMPTS = 5
_BASE_DELAY_SECONDS = 1.0
_MAX_DELAY_SECONDS = 30.0
_RETRYABLE_ERRORS = (RateLimitError, ServiceUnavailableError, Timeout)
# Retries for a successful but empty response (see generate()). Lower
# than _MAX_ATTEMPTS because the two loops nest.
_MAX_EMPTY_ATTEMPTS = 3
# Answers are short, so cap the output. Uncapped, litellm requests the
# model's maximum, which some providers refuse on a small balance.
_MAX_OUTPUT_TOKENS = 2048


class LiteLLMClientError(RuntimeError):
    pass


class LiteLLMClient:
    """Thin wrapper around litellm.completion(), the only LLM adapter in the
    project, so changing provider is a config change. Always wrap it in
    GuardedLLMClient.

    Rate-limit, unavailable and timeout errors are retried with backoff up to
    _MAX_ATTEMPTS; other errors fail at once. reasoning_effort is passed on when
    set and ignored by models that don't support it."""

    def __init__(self, model: str, api_key: str, reasoning_effort: str | None = None) -> None:
        if not model or not api_key:
            raise LiteLLMClientError("LLM_MODEL and LLM_API_KEY must both be set")
        self._model = model
        self._api_key = api_key
        self._reasoning_effort = reasoning_effort

    def generate(
        self,
        prompt: str,
        *,
        candidate_id: str = "",
        rule_id: str = "",
        raw_permit_candidate_id: str | None = None,
        response_schema: type[BaseModel] = ClassificationSchema,
    ) -> LLMResponse:
        # guard-only metadata, irrelevant to the raw API call
        del candidate_id, rule_id, raw_permit_candidate_id
        start = time.monotonic()
        # An empty reply is a normal response, so _call_with_retry doesn't catch
        # it. Retry it here so the candidate isn't lost.
        response: ModelResponse | None = None
        text = ""
        for attempt in range(_MAX_EMPTY_ATTEMPTS):
            response = self._call_with_retry(
                messages=[{"role": "user", "content": prompt}],
                response_format=response_schema,
            )
            text = response.choices[0].message.content or ""
            if text.strip():
                break
            if attempt < _MAX_EMPTY_ATTEMPTS - 1:
                time.sleep(min(_BASE_DELAY_SECONDS * 2**attempt, _MAX_DELAY_SECONDS))
        latency_ms = (time.monotonic() - start) * 1000

        # mypy can't see `usage` via normal attribute access on ModelResponse
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            input_tokens=(usage.prompt_tokens or 0) if usage else 0,
            output_tokens=(usage.completion_tokens or 0) if usage else 0,
            latency_ms=latency_ms,
        )

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        candidate_id: str = "",
        rule_id: str = "",
        raw_permit_candidate_id: str | None = None,
    ) -> LLMToolResponse:
        # guard-only metadata, irrelevant to the raw API call
        del candidate_id, rule_id, raw_permit_candidate_id
        start = time.monotonic()
        # No response_format: a forced JSON schema with tool-calling isn't reliable
        # across providers, so the prompt asks for the JSON in plain words.
        call_kwargs: dict[str, Any] = {"messages": messages}
        if tools:
            call_kwargs["tools"] = tools
        response = self._call_with_retry(**call_kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        usage = getattr(response, "usage", None)
        message = response.choices[0].message
        tool_calls = []
        for tc in getattr(message, "tool_calls", None) or []:
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))

        return LLMToolResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            input_tokens=(usage.prompt_tokens or 0) if usage else 0,
            output_tokens=(usage.completion_tokens or 0) if usage else 0,
            latency_ms=latency_ms,
        )

    def _call_with_retry(self, **kwargs: Any) -> ModelResponse:
        kwargs.setdefault("max_tokens", _MAX_OUTPUT_TOKENS)
        if self._reasoning_effort:
            kwargs.setdefault("reasoning_effort", self._reasoning_effort)
        delay = _BASE_DELAY_SECONDS
        last_error: Exception = LiteLLMClientError("unreachable")
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = litellm.completion(model=self._model, api_key=self._api_key, **kwargs)
                return cast(ModelResponse, response)
            except _RETRYABLE_ERRORS as exc:
                last_error = exc
                if attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(min(delay, _MAX_DELAY_SECONDS))
                    delay *= 2
            except Exception as exc:
                raise LiteLLMClientError(f"LLM API call failed: {exc}") from exc
        raise LiteLLMClientError(
            f"LLM API call failed after {_MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error
