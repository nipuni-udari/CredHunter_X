from __future__ import annotations

import json
import time
from typing import Any, cast

import litellm
from litellm.exceptions import RateLimitError, ServiceUnavailableError
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from credhunter_x.llm.client import LLMResponse, LLMToolResponse, ToolCall
from credhunter_x.llm.schema import ClassificationSchema

# Silences litellm's noisy "Provider List" debug print (an internal,
# harmless provider lookup fails for openrouter/* models) -- litellm's own
# documented flag for it, not an error-handling change.
litellm.suppress_debug_info = True

_MAX_ATTEMPTS = 5
_BASE_DELAY_SECONDS = 1.0
_MAX_DELAY_SECONDS = 30.0
_RETRYABLE_ERRORS = (RateLimitError, ServiceUnavailableError)


class LiteLLMClientError(RuntimeError):
    pass


class LiteLLMClient:
    """Thin wrapper around litellm.completion() -- the single LLM adapter
    for the whole project. Switching providers is a config change, never a
    new adapter file. Never construct this unwrapped; see guarded_client.py.

    Rate-limit/service-unavailable errors are retried with backoff up to
    _MAX_ATTEMPTS; everything else fails immediately."""

    def __init__(self, model: str, api_key: str) -> None:
        if not model or not api_key:
            raise LiteLLMClientError("LLM_MODEL and LLM_API_KEY must both be set")
        self._model = model
        self._api_key = api_key

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
        response = self._call_with_retry(
            messages=[{"role": "user", "content": prompt}],
            response_format=response_schema,
        )
        latency_ms = (time.monotonic() - start) * 1000

        # mypy can't see `usage` via normal attribute access on ModelResponse
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=response.choices[0].message.content or "",
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
        # No response_format: forced JSON schema + tool-calling isn't
        # reliable across providers. The prompt asks for schema-matching
        # JSON in plain language instead once no more tools are needed.
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
