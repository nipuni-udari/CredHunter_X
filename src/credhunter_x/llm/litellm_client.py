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

# Silences litellm's noisy "Provider List" debug print (an internal,
# harmless provider lookup fails for openrouter/* models) -- litellm's own
# documented flag for it, not an error-handling change.
litellm.suppress_debug_info = True

_MAX_ATTEMPTS = 5
_BASE_DELAY_SECONDS = 1.0
_MAX_DELAY_SECONDS = 30.0
_RETRYABLE_ERRORS = (RateLimitError, ServiceUnavailableError, Timeout)
# A provider occasionally returns a successful response carrying no content
# at all. That is not an exception, so _call_with_retry never sees it -- see
# generate(). Kept lower than _MAX_ATTEMPTS since the two nest.
_MAX_EMPTY_ATTEMPTS = 3
# Classification JSON and agentic tool calls are both short -- nothing here
# needs the model's full output ceiling. Left uncapped, litellm requests up
# to the model's max (65536+ tokens) on every call, which OpenRouter's
# affordability pre-check reserves against the worst case and rejects
# outright on a small balance, even though the real response is a few
# hundred tokens.
_MAX_OUTPUT_TOKENS = 2048


class LiteLLMClientError(RuntimeError):
    pass


class LiteLLMClient:
    """Thin wrapper around litellm.completion() -- the single LLM adapter
    for the whole project. Switching providers is a config change, never a
    new adapter file. Never construct this unwrapped; see guarded_client.py.

    Rate-limit/service-unavailable/timeout errors are retried with backoff
    up to _MAX_ATTEMPTS; everything else fails immediately.

    reasoning_effort (LLM_REASONING_EFFORT in .env) is forwarded as-is when
    set; non-reasoning models ignore it, so it's harmless to leave unset."""

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
        # An empty completion arrives as an ordinary successful response with
        # no content, so _call_with_retry -- which only catches exceptions --
        # never sees it. It reaches the caller as unparseable output and the
        # candidate is dropped from the evaluation entirely. Seen once in a
        # 517-candidate run; the same candidate classified fine on every
        # retry, so retry here rather than lose it.
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
