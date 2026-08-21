from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from credhunter_x.llm.schema import ClassificationSchema


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMToolResponse:
    """Result of one turn in a tool-calling conversation. `text` is the
    model's final answer when `tool_calls` is empty; when `tool_calls` is
    non-empty, `text` may be empty or a short aside and the caller must
    dispatch the tool calls and continue the conversation rather than
    treating this as a final answer."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class LLMClient(Protocol):
    """Every classifier depends on this Protocol, never a concrete provider.
    The guard-related keyword args live on the interface itself (not just
    GuardedLLMClient) so classifiers always pass candidate context through
    regardless of which concrete client they're holding — an unguarded
    client just accepts and ignores them."""

    def generate(
        self,
        prompt: str,
        *,
        candidate_id: str = "",
        rule_id: str = "",
        raw_permit_candidate_id: str | None = None,
        response_schema: type[BaseModel] = ClassificationSchema,
    ) -> LLMResponse: ...

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        candidate_id: str = "",
        rule_id: str = "",
        raw_permit_candidate_id: str | None = None,
    ) -> LLMToolResponse:
        """messages/tools use the OpenAI chat-completions message and
        function-calling schema — the common format litellm normalises
        every provider to. The guard checks the ENTIRE serialised message
        history on every call (see guarded_client.py), not just the latest
        message, so an unmasked tool result appended earlier is still
        caught on the next turn."""
        ...
