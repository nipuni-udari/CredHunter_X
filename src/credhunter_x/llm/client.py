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
    """One turn in a tool-calling conversation. `text` is the final answer
    only when `tool_calls` is empty; otherwise the caller must dispatch
    the tool calls and continue."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class LLMClient(Protocol):
    """Every classifier depends on this Protocol, never a concrete provider.
    Guard-related kwargs live here too so classifiers always pass them
    through; an unguarded client just ignores them."""

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
        """messages/tools use the OpenAI chat-completions/function-calling
        schema. The guard checks the whole message history every call, not
        just the latest message."""
        ...
