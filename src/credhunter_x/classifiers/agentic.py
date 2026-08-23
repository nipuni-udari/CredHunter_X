from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from credhunter_x.classifiers.tools.check_file_exists import check_file_exists
from credhunter_x.classifiers.tools.get_gitleaks_rule import get_gitleaks_rule
from credhunter_x.classifiers.tools.search_file import search_file
from credhunter_x.llm.client import LLMClient, LLMResponse, LLMToolResponse, ToolCall
from credhunter_x.llm.parsing import parse_classification
from credhunter_x.llm.prompts import build_agentic_prompt
from credhunter_x.masking.masker import mask_arbitrary_text
from credhunter_x.models.candidate import Candidate
from credhunter_x.models.classification import ClassificationResult, Label, Severity, ToolCallRecord
from credhunter_x.models.treatment import SanitisedContext, Treatment

_MAX_TURNS = 4
_MAX_TOOL_RESULT_CHARS = 2000
_SUBMIT_TOOL_NAME = "submit_classification"

_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_file",
            "description": (
                "Search a specific file in the repository for a literal text "
                "pattern, returning matching lines with a little surrounding context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Repo-relative path of the file to search",
                    },
                    "query": {"type": "string", "description": "Literal text to search for"},
                },
                "required": ["file_path", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_file_exists",
            "description": "Check whether a file path exists in the repository being scanned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Repo-relative path to check"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gitleaks_rule",
            "description": "Look up what a GitLeaks scanner rule actually checks for.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {"type": "string", "description": "The rule_id to look up"},
                },
                "required": ["rule_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": _SUBMIT_TOOL_NAME,
            "description": (
                "Submit your final classification. Call this once you have "
                "enough information to decide, instead of responding with plain text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": ["true_secret", "false_positive", "uncertain"],
                    },
                    "confidence": {"type": "number", "description": "A float between 0 and 1"},
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Impact if label is true_secret; low otherwise",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "A short, concrete justification citing what you observed",
                    },
                    "remediation": {
                        "type": "string",
                        "description": "A short, actionable next step",
                    },
                },
                "required": ["label", "confidence", "severity", "explanation", "remediation"],
            },
        },
    },
]


class AgenticClassifier:
    """Arm B: a turn-capped loop where the LLM can call tools before
    answering. Every tool result is masked against all known candidates
    before being appended to the conversation; the guard re-checks the
    full history on every call anyway, as a second layer."""

    def __init__(
        self, client: LLMClient, source_root: Path, all_candidates: list[Candidate]
    ) -> None:
        self._client = client
        self._source_root = source_root
        self._all_candidates = all_candidates

    def classify(self, candidate: Candidate, context: SanitisedContext) -> ClassificationResult:
        raw_permit = candidate.id if context.treatment == Treatment.RAW else None
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": build_agentic_prompt(candidate, context)}
        ]
        tool_call_records: list[ToolCallRecord] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = 0.0

        for turn in range(1, _MAX_TURNS + 1):
            if turn == _MAX_TURNS:
                # Dropping the tools list here would be cleaner, but Groq
                # errors out if tool_calls already happened and none are
                # on offer. A text nudge is weaker but won't crash the call.
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "This is your final turn. Call submit_classification now "
                            "with your final answer instead of any other tool."
                        ),
                    }
                )
            response = self._client.generate_with_tools(
                messages,
                _TOOL_SCHEMAS,
                candidate_id=candidate.id,
                rule_id=candidate.rule_id,
                raw_permit_candidate_id=raw_permit,
            )
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            total_latency_ms += response.latency_ms

            if not response.tool_calls:
                return self._finalise(
                    candidate,
                    context,
                    response.text,
                    turn,
                    tool_call_records,
                    total_input_tokens,
                    total_output_tokens,
                    total_latency_ms,
                )

            submission = next(
                (tc for tc in response.tool_calls if tc.name == _SUBMIT_TOOL_NAME), None
            )
            if submission is not None:
                # Some models route structured output through tool-calling
                # even when asked for plain text, so a call to this tool
                # counts as the final answer.
                return self._finalise(
                    candidate,
                    context,
                    json.dumps(submission.arguments),
                    turn,
                    tool_call_records,
                    total_input_tokens,
                    total_output_tokens,
                    total_latency_ms,
                )

            messages.append(self._assistant_message(response))
            for tool_call in response.tool_calls:
                raw_result = self._dispatch(tool_call)
                masked_result = mask_arbitrary_text(raw_result, self._all_candidates)
                masked_result = masked_result[:_MAX_TOOL_RESULT_CHARS]
                tool_call_records.append(
                    ToolCallRecord(
                        tool_name=tool_call.name,
                        arguments={k: str(v) for k, v in tool_call.arguments.items()},
                        result_summary=masked_result[:200],
                    )
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": masked_result}
                )

        # Ran out of turns -- fail safe with UNCERTAIN instead of crashing.
        return ClassificationResult(
            candidate_id=candidate.id,
            arm="agentic",
            treatment=context.treatment,
            label=Label.UNCERTAIN,
            confidence=0.0,
            severity=Severity.LOW,
            explanation="No answer produced within the turn cap.",
            remediation="Review manually.",
            raw_model_output="",
            tool_calls=tool_call_records,
            turns=_MAX_TURNS,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            latency_ms=total_latency_ms,
        )

    def _finalise(
        self,
        candidate: Candidate,
        context: SanitisedContext,
        text: str,
        turn: int,
        tool_call_records: list[ToolCallRecord],
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> ClassificationResult:
        response = LLMResponse(
            text=text, input_tokens=input_tokens, output_tokens=output_tokens, latency_ms=latency_ms
        )
        result = parse_classification(
            response, candidate_id=candidate.id, arm="agentic", treatment=context.treatment
        )
        return replace(result, tool_calls=tool_call_records, turns=turn)

    def _assistant_message(self, response: LLMToolResponse) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": response.text or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in response.tool_calls
            ],
        }

    def _dispatch(self, tool_call: ToolCall) -> str:
        args = tool_call.arguments
        if tool_call.name == "search_file":
            return search_file(
                self._source_root, str(args.get("file_path", "")), str(args.get("query", ""))
            )
        if tool_call.name == "check_file_exists":
            return check_file_exists(self._source_root, str(args.get("file_path", "")))
        if tool_call.name == "get_gitleaks_rule":
            return get_gitleaks_rule(str(args.get("rule_id", "")))
        return f"Unknown tool: {tool_call.name}"
