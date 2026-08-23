from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from credhunter_x.llm.client import LLMClient, LLMResponse
from credhunter_x.llm.element_check_prompts import build_element_check_prompt
from credhunter_x.llm.schema import ElementCheckSchema


class ElementCheckParsingError(RuntimeError):
    pass


@dataclass(frozen=True)
class ElementCheckResult:
    candidate_id: str
    rule_id: str
    required_elements: list[str]
    element_present: list[bool]
    pass_rate: float
    raw_model_output: str


def parse_element_check(
    response: LLMResponse,
    *,
    candidate_id: str,
    rule_id: str,
    required_elements: list[str],
) -> ElementCheckResult:
    """Mirrors parse_classification, plus one extra check: the schema is
    positional, so element_present's length must match required_elements'."""
    try:
        data = ElementCheckSchema.model_validate_json(response.text)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ElementCheckParsingError(
            f"malformed model output for candidate={candidate_id}: {exc}"
        ) from exc

    if len(data.element_present) != len(required_elements):
        raise ElementCheckParsingError(
            f"element_present has {len(data.element_present)} entries but "
            f"{len(required_elements)} required_elements were asked about "
            f"(candidate={candidate_id})"
        )

    pass_rate = sum(data.element_present) / len(required_elements) if required_elements else 0.0
    return ElementCheckResult(
        candidate_id=candidate_id,
        rule_id=rule_id,
        required_elements=required_elements,
        element_present=data.element_present,
        pass_rate=pass_rate,
        raw_model_output=response.text,
    )


def score_remediation(
    client: LLMClient,
    *,
    candidate_id: str,
    rule_id: str,
    remediation: str,
    required_elements: list[str],
) -> ElementCheckResult:
    prompt = build_element_check_prompt(remediation, required_elements)
    response = client.generate(
        prompt,
        candidate_id=candidate_id,
        rule_id=rule_id,
        response_schema=ElementCheckSchema,
    )
    return parse_element_check(
        response,
        candidate_id=candidate_id,
        rule_id=rule_id,
        required_elements=required_elements,
    )
