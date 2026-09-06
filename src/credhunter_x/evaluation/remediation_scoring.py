from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

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
    # Scored and reported, never gating the pass -- the reference file's
    # own grading_policy.
    optional_elements: list[str] = field(default_factory=list)
    optional_present: list[bool] = field(default_factory=list)
    # Answers to the waived_unless questions, in required_elements order
    # for the gated ones only.
    gate_answers: list[bool] = field(default_factory=list)


def build_questions(
    required_elements: list[str],
    optional_elements: Sequence[str] = (),
    required_gates: Sequence[str | None] = (),
) -> list[str]:
    """The checker answers one flat array of booleans, so everything it's
    asked has to go in one ordered list: required, then optional, then the
    waived_unless gates. Both ends use this so they can't disagree."""
    gates = list(required_gates) or [None] * len(required_elements)
    if len(gates) != len(required_elements):
        raise ValueError("required_gates must align 1:1 with required_elements")
    return [*required_elements, *optional_elements, *[g for g in gates if g is not None]]


def parse_element_check(
    response: LLMResponse,
    *,
    candidate_id: str,
    rule_id: str,
    required_elements: list[str],
    optional_elements: Sequence[str] = (),
    required_gates: Sequence[str | None] = (),
) -> ElementCheckResult:
    """Mirrors parse_classification, plus one extra check: the schema is
    positional, so element_present's length must match the number of
    questions asked."""
    optional = list(optional_elements)
    gates = list(required_gates) or [None] * len(required_elements)
    expected = len(build_questions(required_elements, optional, gates))

    try:
        data = ElementCheckSchema.model_validate_json(response.text)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ElementCheckParsingError(
            f"malformed model output for candidate={candidate_id}: {exc}"
        ) from exc

    if len(data.element_present) != expected:
        raise ElementCheckParsingError(
            f"element_present has {len(data.element_present)} entries but "
            f"{expected} question(s) were asked "
            f"(candidate={candidate_id})"
        )

    n_required = len(required_elements)
    required_present = data.element_present[:n_required]
    optional_present = data.element_present[n_required : n_required + len(optional)]
    gate_answers = data.element_present[n_required + len(optional) :]

    # A gated element is satisfied when its gate answers false: the
    # element is waived, not failed.
    answers = iter(gate_answers)
    satisfied = [
        present if gate is None else (present or not next(answers))
        for present, gate in zip(required_present, gates, strict=True)
    ]
    pass_rate = sum(satisfied) / n_required if n_required else 0.0
    return ElementCheckResult(
        candidate_id=candidate_id,
        rule_id=rule_id,
        required_elements=required_elements,
        element_present=required_present,
        pass_rate=pass_rate,
        raw_model_output=response.text,
        optional_elements=optional,
        optional_present=optional_present,
        gate_answers=gate_answers,
    )


def score_remediation(
    client: LLMClient,
    *,
    candidate_id: str,
    rule_id: str,
    remediation: str,
    required_elements: list[str],
    optional_elements: Sequence[str] = (),
    required_gates: Sequence[str | None] = (),
) -> ElementCheckResult:
    prompt = build_element_check_prompt(
        remediation, build_questions(required_elements, optional_elements, required_gates)
    )
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
        optional_elements=optional_elements,
        required_gates=required_gates,
    )
