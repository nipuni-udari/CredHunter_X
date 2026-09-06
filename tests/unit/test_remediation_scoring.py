from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from credhunter_x.evaluation.remediation_scoring import (
    ElementCheckParsingError,
    build_questions,
    parse_element_check,
    score_remediation,
)
from credhunter_x.llm.client import LLMResponse
from credhunter_x.llm.schema import ClassificationSchema

REQUIRED_ELEMENTS = ["rotate the key", "use an environment variable", "add to .gitignore"]


def make_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, input_tokens=10, output_tokens=5, latency_ms=42.0)


def test_parses_well_formed_response_and_computes_pass_rate():
    text = json.dumps({"element_present": [True, False, True]})

    result = parse_element_check(
        make_response(text),
        candidate_id="c1",
        rule_id="aws-access-token",
        required_elements=REQUIRED_ELEMENTS,
    )

    assert result.candidate_id == "c1"
    assert result.rule_id == "aws-access-token"
    assert result.element_present == [True, False, True]
    assert result.pass_rate == pytest.approx(2 / 3)


def test_raises_typed_error_on_invalid_json():
    with pytest.raises(ElementCheckParsingError):
        parse_element_check(
            make_response("not json at all"),
            candidate_id="c1",
            rule_id="aws-access-token",
            required_elements=REQUIRED_ELEMENTS,
        )


def test_raises_typed_error_on_length_mismatch():
    """The one validation rule with no analogue in parse_classification —
    the schema is positional, so a wrong-length answer is unusable, not
    just differently shaped."""
    text = json.dumps({"element_present": [True, False]})  # only 2, expected 3

    with pytest.raises(ElementCheckParsingError):
        parse_element_check(
            make_response(text),
            candidate_id="c1",
            rule_id="aws-access-token",
            required_elements=REQUIRED_ELEMENTS,
        )


def test_pass_rate_is_zero_when_no_elements_are_required():
    text = json.dumps({"element_present": []})

    result = parse_element_check(
        make_response(text), candidate_id="c1", rule_id="unknown-rule", required_elements=[]
    )

    assert result.pass_rate == 0.0


class _FakeLLMClient:
    def __init__(self, response: LLMResponse) -> None:
        self._response = response
        self.received_prompts: list[str] = []
        self.received_schemas: list[type[BaseModel]] = []

    def generate(
        self,
        prompt: str,
        *,
        candidate_id: str = "",
        rule_id: str = "",
        raw_permit_candidate_id: str | None = None,
        response_schema: type[BaseModel] = ClassificationSchema,
    ) -> LLMResponse:
        self.received_prompts.append(prompt)
        self.received_schemas.append(response_schema)
        return self._response


def test_score_remediation_sends_the_element_check_schema_and_every_element():
    from credhunter_x.llm.schema import ElementCheckSchema

    text = json.dumps({"element_present": [True, True, False]})
    client = _FakeLLMClient(make_response(text))

    result = score_remediation(
        client,
        candidate_id="c1",
        rule_id="aws-access-token",
        remediation="Rotate the key and load it from an env var.",
        required_elements=REQUIRED_ELEMENTS,
    )

    assert result.element_present == [True, True, False]
    assert client.received_schemas == [ElementCheckSchema]
    prompt = client.received_prompts[0]
    for element in REQUIRED_ELEMENTS:
        assert element in prompt


def test_build_questions_orders_required_then_optional_then_gates():
    questions = build_questions(["req1", "req2"], ["opt1"], [None, "gate for req2"])

    assert questions == ["req1", "req2", "opt1", "gate for req2"]


def test_build_questions_raises_when_gates_do_not_align_with_required():
    with pytest.raises(ValueError):
        build_questions(["req1", "req2"], [], ["only one gate"])


def test_optional_elements_are_scored_but_never_gate_the_pass():
    # both required present, both optional absent -> still a full pass
    text = json.dumps({"element_present": [True, True, False, False]})

    result = parse_element_check(
        make_response(text),
        candidate_id="c1",
        rule_id="generic-api-key",
        required_elements=["req1", "req2"],
        optional_elements=["opt1", "opt2"],
    )

    assert result.pass_rate == 1.0
    assert result.element_present == [True, True]
    assert result.optional_present == [False, False]


def test_gated_element_is_waived_when_its_gate_answers_false():
    """The high-entropy case: the remediation says the value is not a
    credential, so 'revoke it' is not a defect."""
    # [identify=True, revoke=False, gate=False]
    text = json.dumps({"element_present": [True, False, False]})

    result = parse_element_check(
        make_response(text),
        candidate_id="c1",
        rule_id="high-entropy",
        required_elements=["identify the value", "revoke at provider"],
        required_gates=[None, "the remediation treats it as a live credential"],
    )

    assert result.pass_rate == 1.0
    assert result.gate_answers == [False]


def test_gated_element_is_still_required_when_its_gate_answers_true():
    text = json.dumps({"element_present": [True, False, True]})

    result = parse_element_check(
        make_response(text),
        candidate_id="c1",
        rule_id="high-entropy",
        required_elements=["identify the value", "revoke at provider"],
        required_gates=[None, "the remediation treats it as a live credential"],
    )

    assert result.pass_rate == 0.5
    assert result.gate_answers == [True]


def test_length_check_counts_gates_and_optional_elements_too():
    text = json.dumps({"element_present": [True, True]})

    with pytest.raises(ElementCheckParsingError):
        parse_element_check(
            make_response(text),
            candidate_id="c1",
            rule_id="high-entropy",
            required_elements=["req1", "req2"],
            optional_elements=["opt1"],
            required_gates=[None, "a gate"],
        )
