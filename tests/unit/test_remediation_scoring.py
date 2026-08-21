from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from credhunter_x.evaluation.remediation_scoring import (
    ElementCheckParsingError,
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
