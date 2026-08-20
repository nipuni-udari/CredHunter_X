from __future__ import annotations

import json

import pytest

from credhunter_x.llm.client import LLMResponse
from credhunter_x.llm.parsing import LLMParsingError, parse_classification
from credhunter_x.models.classification import Label, Severity
from credhunter_x.models.treatment import Treatment

WELL_FORMED = json.dumps(
    {
        "label": "true_secret",
        "confidence": 0.92,
        "severity": "high",
        "explanation": "Looks like a live GitHub PAT hardcoded in a config module.",
        "remediation": "Rotate the token and load it from an environment variable.",
    }
)


def make_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, input_tokens=10, output_tokens=20, latency_ms=123.4)


def test_parses_well_formed_json_into_classification_result():
    result = parse_classification(
        make_response(WELL_FORMED), candidate_id="c1", arm="single", treatment=Treatment.RAW
    )

    assert result.candidate_id == "c1"
    assert result.arm == "single"
    assert result.treatment == Treatment.RAW
    assert result.label == Label.TRUE_SECRET
    assert result.severity == Severity.HIGH
    assert result.confidence == 0.92
    assert result.input_tokens == 10
    assert result.output_tokens == 20
    assert result.latency_ms == 123.4
    assert result.raw_model_output == WELL_FORMED


def test_raises_typed_error_on_invalid_json():
    with pytest.raises(LLMParsingError):
        parse_classification(
            make_response("not json at all"),
            candidate_id="c1",
            arm="single",
            treatment=Treatment.RAW,
        )


def test_raises_typed_error_on_missing_required_field():
    incomplete = json.dumps({"label": "true_secret", "confidence": 0.9})
    with pytest.raises(LLMParsingError):
        parse_classification(
            make_response(incomplete), candidate_id="c1", arm="single", treatment=Treatment.RAW
        )


def test_raises_typed_error_on_invalid_label_value():
    bad_label = json.dumps(
        {
            "label": "not_a_real_label",
            "confidence": 0.9,
            "severity": "low",
            "explanation": "x",
            "remediation": "x",
        }
    )
    with pytest.raises(LLMParsingError):
        parse_classification(
            make_response(bad_label), candidate_id="c1", arm="single", treatment=Treatment.RAW
        )
