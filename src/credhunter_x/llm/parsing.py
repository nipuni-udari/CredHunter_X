from __future__ import annotations

import json

from pydantic import ValidationError

from credhunter_x.llm.client import LLMResponse
from credhunter_x.llm.schema import ClassificationSchema
from credhunter_x.models.classification import Arm, ClassificationResult
from credhunter_x.models.treatment import Treatment


class LLMParsingError(RuntimeError):
    pass


def parse_classification(
    response: LLMResponse,
    *,
    candidate_id: str,
    arm: Arm,
    treatment: Treatment,
) -> ClassificationResult:
    """The only place model output turns into a ClassificationResult — both
    arms call this on their final answer. Malformed or missing-field output
    raises a typed error rather than crashing the batch."""
    try:
        data = ClassificationSchema.model_validate_json(response.text)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise LLMParsingError(
            f"malformed model output for candidate={candidate_id}: {exc}"
        ) from exc

    return ClassificationResult(
        candidate_id=candidate_id,
        arm=arm,
        treatment=treatment,
        label=data.label,
        confidence=data.confidence,
        severity=data.severity,
        explanation=data.explanation,
        remediation=data.remediation,
        raw_model_output=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
    )
