from __future__ import annotations

import json
import re

from pydantic import ValidationError

from credhunter_x.llm.client import LLMResponse
from credhunter_x.llm.schema import ClassificationSchema
from credhunter_x.models.classification import Arm, ClassificationResult
from credhunter_x.models.treatment import Treatment


class LLMParsingError(RuntimeError):
    pass


_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Some models (e.g. minimax-m3) ignore response_format and wrap JSON in
    a markdown code fence regardless of instructions -- strip it if present
    so a well-formed answer isn't rejected as malformed."""
    match = _CODE_FENCE.match(text.strip())
    return match.group(1) if match else text


def parse_classification(
    response: LLMResponse,
    *,
    candidate_id: str,
    arm: Arm,
    treatment: Treatment,
) -> ClassificationResult:
    """The only place model output turns into a ClassificationResult.
    Malformed output raises a typed error instead of crashing."""
    try:
        data = ClassificationSchema.model_validate_json(_strip_code_fence(response.text))
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
