from __future__ import annotations

from pydantic import BaseModel

from credhunter_x.models.classification import Label, Severity


class ClassificationSchema(BaseModel):
    """The exact JSON shape requested from the model via Gemini's
    response_schema structured-output mode — also the only shape
    llm/parsing.py ever parses a response against."""

    label: Label
    confidence: float
    severity: Severity
    explanation: str
    remediation: str
