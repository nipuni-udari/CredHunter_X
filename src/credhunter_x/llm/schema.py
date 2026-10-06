from __future__ import annotations

from pydantic import BaseModel

from credhunter_x.models.classification import Label, Severity


class ClassificationSchema(BaseModel):
    """The JSON shape requested from the model, and the only one parsing.py
    accepts."""

    label: Label
    confidence: float
    severity: Severity
    explanation: str
    remediation: str


class ElementCheckSchema(BaseModel):
    """Output of the RQ3 element checker: a list of booleans in the order of
    required_elements. Structured-output modes handle a list more reliably than
    a dict keyed by element text."""

    element_present: list[bool]
