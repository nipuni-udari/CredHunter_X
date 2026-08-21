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


class ElementCheckSchema(BaseModel):
    """RQ3's remediation element-checker output. Positional, aligned 1:1
    with a NUMBERED required_elements list in the prompt (same order) —
    deliberately not a dict keyed by element text, which would map to an
    open JSON-schema object that structured-output modes handle far less
    reliably than a closed array, and would depend on the model echoing
    element text back verbatim as dict keys. See
    llm/element_check_prompts.py and evaluation/remediation_scoring.py."""

    element_present: list[bool]
