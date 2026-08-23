from __future__ import annotations

from pydantic import BaseModel

from credhunter_x.models.classification import Label, Severity


class ClassificationSchema(BaseModel):
    """The exact JSON shape requested via structured-output mode -- the
    only shape parsing.py ever validates against."""

    label: Label
    confidence: float
    severity: Severity
    explanation: str
    remediation: str


class ElementCheckSchema(BaseModel):
    """RQ3's element-checker output. Positional, matching the numbered
    required_elements list order -- not a dict keyed by element text,
    which structured-output modes handle less reliably."""

    element_present: list[bool]
