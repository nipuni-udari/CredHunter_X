from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from credhunter_x.models.treatment import Treatment

Arm = Literal["gitleaks_only", "single", "agentic"]


class Label(StrEnum):
    TRUE_SECRET = "true_secret"
    FALSE_POSITIVE = "false_positive"
    UNCERTAIN = "uncertain"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, str]
    result_summary: str


@dataclass(frozen=True)
class ClassificationResult:
    candidate_id: str
    arm: Arm
    treatment: Treatment
    label: Label
    confidence: float
    severity: Severity
    explanation: str
    remediation: str
    raw_model_output: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    turns: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
