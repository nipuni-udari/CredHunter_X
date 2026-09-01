from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Treatment(StrEnum):
    RAW = "raw"
    MASKED = "masked"
    PSEUDONYMISED = "pseudonymised"
    METADATA_ONLY = "metadata_only"


@dataclass(frozen=True)
class SecretMetadata:
    length: int
    entropy: float
    charset: str
    prefix_hint: str


@dataclass(frozen=True)
class MaskedSpan:
    """One masked secret within a context window. Carries its own metadata
    since a window can hold several distinct secrets.

    is_target records whether this span is the candidate being classified
    rather than a neighbour. It has to be carried here because start/end
    cannot recover it: two distinct secrets on one line produce spans with
    identical line ranges, so a line-number comparison labels both as the
    candidate under review and hands the model contradictory metadata."""

    start: int
    end: int
    placeholder: str
    metadata: SecretMetadata
    is_target: bool = False


@dataclass(frozen=True)
class SanitisedContext:
    treatment: Treatment
    sanitised_snippet: str
    masked_spans: list[MaskedSpan]
