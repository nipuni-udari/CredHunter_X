from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Treatment(StrEnum):
    RAW = "raw"
    MASKED = "masked"
    # Reserved for RQ4, not implemented in the MVP.
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
    rather than SanitisedContext holding a single shared one, since a
    context window can contain several distinct secrets that must not be
    conflated into one placeholder."""

    start: int
    end: int
    placeholder: str
    metadata: SecretMetadata


@dataclass(frozen=True)
class SanitisedContext:
    treatment: Treatment
    sanitised_snippet: str
    masked_spans: list[MaskedSpan]
