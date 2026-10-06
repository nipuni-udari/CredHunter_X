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
    """One masked secret in a context window, with its own metadata, since a
    window can hold several secrets.

    is_target says whether this is the candidate being classified. It can't be
    worked out from start/end, because two secrets on one line have the same
    line range."""

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
