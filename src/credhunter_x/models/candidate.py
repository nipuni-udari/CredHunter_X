from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    id: str
    file_path: str
    line_start: int
    line_end: int
    rule_id: str
    matched_value: str  # the real secret — must never reach logs or an unguarded LLM call
    value_start: int  # offset of matched_value in its line; -1 if not found verbatim
    value_end: int
    entropy: float
    matched_lines: list[str]  # raw text of line_start..line_end, for masking
    context_before: list[str]
    context_after: list[str]
    repo_id: str
