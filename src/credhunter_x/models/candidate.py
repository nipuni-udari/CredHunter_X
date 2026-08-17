from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    id: str
    file_path: str
    line_start: int
    line_end: int
    rule_id: str
    matched_value: str  # the real secret value — must never reach logs or an unguarded LLM call
    value_start: int  # character offset of matched_value within its line — scanner-reported,
    value_end: int  # not always precise; do not rely on this for exact masking, see masker.py
    entropy: float
    matched_lines: list[str]  # raw text of line_start..line_end inclusive, for masking
    context_before: list[str]
    context_after: list[str]
    repo_id: str
