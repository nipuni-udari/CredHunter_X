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
    value_start: int  # character offset of matched_value within its line — derived via exact
    value_end: int  # string search in gitleaks/parser.py, not gitleaks' own (unreliable) columns.
    # -1 for both when matched_value can't be found verbatim in its own line.
    entropy: float
    matched_lines: list[str]  # raw text of line_start..line_end inclusive, for masking
    context_before: list[str]
    context_after: list[str]
    repo_id: str
