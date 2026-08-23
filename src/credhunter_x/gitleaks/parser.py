from __future__ import annotations

from pathlib import Path
from typing import Any

from credhunter_x.models.candidate import Candidate


def parse_gitleaks_report(
    findings: list[dict[str, Any]],
    source_root: Path,
    repo_id: str = "",
    context_lines: int = 10,
) -> list[Candidate]:
    """Convert gitleaks' raw JSON findings into Candidate objects.
    file_path is relative to source_root (POSIX-style), matching CredData's
    own convention. value_start/value_end come from searching for
    matched_value in its own line rather than trusting gitleaks'
    StartColumn/EndColumn, which we've seen be wrong."""
    source_root = source_root.resolve()
    file_cache: dict[Path, list[str]] = {}
    candidates = []

    for finding in findings:
        abs_path = Path(finding["File"])
        if not abs_path.is_absolute():
            abs_path = source_root / abs_path
        abs_path = abs_path.resolve()
        rel_path = abs_path.relative_to(source_root).as_posix()

        if abs_path not in file_cache:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
            file_cache[abs_path] = text.splitlines()
        lines = file_cache[abs_path]

        line_start = int(finding["StartLine"])
        line_end = int(finding["EndLine"])

        before_start = max(0, line_start - 1 - context_lines)
        context_before = lines[before_start : line_start - 1]
        context_after = lines[line_end : line_end + context_lines]
        matched_lines = lines[line_start - 1 : line_end]
        matched_value = finding["Secret"]
        first_line = matched_lines[0] if matched_lines else ""
        derived_start = first_line.find(matched_value)
        if derived_start == -1:
            value_start, value_end = -1, -1
        else:
            value_start = derived_start
            value_end = derived_start + len(matched_value)

        candidates.append(
            Candidate(
                id=finding["Fingerprint"],
                file_path=rel_path,
                line_start=line_start,
                line_end=line_end,
                rule_id=finding["RuleID"],
                matched_value=matched_value,
                value_start=value_start,
                value_end=value_end,
                entropy=float(finding["Entropy"]),
                matched_lines=matched_lines,
                context_before=context_before,
                context_after=context_after,
                repo_id=repo_id,
            )
        )

    return candidates
