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
    """Converts gitleaks' JSON findings into Candidate objects. file_path is
    relative to source_root, as in CredData. value_start/value_end are found by
    searching the line, because gitleaks' StartColumn/EndColumn are sometimes
    wrong."""
    source_root = source_root.resolve()
    file_cache: dict[Path, list[str]] = {}
    candidates = []
    # gitleaks' Fingerprint (file:rule:line) isn't always unique, so add an
    # occurrence index; otherwise two findings share an id and one result is lost.
    occurrence_counts: dict[str, int] = {}

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

        fingerprint = finding["Fingerprint"]
        occurrence_index = occurrence_counts.get(fingerprint, 0)
        occurrence_counts[fingerprint] = occurrence_index + 1

        candidates.append(
            Candidate(
                id=f"{fingerprint}:{occurrence_index}",
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
