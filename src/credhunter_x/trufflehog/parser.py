from __future__ import annotations

from pathlib import Path
from typing import Any

from credhunter_x.masking.entropy import shannon_entropy
from credhunter_x.models.candidate import Candidate


def parse_trufflehog_report(
    findings: list[dict[str, Any]],
    source_root: Path,
    repo_id: str = "",
    context_lines: int = 10,
) -> list[Candidate]:
    """Convert trufflehog3's raw JSON findings into Candidate objects.
    file_path is relative to source_root (POSIX-style), matching the
    gitleaks parser's convention. trufflehog3 doesn't report a fixed
    "secret" span the way gitleaks does -- its "secret" field is often just
    the high-entropy fragment within the line, not the whole token -- so
    value_start/value_end come from searching for it in the line, same as
    the gitleaks parser. trufflehog3 also reports no entropy score, so it's
    computed here from the matched value itself."""
    source_root = source_root.resolve()
    file_cache: dict[Path, list[str]] = {}
    candidates = []
    # trufflehog3's generic high-entropy rule can flag more than one distinct
    # secret on the same line -- file:rule:line alone isn't a unique id then,
    # which silently drops one of the two from classify_candidates' id-keyed
    # results (last one processed wins). An occurrence index closes that.
    occurrence_counts: dict[tuple[str, str, int], int] = {}

    for finding in findings:
        abs_path = Path(finding["path"])
        if not abs_path.is_absolute():
            abs_path = source_root / abs_path
        abs_path = abs_path.resolve()
        rel_path = abs_path.relative_to(source_root).as_posix()

        if abs_path not in file_cache:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
            file_cache[abs_path] = text.splitlines()
        lines = file_cache[abs_path]

        context = finding["context"]
        issue_lines = sorted(int(n) for n in context)
        line_start = issue_lines[0]
        line_end = issue_lines[-1]
        matched_lines = [context[str(n)] for n in issue_lines]

        before_start = max(0, line_start - 1 - context_lines)
        context_before = lines[before_start : line_start - 1]
        context_after = lines[line_end : line_end + context_lines]

        matched_value = finding["secret"]
        rule_id = finding["rule"]["id"]
        first_line = matched_lines[0] if matched_lines else ""
        derived_start = first_line.find(matched_value)
        if derived_start == -1:
            value_start, value_end = -1, -1
        else:
            value_start = derived_start
            value_end = derived_start + len(matched_value)

        occurrence_key = (rel_path, rule_id, line_start)
        occurrence_index = occurrence_counts.get(occurrence_key, 0)
        occurrence_counts[occurrence_key] = occurrence_index + 1

        candidates.append(
            Candidate(
                id=f"{rel_path}:{rule_id}:{line_start}:{occurrence_index}",
                file_path=rel_path,
                line_start=line_start,
                line_end=line_end,
                rule_id=rule_id,
                matched_value=matched_value,
                value_start=value_start,
                value_end=value_end,
                entropy=shannon_entropy(matched_value),
                matched_lines=matched_lines,
                context_before=context_before,
                context_after=context_after,
                repo_id=repo_id,
                source="trufflehog",
            )
        )

    return candidates
