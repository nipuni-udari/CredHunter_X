from __future__ import annotations

from dataclasses import replace

from credhunter_x.models.candidate import Candidate


def _same_secret(a: Candidate, b: Candidate) -> bool:
    return (
        a.file_path == b.file_path
        and a.line_start <= b.line_end
        and b.line_start <= a.line_end
        and (a.matched_value in b.matched_value or b.matched_value in a.matched_value)
    )


def merge_candidates(primary: list[Candidate], secondary: list[Candidate]) -> list[Candidate]:
    """Combines two independently-generated candidate lists (e.g. gitleaks +
    trufflehog) so the same real secret never reaches the LLM twice. Two
    candidates count as the same secret when they're in the same file,
    their line ranges overlap, and one's matched value is a substring of
    the other's -- trufflehog often reports just the high-entropy fragment
    it matched, not the whole token gitleaks reports. `primary`'s version
    is kept on a match (its value-offset resolution is the tested one),
    with `source` updated to record that both detectors found it."""
    merged: list[Candidate] = []
    secondary_matched = [False] * len(secondary)

    for p in primary:
        match_idx = next(
            (i for i, s in enumerate(secondary) if not secondary_matched[i] and _same_secret(p, s)),
            None,
        )
        if match_idx is None:
            merged.append(p)
            continue
        secondary_matched[match_idx] = True
        merged.append(replace(p, source=f"{p.source}+{secondary[match_idx].source}"))

    merged.extend(s for i, s in enumerate(secondary) if not secondary_matched[i])
    return merged
