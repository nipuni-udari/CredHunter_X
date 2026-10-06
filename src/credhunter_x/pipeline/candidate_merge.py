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
    """Merges two candidate lists (e.g. gitleaks and trufflehog) so the same
    secret never reaches the LLM twice. Two candidates match if they are in the
    same file, their line ranges overlap and one value contains the other. The
    primary candidate is kept and absorbs every match, since one value often
    trips several rules; its source records both detectors. Repeats within one
    detector's output are dropped too."""
    merged: list[Candidate] = []
    secondary_matched = [False] * len(secondary)

    for p in primary:
        if any(_same_secret(p, kept) for kept in merged):
            continue
        matched = [
            i for i, s in enumerate(secondary) if not secondary_matched[i] and _same_secret(p, s)
        ]
        for i in matched:
            secondary_matched[i] = True
        sources = sorted({secondary[i].source for i in matched})
        merged.append(replace(p, source="+".join([p.source, *sources])) if matched else p)

    for i, s in enumerate(secondary):
        if secondary_matched[i] or any(_same_secret(s, kept) for kept in merged):
            continue
        merged.append(s)
    return merged
