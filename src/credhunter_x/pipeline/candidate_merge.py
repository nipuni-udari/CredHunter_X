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
    with `source` updated to record that both detectors found it.

    A candidate absorbs *every* other detection of the same secret, not just
    the first. One value routinely trips several rules at once -- gitleaks'
    `private-key` and trufflehog's `private.key` on one PEM header, or
    `generic-api-key` + `generic.secret` + `high-entropy` on one assignment --
    and pairing them off one-to-one left the surplus behind as its own
    candidate, costing an LLM call per surplus row and raising one alert per
    rule instead of one per secret. Repeats inside a single detector's own
    output are dropped for the same reason."""
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
