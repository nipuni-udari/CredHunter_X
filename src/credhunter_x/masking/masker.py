from __future__ import annotations

from credhunter_x.masking.entropy import classify_charset, shannon_entropy
from credhunter_x.models.candidate import Candidate
from credhunter_x.models.treatment import MaskedSpan, SanitisedContext, SecretMetadata, Treatment

_MASK_CHAR = "•"  # bullet


def mask_arbitrary_text(text: str, candidates: list[Candidate]) -> str:
    """Masks every occurrence of any candidate's real value found anywhere
    in arbitrary text — for Arm B's tool results, which can pull in content
    from any file/location, not just a candidate's own precomputed context
    window. Unlike the window-based masking below, this needs no per-line
    bookkeeping: a plain string replace handles multi-line values fine, and
    there's no "target vs bystander" distinction for tool output — every
    known secret is masked regardless of whose window it came from."""
    for candidate in candidates:
        if candidate.matched_value and candidate.matched_value in text:
            placeholder, _ = mask_value(candidate.matched_value, candidate.rule_id)
            text = text.replace(candidate.matched_value, placeholder)
    return text


def mask_value(value: str, type_hint: str = "") -> tuple[str, SecretMetadata]:
    """Fully masks `value` — no real characters are ever shown. type_hint
    should come from something already-public about the finding (e.g. the
    scanner's rule_id), never sliced from the secret value itself: only a
    handful of credential types have a prefix that's genuinely constant
    across every instance of that type (safe to reveal), and hardcoding
    that list isn't worth the risk of getting it wrong for MVP."""
    metadata = SecretMetadata(
        length=len(value),
        entropy=shannon_entropy(value),
        charset=classify_charset(value),
        prefix_hint=type_hint,
    )
    placeholder = _MASK_CHAR * len(value)
    return placeholder, metadata


def _window_lines(target: Candidate) -> dict[int, str]:
    """Absolute line numbers -> text for target's +/-N context window."""
    window_start_line = target.line_start - len(target.context_before)
    lines_by_number: dict[int, str] = {}
    for offset, text in enumerate(target.context_before):
        lines_by_number[window_start_line + offset] = text
    for offset, text in enumerate(target.matched_lines):
        lines_by_number[target.line_start + offset] = text
    for offset, text in enumerate(target.context_after):
        lines_by_number[target.line_end + 1 + offset] = text
    return lines_by_number


def _others_in_window(
    target: Candidate, other_candidates: list[Candidate], window_line_numbers: set[int]
) -> list[Candidate]:
    return [
        c
        for c in other_candidates
        if c.file_path == target.file_path
        and c.id != target.id
        and window_line_numbers & set(range(c.line_start, c.line_end + 1))
    ]


def _mask_candidate_in_lines(candidate: Candidate, lines_by_number: dict[int, str]) -> MaskedSpan:
    """Masks every occurrence of candidate's value anywhere in the window —
    not just within its own reported line range, since the identical value
    can legitimately recur elsewhere nearby (e.g. a test assertion
    re-checking a value gitleaks only flagged once, at its first
    occurrence). Tries an exact substring replace first (preserves
    surrounding code on each line). If that finds no match anywhere —
    matched_value spans multiple lines and can never be "in" a single
    line's text, or gitleaks reported a normalised value (e.g. URL-decoded)
    that doesn't literally appear in the source — falls back to masking
    candidate's own line range outright, rather than silently leaving real
    content unmasked."""
    placeholder, metadata = mask_value(candidate.matched_value, candidate.rule_id)
    replaced_any = False
    if "\n" not in candidate.matched_value:
        for line_no, text in lines_by_number.items():
            if candidate.matched_value in text:
                lines_by_number[line_no] = text.replace(candidate.matched_value, placeholder)
                replaced_any = True

    if not replaced_any:
        for line_no in range(candidate.line_start, candidate.line_end + 1):
            line_text = lines_by_number.get(line_no)
            if line_text is not None:
                lines_by_number[line_no] = _MASK_CHAR * len(line_text)
    return MaskedSpan(
        start=candidate.line_start,
        end=candidate.line_end,
        placeholder=placeholder,
        metadata=metadata,
    )


def build_raw_context(target: Candidate, other_candidates: list[Candidate]) -> SanitisedContext:
    """Unmasked window for target only — research baseline, never the
    shipped default (see config/settings.py's load_scan_config). Every
    OTHER candidate sharing this window is still masked: RAW treatment
    means "show this one candidate's real value," not "expose every
    secret nearby" — a bystander secret must never ride along just
    because it happened to sit near the one under evaluation."""
    lines_by_number = _window_lines(target)
    others = _others_in_window(target, other_candidates, set(lines_by_number))

    masked_spans = [_mask_candidate_in_lines(c, lines_by_number) for c in others]
    snippet = "\n".join(lines_by_number[n] for n in sorted(lines_by_number))

    return SanitisedContext(
        treatment=Treatment.RAW, sanitised_snippet=snippet, masked_spans=masked_spans
    )


def mask_context_window(target: Candidate, other_candidates: list[Candidate]) -> SanitisedContext:
    """Builds the masked text window around `target` — masking not just
    target's own secret but every other candidate whose line falls inside
    this same window, since sending +/-N lines of context would otherwise
    leak whatever other secrets happen to sit in those lines. Masking is
    done by exact string replacement of each candidate's known matched_value
    directly, rather than slicing by value_start/value_end — simpler, and
    doesn't depend on those offsets being derived correctly for every
    candidate in the window."""
    lines_by_number = _window_lines(target)
    others = _others_in_window(target, other_candidates, set(lines_by_number))

    masked_spans = [_mask_candidate_in_lines(c, lines_by_number) for c in [target, *others]]
    snippet = "\n".join(lines_by_number[n] for n in sorted(lines_by_number))

    return SanitisedContext(
        treatment=Treatment.MASKED, sanitised_snippet=snippet, masked_spans=masked_spans
    )
