from __future__ import annotations

from credhunter_x.masking.entropy import classify_charset, shannon_entropy
from credhunter_x.models.candidate import Candidate
from credhunter_x.models.treatment import MaskedSpan, SanitisedContext, SecretMetadata, Treatment

_MASK_CHAR = "•"  # bullet


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


def build_raw_context(target: Candidate) -> SanitisedContext:
    """Unmasked text window — research baseline only, never the shipped
    default (see config/settings.py's ScanConfig)."""
    lines = [*target.context_before, *target.matched_lines, *target.context_after]
    return SanitisedContext(
        treatment=Treatment.RAW, sanitised_snippet="\n".join(lines), masked_spans=[]
    )


def _mask_candidate_in_lines(candidate: Candidate, lines_by_number: dict[int, str]) -> MaskedSpan:
    placeholder, metadata = mask_value(candidate.matched_value, candidate.rule_id)
    for line_no in range(candidate.line_start, candidate.line_end + 1):
        text = lines_by_number.get(line_no)
        if text is not None and candidate.matched_value in text:
            lines_by_number[line_no] = text.replace(candidate.matched_value, placeholder)
    return MaskedSpan(
        start=candidate.line_start,
        end=candidate.line_end,
        placeholder=placeholder,
        metadata=metadata,
    )


def mask_context_window(target: Candidate, other_candidates: list[Candidate]) -> SanitisedContext:
    """Builds the masked text window around `target` — masking not just
    target's own secret but every other candidate whose line falls inside
    this same window, since sending +/-N lines of context would otherwise
    leak whatever other secrets happen to sit in those lines. Masking is
    done by exact string replacement of each candidate's known matched_value
    (not by its scanner-reported column offsets, which aren't always
    precise — see the Candidate model's value_start/value_end docstring)."""
    window_start_line = target.line_start - len(target.context_before)

    lines_by_number: dict[int, str] = {}
    for offset, text in enumerate(target.context_before):
        lines_by_number[window_start_line + offset] = text
    for offset, text in enumerate(target.matched_lines):
        lines_by_number[target.line_start + offset] = text
    for offset, text in enumerate(target.context_after):
        lines_by_number[target.line_end + 1 + offset] = text

    window_line_numbers = set(lines_by_number)
    to_mask = [target] + [
        c
        for c in other_candidates
        if c.file_path == target.file_path
        and c.id != target.id
        and window_line_numbers & set(range(c.line_start, c.line_end + 1))
    ]

    masked_spans = [_mask_candidate_in_lines(c, lines_by_number) for c in to_mask]
    snippet = "\n".join(lines_by_number[n] for n in sorted(lines_by_number))

    return SanitisedContext(
        treatment=Treatment.MASKED, sanitised_snippet=snippet, masked_spans=masked_spans
    )
