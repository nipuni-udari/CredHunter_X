from __future__ import annotations

from collections.abc import Callable

from credhunter_x.masking.entropy import classify_charset, shannon_entropy
from credhunter_x.masking.pseudonymiser import generate_fake_value
from credhunter_x.masking.secret_registry import SecretRegistry
from credhunter_x.models.candidate import Candidate
from credhunter_x.models.treatment import MaskedSpan, SanitisedContext, SecretMetadata, Treatment

_MASK_CHAR = "•"  # bullet
_REDACTED_MARKER = "[REDACTED]"


def mask_arbitrary_text(text: str, candidates: list[Candidate]) -> str:
    """Masks every occurrence of any candidate's real value anywhere in
    arbitrary text -- for Arm B's tool results, which can pull in content
    from anywhere in the repo. Matches at the fragment level (same as
    SecretRegistry.fragments()/LeakGuard), not just a whole-value match --
    a small tool snippet often only contains part of a multi-line secret
    like a PEM key. Matching fragments are merged into contiguous spans
    before replacement so a longer leak is blanked out in full."""
    registry = SecretRegistry()
    registry.register_candidates(candidates)
    fragments = registry.fragments()
    if not fragments:
        return text

    covered = bytearray(len(text))
    for fragment in fragments:
        start = text.find(fragment)
        while start != -1:
            end = start + len(fragment)
            for i in range(start, end):
                covered[i] = 1
            start = text.find(fragment, start + 1)

    if not any(covered):
        return text

    pieces: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if covered[i]:
            j = i
            while j < n and covered[j]:
                j += 1
            pieces.append(_MASK_CHAR * (j - i))
            i = j
        else:
            pieces.append(text[i])
            i += 1
    return "".join(pieces)


def _compute_metadata(value: str, type_hint: str) -> SecretMetadata:
    """type_hint should come from something already-public (e.g. rule_id),
    never sliced from the secret value itself."""
    return SecretMetadata(
        length=len(value),
        entropy=shannon_entropy(value),
        charset=classify_charset(value),
        prefix_hint=type_hint,
    )


def mask_value(value: str, type_hint: str = "") -> tuple[str, SecretMetadata]:
    """Fully masks `value` — no real characters are ever shown."""
    return _MASK_CHAR * len(value), _compute_metadata(value, type_hint)


def pseudonymise_value(value: str, type_hint: str = "") -> tuple[str, SecretMetadata]:
    """Replaces `value` with a fake-but-realistic same-shape value. See
    pseudonymiser.py for the generation strategy."""
    return generate_fake_value(value, type_hint), _compute_metadata(value, type_hint)


def redact_value(value: str, type_hint: str = "") -> tuple[str, SecretMetadata]:
    """Replaces `value` with a short fixed marker, no length signal --
    unlike mask_value's bullet-fill, which reveals length."""
    return _REDACTED_MARKER, _compute_metadata(value, type_hint)


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


def _bullet_line_fallback(line_text: str) -> str:
    return _MASK_CHAR * len(line_text)


def _redacted_line_fallback(line_text: str) -> str:
    del line_text
    return _REDACTED_MARKER


def _replace_candidate_in_lines(
    candidate: Candidate,
    lines_by_number: dict[int, str],
    *,
    value_fn: Callable[[str, str], tuple[str, SecretMetadata]],
    line_fallback_fn: Callable[[str], str],
) -> MaskedSpan:
    """Replaces every occurrence of candidate's value anywhere in the
    window, not just its own line range -- the same value can legitimately
    recur nearby. Falls back to line_fallback_fn over the candidate's own
    lines if no exact match is found (e.g. a multi-line value). value_fn/
    line_fallback_fn are parameterised so each treatment gets its own
    correct fallback."""
    placeholder, metadata = value_fn(candidate.matched_value, candidate.rule_id)
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
                lines_by_number[line_no] = line_fallback_fn(line_text)
    return MaskedSpan(
        start=candidate.line_start,
        end=candidate.line_end,
        placeholder=placeholder,
        metadata=metadata,
    )


def _mask_candidate_in_lines(candidate: Candidate, lines_by_number: dict[int, str]) -> MaskedSpan:
    return _replace_candidate_in_lines(
        candidate, lines_by_number, value_fn=mask_value, line_fallback_fn=_bullet_line_fallback
    )


def build_raw_context(target: Candidate, other_candidates: list[Candidate]) -> SanitisedContext:
    """Unmasked window for target only -- research baseline, never the
    shipped default. Every other candidate sharing this window is still
    masked; RAW shows this one value, not everything nearby."""
    lines_by_number = _window_lines(target)
    others = _others_in_window(target, other_candidates, set(lines_by_number))

    masked_spans = [_mask_candidate_in_lines(c, lines_by_number) for c in others]
    snippet = "\n".join(lines_by_number[n] for n in sorted(lines_by_number))

    return SanitisedContext(
        treatment=Treatment.RAW, sanitised_snippet=snippet, masked_spans=masked_spans
    )


def mask_context_window(target: Candidate, other_candidates: list[Candidate]) -> SanitisedContext:
    """Builds the masked window around `target`, masking every other
    candidate whose line falls inside it too -- otherwise +/-N lines of
    context would leak whatever else sits there."""
    lines_by_number = _window_lines(target)
    others = _others_in_window(target, other_candidates, set(lines_by_number))

    masked_spans = [_mask_candidate_in_lines(c, lines_by_number) for c in [target, *others]]
    snippet = "\n".join(lines_by_number[n] for n in sorted(lines_by_number))

    return SanitisedContext(
        treatment=Treatment.MASKED, sanitised_snippet=snippet, masked_spans=masked_spans
    )


def build_pseudonymised_context(
    target: Candidate, other_candidates: list[Candidate]
) -> SanitisedContext:
    """Same window-scoping as mask_context_window, but every value is
    replaced with a fake-but-realistic same-shape value -- see
    pseudonymiser.py. Multi-line values (private keys) fall back to
    plain bullet-masking instead; synthesising a realistic multi-line PEM
    block is out of scope."""
    lines_by_number = _window_lines(target)
    others = _others_in_window(target, other_candidates, set(lines_by_number))

    masked_spans = [
        _replace_candidate_in_lines(
            c, lines_by_number, value_fn=pseudonymise_value, line_fallback_fn=_bullet_line_fallback
        )
        for c in [target, *others]
    ]
    snippet = "\n".join(lines_by_number[n] for n in sorted(lines_by_number))

    return SanitisedContext(
        treatment=Treatment.PSEUDONYMISED, sanitised_snippet=snippet, masked_spans=masked_spans
    )


def build_metadata_only_context(
    target: Candidate, other_candidates: list[Candidate]
) -> SanitisedContext:
    """Same window-scoping as mask_context_window, but every value becomes
    a short fixed marker with no length signal, instead of a length-matched
    bullet-fill."""
    lines_by_number = _window_lines(target)
    others = _others_in_window(target, other_candidates, set(lines_by_number))

    masked_spans = [
        _replace_candidate_in_lines(
            c, lines_by_number, value_fn=redact_value, line_fallback_fn=_redacted_line_fallback
        )
        for c in [target, *others]
    ]
    snippet = "\n".join(lines_by_number[n] for n in sorted(lines_by_number))

    return SanitisedContext(
        treatment=Treatment.METADATA_ONLY, sanitised_snippet=snippet, masked_spans=masked_spans
    )
