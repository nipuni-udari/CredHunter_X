from __future__ import annotations

from collections.abc import Callable

from credhunter_x.masking.entropy import classify_charset, shannon_entropy
from credhunter_x.masking.pseudonymiser import generate_fake_value
from credhunter_x.masking.secret_registry import SecretRegistry, fragments_of, secret_components
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


def _others_to_scrub(target: Candidate, other_candidates: list[Candidate]) -> list[Candidate]:
    """Every other known secret in the same file -- deliberately NOT
    filtered to the window's line range.

    Which secrets to *scrub* and which to *describe* are different
    questions, and this used to be one line-range filter answering both. A
    candidate reported at line 500 whose value is also used at line 6 sits
    squarely inside this window's text while its line number says
    otherwise, so filtering the scrub by line range sent it to the model
    unmasked. Scrubbing a value that isn't present is a harmless no-op, so
    the scrub side can afford to be broad; the metadata side stays narrow
    by describing only spans that actually replaced something."""
    return [c for c in other_candidates if c.file_path == target.file_path and c.id != target.id]


def _bullet_line_fallback(line_text: str) -> str:
    return _MASK_CHAR * len(line_text)


def _redacted_line_fallback(line_text: str) -> str:
    del line_text
    return _REDACTED_MARKER


def _scrub_fragments_in_line(
    text: str, fragments: set[str], line_fallback_fn: Callable[[str], str]
) -> str:
    """Blanks every fragment occurrence in `text` in a single pass. Every
    position is resolved against the original text before anything is
    replaced: fragments overlap each other by design (they're sliding
    FRAGMENT_LEN-character windows of one value), so replacing them one at a
    time mutates the very text the remaining ones still need to match. That
    strands real secret characters between replacements and makes the result
    depend on set iteration order, which Python randomises per process. Same
    coverage-then-merge technique as mask_arbitrary_text."""
    covered = bytearray(len(text))
    for fragment in fragments:
        if not fragment:
            continue
        start = text.find(fragment)
        while start != -1:
            for i in range(start, start + len(fragment)):
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
            pieces.append(line_fallback_fn(text[i:j]))
            i = j
        else:
            pieces.append(text[i])
            i += 1
    return "".join(pieces)


def _replace_candidate_in_lines(
    candidate: Candidate,
    lines_by_number: dict[int, str],
    *,
    value_fn: Callable[[str, str], tuple[str, SecretMetadata]],
    line_fallback_fn: Callable[[str], str],
    is_target: bool = False,
) -> MaskedSpan | None:
    """Replaces every occurrence of candidate's value anywhere in the
    window, not just its own line range -- the same value can legitimately
    recur nearby. Falls back to line_fallback_fn over the candidate's own
    lines if no exact whole-value match is found (e.g. a multi-line
    value). value_fn/line_fallback_fn are parameterised so each treatment
    gets its own correct fallback.

    Afterwards, scrubs any remaining FRAGMENT_LEN-character chunk of the
    value anywhere else in the window (see _scrub_fragments_in_line).
    LeakGuard checks payloads at that same granularity (see
    SecretRegistry.fragments()), which is stricter than "the whole value
    appears verbatim" -- a partial or reformatted repeat the passes above
    missed would otherwise sail through masking and still trip the guard,
    silently dropping a legitimate scanner-found candidate from the results
    (see LeakGuard.check()).

    Returns None when this candidate's secret turned out not to be in the
    window at all and nothing was touched -- callers pass in every
    candidate from the file, so "not present here" is the common case and
    must not produce a metadata block describing a secret the model cannot
    see."""
    original = dict(lines_by_number)
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

    # Credential components are registered as secrets in their own right
    # (see secret_components), so scrub them at their own granularity --
    # a 9-character url password is never a FRAGMENT_LEN-character window
    # of the url it sits inside.
    fragments = fragments_of(candidate.matched_value)
    for component in secret_components(candidate.matched_value):
        fragments |= fragments_of(component)

    for line_no, text in lines_by_number.items():
        lines_by_number[line_no] = _scrub_fragments_in_line(text, fragments, line_fallback_fn)

    if lines_by_number == original:
        return None

    return MaskedSpan(
        start=candidate.line_start,
        end=candidate.line_end,
        placeholder=placeholder,
        metadata=metadata,
        is_target=is_target,
    )


def _collect_spans(
    candidates: list[Candidate],
    lines_by_number: dict[int, str],
    *,
    value_fn: Callable[[str, str], tuple[str, SecretMetadata]],
    line_fallback_fn: Callable[[str], str],
    target_id: str | None = None,
) -> list[MaskedSpan]:
    """Scrubs each candidate into the shared window and keeps a metadata
    span only for those that actually replaced something.

    target_id marks which candidate is the one being classified, so the
    prompt can say so. Matched by id, not by line range: two secrets on the
    same line share a line range, and raw treatment passes no target here
    at all (it shows the target's real value, so only neighbours are
    masked)."""
    spans = []
    for candidate in candidates:
        span = _replace_candidate_in_lines(
            candidate,
            lines_by_number,
            value_fn=value_fn,
            line_fallback_fn=line_fallback_fn,
            is_target=candidate.id == target_id,
        )
        if span is not None:
            spans.append(span)
    return spans


def build_raw_context(target: Candidate, other_candidates: list[Candidate]) -> SanitisedContext:
    """Unmasked window for target only -- research baseline, never the
    shipped default. Every other candidate sharing this window is still
    masked; RAW shows this one value, not everything nearby."""
    lines_by_number = _window_lines(target)
    others = _others_to_scrub(target, other_candidates)

    masked_spans = _collect_spans(
        others, lines_by_number, value_fn=mask_value, line_fallback_fn=_bullet_line_fallback
    )
    snippet = "\n".join(lines_by_number[n] for n in sorted(lines_by_number))

    return SanitisedContext(
        treatment=Treatment.RAW, sanitised_snippet=snippet, masked_spans=masked_spans
    )


def mask_context_window(target: Candidate, other_candidates: list[Candidate]) -> SanitisedContext:
    """Builds the masked window around `target`, masking every other
    candidate whose line falls inside it too -- otherwise +/-N lines of
    context would leak whatever else sits there."""
    lines_by_number = _window_lines(target)
    others = _others_to_scrub(target, other_candidates)

    masked_spans = _collect_spans(
        [target, *others],
        lines_by_number,
        value_fn=mask_value,
        line_fallback_fn=_bullet_line_fallback,
        target_id=target.id,
    )
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
    others = _others_to_scrub(target, other_candidates)

    masked_spans = _collect_spans(
        [target, *others],
        lines_by_number,
        value_fn=pseudonymise_value,
        line_fallback_fn=_bullet_line_fallback,
        target_id=target.id,
    )
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
    others = _others_to_scrub(target, other_candidates)

    masked_spans = _collect_spans(
        [target, *others],
        lines_by_number,
        value_fn=redact_value,
        line_fallback_fn=_redacted_line_fallback,
        target_id=target.id,
    )
    snippet = "\n".join(lines_by_number[n] for n in sorted(lines_by_number))

    return SanitisedContext(
        treatment=Treatment.METADATA_ONLY, sanitised_snippet=snippet, masked_spans=masked_spans
    )
