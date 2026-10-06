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
    """Masks every candidate's value anywhere in arbitrary text, used for Arm
    B's tool results, which can come from anywhere in the repo. Matches at
    fragment level, like the guard, because a snippet often holds only part of
    a secret such as a PEM key. Matches are merged before replacing."""
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
    """Fully masks value; no real characters are shown."""
    return _MASK_CHAR * len(value), _compute_metadata(value, type_hint)


def pseudonymise_value(value: str, type_hint: str = "") -> tuple[str, SecretMetadata]:
    """Replaces `value` with a fake-but-realistic same-shape value. See
    pseudonymiser.py for the generation strategy."""
    return generate_fake_value(value, type_hint), _compute_metadata(value, type_hint)


def redact_value(value: str, type_hint: str = "") -> tuple[str, SecretMetadata]:
    """Replaces value with a short fixed marker that, unlike mask_value's
    bullets, doesn't reveal the length."""
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
    """Every other known secret in the same file, not only those inside the
    window's line range, since a value can also appear away from its reported
    line. Only spans that actually replaced something get metadata."""
    return [c for c in other_candidates if c.file_path == target.file_path and c.id != target.id]


def _bullet_line_fallback(line_text: str) -> str:
    return _MASK_CHAR * len(line_text)


def _redacted_line_fallback(line_text: str) -> str:
    del line_text
    return _REDACTED_MARKER


def _scrub_fragments_in_line(
    text: str, fragments: set[str], line_fallback_fn: Callable[[str], str]
) -> str:
    """Blanks every fragment in text in one pass. Fragments overlap, so all
    positions are found on the original text and merged before anything is
    replaced (the same method as mask_arbitrary_text)."""
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
    """Replaces every occurrence of candidate's value in the window, falling back
    to line_fallback_fn on its own lines when there is no exact match. Any
    leftover FRAGMENT_LEN piece is then scrubbed, at the guard's granularity.
    Returns None if the value isn't in the window, so no metadata is made for it."""
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

    # Components (see secret_components) are scrubbed at their own size; a
    # short url password is never a FRAGMENT_LEN window of the url.
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
    """Scrubs each candidate from the shared window and keeps a metadata span
    only for those that replaced something. target_id marks the candidate being
    classified, matched by id because two secrets on one line share a line range."""
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
    """Unmasked window for target only: a research baseline, never the
    default. Every other candidate in the window is still masked."""
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
    """Builds the masked window around target, also masking any other
    candidate inside the +/-N lines."""
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
    """Like mask_context_window, but each value is replaced with a realistic
    fake of the same shape (see pseudonymiser.py). Multi-line values such as
    private keys fall back to bullet masking."""
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
    """Like mask_context_window, but each value becomes a short fixed marker
    that doesn't reveal the length."""
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
