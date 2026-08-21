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
    """Masks every occurrence of any candidate's real value found anywhere
    in arbitrary text — for Arm B's tool results, which can pull in content
    from any file/location, not just a candidate's own precomputed context
    window. There's no "target vs bystander" distinction for tool output —
    every known secret is masked regardless of whose window it came from.

    Matches at the fragment level (the same FRAGMENT_LEN-character
    substrings SecretRegistry.fragments() generates, and LeakGuard.check()
    itself scans for) rather than requiring an exact whole-value match. A
    tool result is often a small snippet (e.g. search_file's +/-2 lines of
    context), which for a multi-line secret like a PEM private key almost
    never contains the *entire* value — a whole-value-only replace would
    silently miss that snippet and leave real key bytes in the payload,
    with nothing catching it until (and unless) the guard's own fragment
    check aborts the call outright. Matching fragments are merged into
    contiguous spans before replacement so a longer leaked run is blanked
    out in full, not just its first FRAGMENT_LEN characters."""
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
    """type_hint should come from something already-public about the
    finding (e.g. the scanner's rule_id), never sliced from the secret
    value itself: only a handful of credential types have a prefix that's
    genuinely constant across every instance of that type (safe to
    reveal), and hardcoding that list isn't worth the risk of getting it
    wrong for MVP."""
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
    """Replaces `value` with a fake-but-realistic same-shape value — the
    model sees text that looks like a real credential, but nothing real.
    See masking/pseudonymiser.py for the generation strategy."""
    return generate_fake_value(value, type_hint), _compute_metadata(value, type_hint)


def redact_value(value: str, type_hint: str = "") -> tuple[str, SecretMetadata]:
    """Replaces `value` with a short fixed marker carrying NO length
    signal — unlike mask_value's bullet-fill, which deliberately reveals
    length via placeholder length. metadata_only's whole point is that the
    only signal available is the explicit metadata line, not anything
    inferable from the snippet itself."""
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
    window — not just within its own reported line range, since the
    identical value can legitimately recur elsewhere nearby (e.g. a test
    assertion re-checking a value gitleaks only flagged once, at its first
    occurrence). Tries an exact substring replace first (preserves
    surrounding code on each line). If that finds no match anywhere —
    matched_value spans multiple lines and can never be "in" a single
    line's text, or gitleaks reported a normalised value (e.g. URL-decoded)
    that doesn't literally appear in the source — falls back to
    line_fallback_fn over candidate's own line range outright, rather than
    silently leaving real content unreplaced.

    value_fn/line_fallback_fn are parameterised (rather than this function
    hardcoding mask_value + a bullet-fill) so every treatment gets its own
    correct fallback: a length-matched bullet fill is fine for masked, but
    would leak length exactly where metadata_only promises not to."""
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


def build_pseudonymised_context(
    target: Candidate, other_candidates: list[Candidate]
) -> SanitisedContext:
    """Same window-scoping as mask_context_window, but every value (target
    and bystanders) is replaced with a fake-but-realistic same-shape value
    instead of a bullet-fill — see masking/pseudonymiser.py.

    Known, deliberate limitation: for a multi-line matched_value (private
    keys), the exact-match replace can never succeed (see
    _replace_candidate_in_lines), so this silently falls back to
    _bullet_line_fallback — the same bullet-masking mask_context_window
    uses — rather than attempting to synthesise a realistic multi-line PEM
    block, which is out of scope for MVP. That candidate's window still
    ends up fully sanitised, just not pseudonymised the way a single-line
    secret's would be."""
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
    """Same window-scoping as mask_context_window, but every value (target
    and bystanders) is replaced with a short fixed marker carrying no
    length signal, instead of a length-matched bullet-fill — the model's
    only available signal is the metadata line appended by
    llm/prompts.py's _context_sections, never anything inferable from the
    snippet itself."""
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
