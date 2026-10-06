from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import unquote

from credhunter_x.models.candidate import Candidate

# 16 characters is long enough not to match unrelated text by chance.
# Shorter values are matched whole.
_FRAGMENT_LEN = 16
_PEM_MARKER_RE = re.compile(r"^-----(BEGIN|END) [^-]+-----$")

# A matched value isn't always all secret: password-in-url reports the
# whole url, while the credential is only part of it. The split below is by
# delimiter, so it also covers connection strings, .env pairs, JSON and query
# strings.
_URL_CREDENTIAL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://(?P<cred>[^/@\s]*)@")
_KEY_VALUE_RE = re.compile(
    r"""["']?(?P<name>[A-Za-z_][A-Za-z0-9_.\-]*)["']?\s*[:=]\s*["']?(?P<value>[^"'&;,\s]+)"""
)
_SECRET_NAME_RE = re.compile(r"pass|pwd|secret|token|key|credential|auth|sig", re.I)
# Field separators common to credential formats. "." is left out because
# it splits hostnames and version numbers.
_COMPONENT_DELIMITERS = re.compile(r"""[:@/;=&?,|\\<>\s"']+""")
# A short bare word or number could just as well be an identifier. "_" and
# "." don't count as symbols, or names like "client_secret" would be masked.
_STRONG_SYMBOL_RE = re.compile(r"[^A-Za-z0-9_.]")
_MIN_ALNUM_COMPONENT_LEN = 8
_MIN_SYMBOLIC_COMPONENT_LEN = 4
# When the format says a token is the credential (a url password, or the
# value of a key like "password"), length alone is enough, though purely
# alphabetic tokens are still excluded.
_MIN_KNOWN_CREDENTIAL_LEN = 4


def _is_distinctive_enough(component: str, *, known_credential: bool = False) -> bool:
    """Is component specific enough to blank everywhere it appears?

    Masking is global within the window, so registering "admin" or "test" would
    blank those words everywhere and hide the context the classifier needs.
    Purely alphabetic tokens shorter than _FRAGMENT_LEN are never registered,
    and identifier-like tokens only when the format marks them as the credential."""
    if len(component) >= _FRAGMENT_LEN:
        return True
    if component.isalpha():
        return False
    if known_credential:
        return len(component) >= _MIN_KNOWN_CREDENTIAL_LEN
    if component.isalnum():
        return len(component) >= _MIN_ALNUM_COMPONENT_LEN
    if _STRONG_SYMBOL_RE.search(component):
        return len(component) >= _MIN_SYMBOLIC_COMPONENT_LEN
    return False


def _forms_of(token: str) -> set[str]:
    """token plus its percent-decoded form (Berners-Lee et al., 2005). A file
    often asserts the decoded password a line later, and that form isn't a
    substring of the encoded value."""
    forms = {token, unquote(token)}
    return {stripped for stripped in (form.strip() for form in forms) if stripped}


def _known_credentials(value: str) -> set[str]:
    """Substrings the format marks as the credential: a url's password, and
    the value of any key=value or "key": "value" pair whose key looks like a
    secret name."""
    found: set[str] = set()

    url_match = _URL_CREDENTIAL_RE.match(value.strip().strip("\"'"))
    if url_match is not None:
        cred = url_match.group("cred")
        if ":" in cred:
            found.add(cred.split(":", 1)[1])

    for pair in _KEY_VALUE_RE.finditer(value):
        if _SECRET_NAME_RE.search(pair.group("name")):
            found.add(pair.group("value"))

    return {stripped for stripped in (item.strip() for item in found) if stripped}


def secret_components(value: str) -> set[str]:
    """The secret parts inside a scanner's matched value, registered as
    secrets alongside the whole value.

    A credential inside a url or connection string can be shorter than
    FRAGMENT_LEN, so no fragment of the whole value matches it. Registering it
    separately closes that gap without changing _FRAGMENT_LEN. Tokens the format
    identifies get a lower bar than tokens from a plain split."""
    components: set[str] = set()

    for token in _known_credentials(value):
        for form in _forms_of(token):
            if _is_distinctive_enough(form, known_credential=True):
                components.add(form)

    for token in _COMPONENT_DELIMITERS.split(value):
        for form in _forms_of(token):
            if _is_distinctive_enough(form):
                components.add(form)

    # A component that is long enough and already a substring of the parent
    # is covered by the parent's fragments, so skip it (this stops a PEM key
    # adding a near-duplicate set per line).
    return {c for c in components if len(c) < _FRAGMENT_LEN or c not in value}


def _strip_pem_boilerplate(value: str) -> str:
    """PEM header and footer lines are public text, so they are left out of
    fragments(). value_for() still returns the full value."""
    lines = value.split("\n")
    kept = [line for line in lines if not _PEM_MARKER_RE.match(line.strip())]
    return "\n".join(kept)


def fragments_of(value: str) -> set[str]:
    """All FRAGMENT_LEN-character substrings of value, without PEM
    boilerplate; shorter values are returned whole. Used by both the guard and
    the masker so they work at the same granularity."""
    stripped = _strip_pem_boilerplate(value)
    if not stripped:
        return set()
    if len(stripped) < _FRAGMENT_LEN:
        return {stripped}
    return {stripped[i : i + _FRAGMENT_LEN] for i in range(len(stripped) - _FRAGMENT_LEN + 1)}


@dataclass
class SecretRegistry:
    """Holds every known secret in a batch, not just the current candidate,
    keyed by candidate id so the raw-mode permit can find a candidate's own
    value."""

    _values_by_candidate: dict[str, str] = field(default_factory=dict)
    _components_by_candidate: dict[str, set[str]] = field(default_factory=dict)

    def register(self, candidate_id: str, value: str) -> None:
        if value:
            self._values_by_candidate[candidate_id] = value
            self._components_by_candidate[candidate_id] = secret_components(value)

    def register_candidates(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            self.register(candidate.id, candidate.matched_value)

    def value_for(self, candidate_id: str) -> str | None:
        return self._values_by_candidate.get(candidate_id)

    def values_for(self, candidate_id: str) -> set[str]:
        """Every secret string of this candidate: the whole value plus any
        components (see secret_components). Raw treatment permits all of them, since
        a decoded password isn't a substring of its encoded url."""
        value = self._values_by_candidate.get(candidate_id)
        if value is None:
            return set()
        return {value} | self._components_by_candidate.get(candidate_id, set())

    def fragments(self) -> set[str]:
        """All FRAGMENT_LEN-character substrings of every registered value and
        component, without PEM boilerplate. Shorter values are kept whole."""
        result: set[str] = set()
        for candidate_id, value in self._values_by_candidate.items():
            result |= fragments_of(value)
            for component in self._components_by_candidate.get(candidate_id, set()):
                result |= fragments_of(component)
        return result

    def __len__(self) -> int:
        return len(self._values_by_candidate)

    def __bool__(self) -> bool:
        return bool(self._values_by_candidate)
