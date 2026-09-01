from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import unquote

from credhunter_x.models.candidate import Candidate

# 8 was short enough to collide with unrelated boilerplate by chance; 16
# consecutive matching characters isn't. Shorter values are still matched
# whole.
_FRAGMENT_LEN = 16
_PEM_MARKER_RE = re.compile(r"^-----(BEGIN|END) [^-]+-----$")

# A scanner's matched value isn't always all-secret. gitleaks'
# password-in-url rule reports the WHOLE url -- scheme, host and path
# included -- so the credential inside it is only a few of those
# characters. Registering just the url means a file that also asserts the
# bare password on its own line ("assert cfg['passwd'] == 'pwd%23%20'")
# leaks it: no FRAGMENT_LEN-character window of the full url equals that
# short password, so neither masking nor the guard ever sees it. The same
# shape occurs in connection strings, .env pairs, JSON blobs and query
# strings, so the split below is by delimiter, not by url grammar.
_URL_CREDENTIAL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://(?P<cred>[^/@\s]*)@")
_KEY_VALUE_RE = re.compile(
    r"""["']?(?P<name>[A-Za-z_][A-Za-z0-9_.\-]*)["']?\s*[:=]\s*["']?(?P<value>[^"'&;,\s]+)"""
)
_SECRET_NAME_RE = re.compile(r"pass|pwd|secret|token|key|credential|auth|sig", re.I)
# Field separators shared by every credential-bearing format we see. "." is
# deliberately absent: it splits hostnames and version numbers, and the
# formats that use it (JWT) have segments long enough to be covered anyway.
_COMPONENT_DELIMITERS = re.compile(r"""[:@/;=&?,|\\<>\s"']+""")
# A bare word or digit run this short is as likely to be an identifier as a
# password. "_" and "." are excluded from what counts as a distinguishing
# symbol: they are identifier and hostname punctuation, so treating them as
# secret-ish registers names like "client_secret", "DB_PASSWORD" and
# "api.io" and blanks those wherever they appear in the code.
_STRONG_SYMBOL_RE = re.compile(r"[^A-Za-z0-9_.]")
_MIN_ALNUM_COMPONENT_LEN = 8
_MIN_SYMBOLIC_COMPONENT_LEN = 4
# Where the format tells us a token IS the credential (a url's password
# field, or a value whose key is named "password"/"secret"/...), it is far
# more likely to be that password than a stray identifier, so the bar drops
# to length alone -- the purely-alphabetic exclusion below still applies.
_MIN_KNOWN_CREDENTIAL_LEN = 4


def _is_distinctive_enough(component: str, *, known_credential: bool = False) -> bool:
    """Is `component` specific enough to blank everywhere it appears?

    This is the whole risk of registering sub-values: masking is global
    within the window, so registering a password of "admin" or "test"
    would blank those words wherever they occur and destroy the variable
    names and file context the classifier judges from. A purely alphabetic
    token is therefore never registered below _FRAGMENT_LEN, however
    confident we are that it's the password -- "_hide_password" is a real
    function name in the corpus, and losing it costs the classifier more
    than the word itself is worth. Identifier-shaped tokens (letters,
    digits, "_" and "." only) are held to the same standard unless the
    format positively identifies them as the credential."""
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
    """`token` plus its percent-decoded form -- a file hardcoding an
    encoded password commonly asserts the decoded one a line or two later,
    and the decoded form is not a substring of the encoded value, so
    nothing else would ever catch it."""
    forms = {token, unquote(token)}
    return {stripped for stripped in (form.strip() for form in forms) if stripped}


def _known_credentials(value: str) -> set[str]:
    """Substrings the format positively identifies as the credential: a
    url's password field, and any key=value / "key": "value" pair whose key
    is named like a secret. Covers connection strings, .env lines, JSON
    blobs and query strings without needing a parser for each."""
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
    """The genuinely-secret parts *inside* a scanner's matched value, to be
    registered as secrets in their own right alongside the whole value.

    A scanner reports one blob -- a whole url, a whole connection string --
    but the credential inside it can be far shorter than FRAGMENT_LEN, and
    no FRAGMENT_LEN-character window of the blob ever equals it. A file
    that also uses that bare credential on its own line therefore leaked it
    past both masking and the guard, which share that granularity.
    Registering the component closes the hole without weakening
    _FRAGMENT_LEN globally.

    Two tiers, because confidence differs: tokens the format identifies as
    the credential clear a lower bar than tokens that merely fell out of a
    delimiter split."""
    components: set[str] = set()

    for token in _known_credentials(value):
        for form in _forms_of(token):
            if _is_distinctive_enough(form, known_credential=True):
                components.add(form)

    for token in _COMPONENT_DELIMITERS.split(value):
        for form in _forms_of(token):
            if _is_distinctive_enough(form):
                components.add(form)

    # A component that's both long enough to be a fragment and a literal
    # substring of the parent is already covered by the parent's own
    # fragments. Dropping those keeps a multi-line PEM key from adding one
    # near-duplicate fragment set per base64 line.
    return {c for c in components if len(c) < _FRAGMENT_LEN or c not in value}


def _strip_pem_boilerplate(value: str) -> str:
    """PEM header/footer lines are public format text, not secret
    material -- left in fragments(), they'd trip the guard on content
    that was never sensitive. Stripped only here; value_for() still
    returns the full original value."""
    lines = value.split("\n")
    kept = [line for line in lines if not _PEM_MARKER_RE.match(line.strip())]
    return "\n".join(kept)


def fragments_of(value: str) -> set[str]:
    """All contiguous FRAGMENT_LEN-character substrings of `value`'s
    non-boilerplate content -- shorter values are matched whole. Shared
    by SecretRegistry.fragments() (what LeakGuard checks payloads
    against) and masker.py's window-scrub, so both operate at the exact
    same granularity -- masking can't leave behind a partial repeat the
    guard would still object to."""
    stripped = _strip_pem_boilerplate(value)
    if not stripped:
        return set()
    if len(stripped) < _FRAGMENT_LEN:
        return {stripped}
    return {stripped[i : i + _FRAGMENT_LEN] for i in range(len(stripped) - _FRAGMENT_LEN + 1)}


@dataclass
class SecretRegistry:
    """Holds every known real secret value across a batch, not just the
    current candidate -- the guard checks payloads against all of it.
    Keyed by candidate id so raw-mode permit can look up which value a
    specific candidate owns."""

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
        """Every secret string belonging to this candidate -- its whole
        matched value plus any credential component extracted from it (see
        secret_components). Raw treatment permits a candidate's own secret,
        and a component is just as much "its own" as the full value: a
        decoded password is not a substring of the encoded url it came
        from, so permitting only the full value would make the guard block
        raw calls over the candidate's own secret."""
        value = self._values_by_candidate.get(candidate_id)
        if value is None:
            return set()
        return {value} | self._components_by_candidate.get(candidate_id, set())

    def fragments(self) -> set[str]:
        """All contiguous FRAGMENT_LEN-character substrings of every
        registered value's non-boilerplate content, plus the same for each
        registered credential component. Shorter values are registered
        whole."""
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
