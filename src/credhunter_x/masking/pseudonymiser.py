from __future__ import annotations

import random
import string
from collections.abc import Callable

from credhunter_x.masking.entropy import classify_charset

_UPPER = string.ascii_uppercase
_LOWER = string.ascii_lowercase
_DIGITS = string.digits
_SYMBOLS = "-_.+/="
_ALNUM = _UPPER + _LOWER + _DIGITS
_BASE64URL = _UPPER + _LOWER + _DIGITS + "-_"

_CHARSET_POOLS: dict[str, str] = {
    "A-Z": _UPPER,
    "a-z": _LOWER,
    "0-9": _DIGITS,
    "symbols": _SYMBOLS,
}


def _random_string(rng: random.Random, length: int, alphabet: str) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def _fake_aws_access_token(rng: random.Random, _real_value: str) -> str:
    return "AKIA" + _random_string(rng, 16, _UPPER + _DIGITS)


def _fake_github_pat(rng: random.Random, _real_value: str) -> str:
    return "ghp_" + _random_string(rng, 36, _ALNUM)


def _fake_slack_bot_token(rng: random.Random, _real_value: str) -> str:
    team_id = _random_string(rng, 12, _DIGITS)
    bot_id = _random_string(rng, 13, _DIGITS)
    token = _random_string(rng, 24, _ALNUM)
    return f"xoxb-{team_id}-{bot_id}-{token}"


def _fake_gcp_api_key(rng: random.Random, _real_value: str) -> str:
    return "AIza" + _random_string(rng, 35, _ALNUM + "_-")


def _fake_jwt(rng: random.Random, _real_value: str) -> str:
    header = _random_string(rng, 36, _BASE64URL)
    payload = _random_string(rng, 64, _BASE64URL)
    signature = _random_string(rng, 43, _BASE64URL)
    return f"{header}.{payload}.{signature}"


# Only rule types with a well-known, fixed canonical shape get a
# format-correct fake generator -- the fake is that shape's real length,
# not the real value's length, since e.g. every genuine AWS access key is
# 20 characters regardless of what this particular one happened to be.
_FAKE_GENERATORS: dict[str, Callable[[random.Random, str], str]] = {
    "aws-access-token": _fake_aws_access_token,
    "github-pat": _fake_github_pat,
    "slack-bot-token": _fake_slack_bot_token,
    "gcp-api-key": _fake_gcp_api_key,
    "jwt": _fake_jwt,
}


def _fake_from_charset(rng: random.Random, real_value: str) -> str:
    """Fallback for rule types with no fixed canonical shape (private-key,
    generic-api-key, anything unrecognised): a same-length, same-character-
    class random string -- mirrors CredData's own third obfuscation method
    ("random string generated matching the original's character set and
    length"). Never touches real_value's actual characters, only its
    length and character classes via classify_charset()."""
    charset = classify_charset(real_value)
    pools = [_CHARSET_POOLS[part] for part in ("A-Z", "a-z", "0-9", "symbols") if part in charset]
    alphabet = "".join(pools) or _ALNUM
    return _random_string(rng, len(real_value), alphabet)


def generate_fake_value(real_value: str, rule_id: str, *, rng: random.Random | None = None) -> str:
    """A fake-but-realistic same-shape replacement for real_value, used by
    the pseudonymised treatment: the model sees text that looks like a real
    credential of this type, but nothing real. rng defaults to a fresh,
    unseeded generator -- production runs should not be predictable; tests
    inject a seeded random.Random for determinism."""
    rng = rng or random.Random()
    generator = _FAKE_GENERATORS.get(rule_id, _fake_from_charset)
    return generator(rng, real_value)
