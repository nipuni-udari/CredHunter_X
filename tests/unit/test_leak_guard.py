from __future__ import annotations

import pytest

from credhunter_x.guard.errors import LeakError
from credhunter_x.guard.leak_guard import LeakGuard
from credhunter_x.masking.secret_registry import SecretRegistry

GITHUB_SECRET = "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"
SLACK_SECRET = "xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"


def make_guard(secrets: dict[str, str]) -> LeakGuard:
    registry = SecretRegistry()
    for candidate_id, value in secrets.items():
        registry.register(candidate_id, value)
    return LeakGuard(registry)


def test_raises_when_full_secret_appears_in_payload():
    guard = make_guard({"c1": GITHUB_SECRET})
    with pytest.raises(LeakError):
        guard.check(f"here is the token: {GITHUB_SECRET}", candidate_id="c1")


def test_raises_when_only_a_partial_fragment_appears_in_payload():
    """A partial leak is caught too: 16 characters of a secret are enough to
    narrow it down."""
    guard = make_guard({"c1": GITHUB_SECRET})
    sneaky_payload = "some text ...Pw5k4aXcaT4fNP0U... more text, nothing to see here"
    assert "Pw5k4aXcaT4fNP0U" in GITHUB_SECRET  # sanity check on the test itself
    with pytest.raises(LeakError):
        guard.check(sneaky_payload, candidate_id="c1")


def test_does_not_raise_for_a_clean_masked_payload():
    guard = make_guard({"c1": GITHUB_SECRET})
    guard.check('GITHUB_TOKEN = "' + "•" * len(GITHUB_SECRET) + '"', candidate_id="c1")


def test_fails_closed_when_registry_is_empty():
    guard = LeakGuard(SecretRegistry())
    with pytest.raises(LeakError):
        guard.check("perfectly innocent text with no secrets at all")


def test_raw_permit_allows_only_that_candidates_own_value():
    guard = make_guard({"c1": GITHUB_SECRET, "c2": SLACK_SECRET})
    guard.check(
        f"sending real value: {GITHUB_SECRET}",
        candidate_id="c1",
        raw_permit_candidate_id="c1",
    )


def test_raw_permit_does_not_excuse_a_different_candidates_secret():
    """The permit covers exactly one candidate: a raw call for c1 must still
    fail if c2's secret is in the window."""
    guard = make_guard({"c1": GITHUB_SECRET, "c2": SLACK_SECRET})
    payload = f"sending: {GITHUB_SECRET} and also accidentally: {SLACK_SECRET}"
    with pytest.raises(LeakError):
        guard.check(payload, candidate_id="c1", raw_permit_candidate_id="c1")


def test_raw_permit_for_wrong_candidate_id_does_not_excuse_anything():
    guard = make_guard({"c1": GITHUB_SECRET})
    with pytest.raises(LeakError):
        guard.check(
            f"sending: {GITHUB_SECRET}",
            candidate_id="c1",
            raw_permit_candidate_id="nonexistent-id",
        )


# A modulus longer than _FRAGMENT_LEN, reused in a private key and its
# public key/certificate, standing in for the overlap in a real key pair.
_SHARED_MODULUS = "Xk29fQpL7mZs4Wn8Rt5Vc3Yb6Hj1Gd0AeKf7Nq2Ms9Pw4Tz"


def _pem_block(header: str, body: str = _SHARED_MODULUS) -> str:
    return f"-----BEGIN {header}-----\n{body}\n-----END {header}-----"


_PRIVATE_KEY_PEM = _pem_block("RSA PRIVATE KEY", f"{_SHARED_MODULUS}\nSecretExtra")


def test_fragment_inside_a_public_key_block_is_excused():
    guard = make_guard({"c1": _PRIVATE_KEY_PEM})
    payload = "here is the matching cert:\n" + _pem_block("CERTIFICATE")
    guard.check(payload, candidate_id="c1")  # must not raise


def test_fragment_inside_a_public_key_variant_block_is_excused():
    guard = make_guard({"c1": _PRIVATE_KEY_PEM})
    payload = _pem_block("RSA PUBLIC KEY")
    guard.check(payload, candidate_id="c1")  # must not raise


def test_fragment_outside_any_safe_block_still_trips_even_with_a_safe_block_present():
    """A payload can hold a real certificate and, elsewhere, a leak. The safe
    block must not excuse the whole payload."""
    guard = make_guard({"c1": _PRIVATE_KEY_PEM})
    payload = _pem_block("CERTIFICATE") + f"\noops, leaked again: {_SHARED_MODULUS}"
    with pytest.raises(LeakError):
        guard.check(payload, candidate_id="c1")


def test_a_second_private_key_sharing_fragments_still_trips_the_guard():
    """A block labelled as a private key is never excused, even if it looks
    like paired key material. That is the leak the guard is for."""
    guard = make_guard({"c1": _PRIVATE_KEY_PEM})
    payload = _pem_block("RSA PRIVATE KEY", f"{_SHARED_MODULUS}\nOtherExtra")
    with pytest.raises(LeakError):
        guard.check(payload, candidate_id="c1")


def test_mismatched_begin_end_pem_markers_are_not_treated_as_a_safe_block():
    """A BEGIN PUBLIC KEY with no matching END must not create a safe span
    that swallows the rest of the payload."""
    guard = make_guard({"c1": _PRIVATE_KEY_PEM})
    mismatched = "-----BEGIN PUBLIC KEY-----\nnot the real body\n-----END CERTIFICATE-----"
    payload = f"{mismatched}\n{_SHARED_MODULUS}"
    with pytest.raises(LeakError):
        guard.check(payload, candidate_id="c1")
