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
    """Adversarial case: even a partial leak must be caught — 8 contiguous
    characters of a real secret is already enough to meaningfully narrow it
    down, so a fragment match is treated exactly as seriously as a full one."""
    guard = make_guard({"c1": GITHUB_SECRET})
    sneaky_payload = "some text ...Pw5k4aXc... more text, nothing to see here"
    assert "Pw5k4aXc" in GITHUB_SECRET  # sanity check on the test itself
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
    """The named-permit escape hatch must be scoped to exactly one
    candidate — a raw-mode call for c1 must still trip if c2's secret has
    leaked into the same context window."""
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
