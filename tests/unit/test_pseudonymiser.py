from __future__ import annotations

import random
import re

from credhunter_x.masking.pseudonymiser import generate_fake_value

GITHUB_SECRET = "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"


def test_aws_access_token_matches_the_real_format():
    fake = generate_fake_value("AKIAIOSFODNN7EXAMPLE", "aws-access-token", rng=random.Random(1))
    assert re.fullmatch(r"AKIA[A-Z0-9]{16}", fake)
    assert fake != "AKIAIOSFODNN7EXAMPLE"


def test_github_pat_matches_the_real_format_and_is_not_the_real_value():
    fake = generate_fake_value(GITHUB_SECRET, "github-pat", rng=random.Random(2))
    assert re.fullmatch(r"ghp_[A-Za-z0-9]{36}", fake)
    assert fake != GITHUB_SECRET


def test_slack_bot_token_matches_the_real_format():
    real = "xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"
    fake = generate_fake_value(real, "slack-bot-token", rng=random.Random(3))
    assert re.fullmatch(r"xoxb-\d{12}-\d{13}-[A-Za-z0-9]{24}", fake)
    assert fake != real


def test_gcp_api_key_matches_the_real_format():
    real = "AIzaSyDaGmWKa4JsXZ-HjGw7ISLan_Mgg5AN7EW"
    fake = generate_fake_value(real, "gcp-api-key", rng=random.Random(4))
    assert re.fullmatch(r"AIza[A-Za-z0-9_-]{35}", fake)


def test_jwt_matches_the_three_segment_shape():
    fake = generate_fake_value("header.payload.signature", "jwt", rng=random.Random(5))
    assert re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", fake)


def test_unknown_rule_id_falls_back_to_same_length_same_charset():
    real = "S0meR4ndomP4ssw0rd!2024xyz"
    fake = generate_fake_value(real, "generic-api-key", rng=random.Random(6))
    assert len(fake) == len(real)
    assert fake != real
    assert any(c.isupper() for c in fake)
    assert any(c.islower() for c in fake)
    assert any(c.isdigit() for c in fake)
    assert any(not c.isalnum() for c in fake)


def test_fallback_never_uses_a_character_class_absent_from_the_real_value():
    real = "12345678"  # digits only
    fake = generate_fake_value(real, "generic-api-key", rng=random.Random(7))
    assert len(fake) == len(real)
    assert all(c.isdigit() for c in fake)


def test_same_seed_is_deterministic():
    a = generate_fake_value(GITHUB_SECRET, "github-pat", rng=random.Random(42))
    b = generate_fake_value(GITHUB_SECRET, "github-pat", rng=random.Random(42))
    assert a == b


def test_unseeded_calls_are_not_forced_into_a_fixed_value():
    # not a strict guarantee (collisions are astronomically unlikely, not
    # impossible) -- just pins that no default seed is silently applied
    values = {generate_fake_value(GITHUB_SECRET, "github-pat") for _ in range(5)}
    assert len(values) > 1
