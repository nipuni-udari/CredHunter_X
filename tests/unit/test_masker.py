from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from credhunter_x.gitleaks.parser import parse_gitleaks_report
from credhunter_x.guard.leak_guard import LeakGuard
from credhunter_x.masking.masker import (
    _bullet_line_fallback,
    _scrub_fragments_in_line,
    build_metadata_only_context,
    build_pseudonymised_context,
    build_raw_context,
    mask_arbitrary_text,
    mask_context_window,
    mask_value,
    pseudonymise_value,
    redact_value,
)
from credhunter_x.masking.secret_registry import SecretRegistry, fragments_of, secret_components
from credhunter_x.models.candidate import Candidate
from credhunter_x.models.treatment import Treatment

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_REPO = FIXTURES_DIR / "sample_repo"
REPORT_PATH = FIXTURES_DIR / "gitleaks_report.json"

GITHUB_SECRET = "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"
SLACK_SECRET = "xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"


def load_candidates(context_lines: int = 10):
    findings = json.loads(REPORT_PATH.read_text())
    return parse_gitleaks_report(
        findings, SAMPLE_REPO, repo_id="test-repo", context_lines=context_lines
    )


def test_mask_value_never_reveals_real_characters():
    placeholder, metadata = mask_value(GITHUB_SECRET, type_hint="github-pat")
    assert len(placeholder) == len(GITHUB_SECRET)
    assert set(placeholder) == {"•"}
    assert metadata.length == len(GITHUB_SECRET)
    assert metadata.prefix_hint == "github-pat"
    assert metadata.entropy > 0


def test_mask_value_hint_is_never_sliced_from_the_real_value():
    value = "S0meR4ndomP4ssw0rd!2024xyz"
    _, metadata = mask_value(value, type_hint="generic-password")
    assert metadata.prefix_hint == "generic-password"
    assert value[:4] not in metadata.prefix_hint


def test_mask_context_window_masks_targets_own_secret():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    others = [c for c in candidates if c.id != github.id]

    result = mask_context_window(github, others)

    assert result.treatment == Treatment.MASKED
    assert GITHUB_SECRET not in result.sanitised_snippet
    assert "•" in result.sanitised_snippet


def test_mask_context_window_also_masks_other_secrets_in_the_same_window():
    """The specific bug the design calls out: a naive implementation only
    masks the flagged candidate's own line and leaves other secrets in the
    surrounding context exposed. Both real secrets here are on adjacent
    lines, well within the default context window."""
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    result = mask_context_window(github, [slack])

    assert GITHUB_SECRET not in result.sanitised_snippet
    assert SLACK_SECRET not in result.sanitised_snippet
    assert len(result.masked_spans) == 2


def test_mask_context_window_excludes_candidates_outside_the_window():
    candidates = load_candidates(context_lines=0)
    github = next(c for c in candidates if c.rule_id == "github-pat")
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    # context_lines=0: github's own window is just its own line, so slack's
    # line (the very next one) never enters the window at all
    result = mask_context_window(github, [slack])

    assert len(result.masked_spans) == 1
    assert result.masked_spans[0].start == github.line_start


def test_mask_context_window_ignores_candidates_from_a_different_file():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")
    unrelated = replace(slack, file_path="app/other.py")

    result = mask_context_window(github, [unrelated])
    assert len(result.masked_spans) == 1


def test_mask_context_window_scrubs_a_leftover_fragment_elsewhere_in_the_window():
    """The real bug this fix closes: the whole-value replace above only
    catches an exact repeat of the *entire* secret. A partial repeat --
    the same value reformatted or embedded in different surrounding text
    elsewhere in the window -- can still contain a FRAGMENT_LEN-character
    run of the real secret, which is exactly the granularity LeakGuard
    checks payloads at (see SecretRegistry.fragments()). Left unscrubbed,
    that fragment would sail through masking untouched, then trip the
    guard on the very next call and get a legitimate scanner-found
    candidate silently dropped from results (skip_candidate_on_error)."""
    fragment = GITHUB_SECRET[4:24]  # 20 chars, well over FRAGMENT_LEN (16)
    candidate = Candidate(
        id="c1",
        file_path="app/config.py",
        line_start=5,
        line_end=5,
        rule_id="github-pat",
        matched_value=GITHUB_SECRET,
        value_start=0,
        value_end=len(GITHUB_SECRET),
        entropy=4.0,
        matched_lines=[f'TOKEN = "{GITHUB_SECRET}"'],
        context_before=[],
        # Not a verbatim repeat of the whole value, so the whole-value
        # replace never fires here -- but it does share a long enough
        # run of the real secret to count as a fragment leak.
        context_after=[f"# backup ref: xxx-{fragment}-yyy"],
        repo_id="test-repo",
    )

    result = mask_context_window(candidate, [])

    assert GITHUB_SECRET not in result.sanitised_snippet
    assert fragment not in result.sanitised_snippet

    registry = SecretRegistry()
    registry.register(candidate.id, candidate.matched_value)
    guard = LeakGuard(registry)
    guard.check(result.sanitised_snippet, candidate_id=candidate.id)  # must not raise


def _fragment_leak_candidate(fragment: str) -> Candidate:
    return Candidate(
        id="c1",
        file_path="app/config.py",
        line_start=5,
        line_end=5,
        rule_id="github-pat",
        matched_value=GITHUB_SECRET,
        value_start=0,
        value_end=len(GITHUB_SECRET),
        entropy=4.0,
        matched_lines=[f'TOKEN = "{GITHUB_SECRET}"'],
        context_before=[],
        context_after=[f"# backup ref: xxx-{fragment}-yyy"],
        repo_id="test-repo",
    )


def test_fragment_scrub_strands_no_real_secret_characters():
    """Passing LeakGuard is necessary but NOT sufficient. The guard only
    looks for whole FRAGMENT_LEN-character runs, so scrubbing fragments one
    at a time can satisfy it while still leaving shorter runs of the real
    secret in the payload -- each replacement destroys the overlap the next
    (heavily overlapping) fragment needed to match, stranding the characters
    between them. Measured against the real corpus, that leaked partial
    values on 83 of 517 candidates. Masked treatment promises no real value
    reaches the model, not merely 'no 16-character run of it'."""
    leftover = GITHUB_SECRET[4:36]  # 32 chars — two fragments' worth, overlapping
    snippet = mask_context_window(_fragment_leak_candidate(leftover), []).sanitised_snippet

    # No run of >=6 real characters may survive anywhere in the window.
    survivors = [
        GITHUB_SECRET[i : i + 6]
        for i in range(len(GITHUB_SECRET) - 5)
        if GITHUB_SECRET[i : i + 6] in snippet
    ]
    assert survivors == [], f"real secret characters left in the payload: {survivors}"


def test_fragment_scrub_is_independent_of_fragment_iteration_order():
    """fragments_of() returns a set, and Python randomises string hashing
    per process -- so a scrub whose result depends on iteration order
    produces a different prompt run to run and the evaluation stops being
    reproducible. Set order is fixed *within* one process, so feeding the
    scrub differently-ordered sequences of the same fragments is what
    actually reproduces the cross-process effect here."""
    text = f"# backup ref: xxx-{GITHUB_SECRET[4:36]}-yyy"
    fragments = sorted(fragments_of(GITHUB_SECRET))

    orderings = [
        fragments,
        list(reversed(fragments)),
        fragments[len(fragments) // 2 :] + fragments[: len(fragments) // 2],
        random.Random(7).sample(fragments, len(fragments)),
        random.Random(99).sample(fragments, len(fragments)),
    ]
    outputs = {_scrub_fragments_in_line(text, o, _bullet_line_fallback) for o in orderings}  # type: ignore[arg-type]

    assert len(outputs) == 1, f"scrub produced {len(outputs)} different outputs: {outputs}"


DB_URL = "mysql://usr:pwd%23%20@hst:123/db"


def _url_candidate() -> Candidate:
    return Candidate(
        id="u1",
        file_path="t.py",
        line_start=2,
        line_end=2,
        rule_id="generic.password-in-url",
        matched_value=DB_URL,
        value_start=17,
        value_end=17 + len(DB_URL),
        entropy=3.5,
        matched_lines=[f"    cfg = parse('{DB_URL}')"],
        context_before=["def test_connection():"],
        context_after=[
            "    assert cfg['user']   == 'usr'",
            "    assert cfg['passwd'] == 'pwd%23%20'",  # the password, on its own
            "    assert cfg['passwd'] == 'pwd# '",  # ...and its decoded form
        ],
        repo_id="test-repo",
    )


@pytest.mark.parametrize(
    ("description", "matched_value", "expected"),
    [
        (
            "url password, encoded and decoded",
            "mysql://usr:pwd%23%20@hst/db",
            {"pwd%23%20", "pwd#"},
        ),
        ("connection string", "Server=db;User=sa;Password=hunter2;", {"hunter2"}),
        ("json blob", '{"client_secret": "aB3$xY9!", "user": "bob"}', {"aB3$xY9!"}),
        ("dotenv pair", "DB_PASSWORD=s3cr3t!x", {"s3cr3t!x"}),
        ("query string", "https://api.io/v1?api_key=Zk9$mQ2&user=bob", {"Zk9$mQ2"}),
        ("underscored password", "ftp://u:my_secret_pw@h/x", {"my_secret_pw"}),
    ],
)
def test_secret_components_extracts_the_credential_from_any_delimited_format(
    description, matched_value, expected
):
    """A scanner reports one blob; the credential inside it is often far
    shorter than FRAGMENT_LEN, so no window of the blob ever equals it.
    Splitting on the field separators every credential format shares keeps
    this working beyond the one url shape the corpus happened to contain."""
    assert secret_components(matched_value) == expected, description


@pytest.mark.parametrize(
    ("description", "matched_value"),
    [
        # Blanking these would destroy the identifiers the classifier reads.
        ("password is a dictionary word", "ftp://u:password@h/x"),
        ("password is 'test'", "ftp://u:test@h/x"),
        ("key names must not become secrets", '{"client_secret": "x", "api_key": "y"}'),
        ("hostnames must not become secrets", "https://example.com/path/to/file"),
        # No delimiters: the whole value's own fragments already cover it.
        ("opaque token", "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"),
        ("aws key", "AKIAIOSFODNN7EXAMPLE"),
    ],
)
def test_secret_components_refuses_tokens_that_would_blank_ordinary_code(
    description, matched_value
):
    """The cost of registering a sub-value is that masking blanks it
    everywhere in the window. "password", "client_secret" and "example.com"
    are code, not credentials, and losing them costs the classifier more
    than hiding them gains."""
    assert secret_components(matched_value) == set(), description


def test_mask_context_window_scrubs_a_url_password_repeated_on_its_own():
    """gitleaks' password-in-url rule reports the WHOLE url, so no
    FRAGMENT_LEN-character window of it ever equals the short password
    inside. A file that also asserts that password on its own line leaked
    it -- past both masking and the guard, which share the same 16-char
    granularity. Registering the credential component closes it (9 real
    cases in the CredData corpus, one a complete password)."""
    result = mask_context_window(_url_candidate(), [])

    assert DB_URL not in result.sanitised_snippet
    assert "pwd%23%20" not in result.sanitised_snippet
    assert "pwd# " not in result.sanitised_snippet  # the decoded form too
    # The surrounding assertions are ordinary code and must survive intact.
    assert "cfg['user']" in result.sanitised_snippet
    assert "'usr'" in result.sanitised_snippet


def test_url_password_scrub_does_not_trip_the_guard_in_raw_treatment():
    """Raw treatment permits a candidate's OWN secret. A decoded password
    is not a substring of the encoded url it came from, so permitting only
    the whole matched value would make the guard block a raw call over the
    candidate's own material -- turning a leak fix into dropped results."""
    candidate = _url_candidate()
    registry = SecretRegistry()
    registry.register(candidate.id, candidate.matched_value)
    guard = LeakGuard(registry)

    snippet = build_raw_context(candidate, []).sanitised_snippet
    guard.check(snippet, candidate_id=candidate.id, raw_permit_candidate_id=candidate.id)


def test_mask_context_window_masks_a_secret_whose_value_is_in_the_window_but_line_is_not():
    """Neighbours used to be selected by line-number overlap, so a
    candidate reported at line 500 whose value is ALSO used inside this
    window went unmasked -- and the guard never sees it either, since it
    only registers the target. Selection is by file now, with presence
    decided by the text itself."""
    key = "AKIAIOSFODNN7EXAMPLE"
    target = Candidate(
        id="t",
        file_path="a.py",
        line_start=5,
        line_end=5,
        rule_id="github-pat",
        matched_value=GITHUB_SECRET,
        value_start=12,
        value_end=12 + len(GITHUB_SECRET),
        entropy=4.0,
        matched_lines=[f'    token = "{GITHUB_SECRET}"'],
        context_before=["import boto3", "", "def upload():", f'    client = client(key="{key}")'],
        context_after=["    return client"],
        repo_id="test-repo",
    )
    far_away = Candidate(
        id="aws",
        file_path="a.py",
        line_start=501,  # far outside the window, but its value is inside it
        line_end=501,
        rule_id="aws-access-token",
        matched_value=key,
        value_start=10,
        value_end=10 + len(key),
        entropy=4.0,
        matched_lines=[f'AWS_KEY = "{key}"'],
        context_before=[],
        context_after=[],
        repo_id="test-repo",
    )

    result = mask_context_window(target, [far_away])

    assert GITHUB_SECRET not in result.sanitised_snippet
    assert key not in result.sanitised_snippet


def test_a_same_file_candidate_absent_from_the_window_gets_no_metadata_span():
    """The scrub side is broad (every candidate in the file); the metadata
    side must stay narrow, or the model is handed a description of a secret
    it cannot see anywhere in its window."""
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    # A value that appears nowhere in github's window, unlike the fixture's
    # own slack token, which really does sit inside it.
    elsewhere = replace(
        next(c for c in candidates if c.rule_id == "slack-bot-token"),
        id="elsewhere",
        line_start=900,
        line_end=900,
        matched_value="AKIAZZZZZZZZZZZZNOTHERE",
    )

    result = mask_context_window(github, [elsewhere])

    assert len(result.masked_spans) == 1  # the target only
    assert result.masked_spans[0].start == github.line_start


def test_build_raw_context_leaves_target_secret_unmasked():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")

    result = build_raw_context(github, [])

    assert result.treatment == Treatment.RAW
    assert GITHUB_SECRET in result.sanitised_snippet
    assert result.masked_spans == []


def test_build_raw_context_still_masks_other_secrets_in_the_same_window():
    """RAW treatment means "reveal this one candidate's real value," not
    "expose every secret nearby" — a bystander secret sharing the window
    (here, Slack's token sits one line below GitHub's) must stay masked
    even when the target itself is sent raw."""
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    result = build_raw_context(github, [slack])

    assert GITHUB_SECRET in result.sanitised_snippet
    assert SLACK_SECRET not in result.sanitised_snippet
    assert len(result.masked_spans) == 1
    assert result.masked_spans[0].start == slack.line_start


def test_mask_arbitrary_text_masks_a_known_secret_found_anywhere():
    """Unlike the window-based masking above, this is for Arm B's tool
    results -- text that can come from anywhere in the repo, not just a
    candidate's own precomputed window."""
    candidates = load_candidates()
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    text = f"some unrelated file content...\n{SLACK_SECRET}\n...more content"
    result = mask_arbitrary_text(text, [slack])

    assert SLACK_SECRET not in result
    assert "some unrelated file content" in result
    assert "more content" in result


def test_mask_arbitrary_text_masks_multiple_occurrences_and_candidates():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    text = f"{GITHUB_SECRET} appears twice: {GITHUB_SECRET}\nand also {SLACK_SECRET}"
    result = mask_arbitrary_text(text, [github, slack])

    assert GITHUB_SECRET not in result
    assert SLACK_SECRET not in result


def test_mask_arbitrary_text_leaves_unrelated_text_untouched_when_no_secret_present():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")

    text = "just some ordinary code with no secrets in it"
    assert mask_arbitrary_text(text, [github]) == text


def test_mask_arbitrary_text_handles_an_empty_candidate_list():
    text = f"contains {GITHUB_SECRET} but nothing is registered"
    assert mask_arbitrary_text(text, []) == text


def test_mask_arbitrary_text_masks_a_truncated_snippet_of_a_multiline_secret():
    """The real bug this rewrite fixes: Arm B's search_file tool only ever
    returns a small +/-2-line snippet, which for a multi-line PEM key never
    contains the *entire* matched_value -- a whole-value-only replace would
    never fire, silently leaving real key bytes in a tool result. This
    snippet is an interior line only (no BEGIN/END markers, exactly what
    search_file would return), well short of the full key."""
    private_key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIICWgIBAAKBgQDTj1bqB4WmayWNPB+8jVSYpZYk80Ujvj680pOTh2bORBjbIAyz\n"
        "iiqU+dqBDBxBtiWQZVywgz5TcZcHG95k17GHGlpMIEortIRm22f23U/Hd6VE+7s/\n"
        "-----END RSA PRIVATE KEY-----"
    )
    key_line = "iiqU+dqBDBxBtiWQZVywgz5TcZcHG95k17GHGlpMIEortIRm22f23U/Hd6VE+7s/"
    candidate = Candidate(
        id="c1",
        file_path="app/keys.py",
        line_start=5,
        line_end=8,
        rule_id="private-key",
        matched_value=private_key,
        value_start=-1,
        value_end=-1,
        entropy=4.0,
        matched_lines=private_key.split("\n"),
        context_before=[],
        context_after=[],
        repo_id="test-repo",
    )

    snippet = f"63: {key_line}\n64: -----END RSA PRIVATE KEY-----"
    assert private_key not in snippet  # confirms this is genuinely partial

    result = mask_arbitrary_text(snippet, [candidate])

    assert key_line not in result
    assert "•" in result


def test_mask_arbitrary_text_merges_overlapping_fragment_matches_fully():
    """A leaked run longer than one FRAGMENT_LEN window must be blanked out
    completely, not just its first FRAGMENT_LEN characters -- an early
    implementation attempt (replace-one-fragment-and-stop) would have left
    a tail of real secret characters exposed."""
    candidates = load_candidates()
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    text = f"before {SLACK_SECRET} after"
    result = mask_arbitrary_text(text, [slack])

    assert SLACK_SECRET not in result
    assert SLACK_SECRET[-4:] not in result
    assert "before" in result
    assert "after" in result


def test_pseudonymise_value_replaces_with_a_same_shape_fake():
    placeholder, metadata = pseudonymise_value(GITHUB_SECRET, type_hint="github-pat")
    assert placeholder != GITHUB_SECRET
    assert placeholder.startswith("ghp_")
    assert metadata.length == len(GITHUB_SECRET)
    assert metadata.prefix_hint == "github-pat"


def test_redact_value_uses_a_fixed_marker_not_a_length_matched_one():
    placeholder, metadata = redact_value(GITHUB_SECRET, type_hint="github-pat")
    assert placeholder == "[REDACTED]"
    assert placeholder != "•" * len(GITHUB_SECRET)
    assert metadata.length == len(GITHUB_SECRET)  # length still recorded in metadata


def test_build_pseudonymised_context_masks_target_and_bystanders():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    result = build_pseudonymised_context(github, [slack])

    assert result.treatment == Treatment.PSEUDONYMISED
    assert GITHUB_SECRET not in result.sanitised_snippet
    assert SLACK_SECRET not in result.sanitised_snippet
    assert len(result.masked_spans) == 2


def test_build_pseudonymised_context_injects_a_recognisable_fake_shape():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")

    result = build_pseudonymised_context(github, [])

    assert "ghp_" in result.sanitised_snippet


def test_build_pseudonymised_context_falls_back_to_bullet_masking_for_multiline_secrets():
    """The documented limitation: a multi-line matched_value (private keys)
    can never match via exact single-line substring replace, so this
    silently falls back to the same bullet-masking mask_context_window
    uses, rather than attempting realistic multi-line PEM synthesis."""
    multiline_key = "-----BEGIN RSA PRIVATE KEY-----\nMIIB\n-----END RSA PRIVATE KEY-----"
    candidate = Candidate(
        id="c1",
        file_path="app/keys.py",
        line_start=5,
        line_end=7,
        rule_id="private-key",
        matched_value=multiline_key,
        value_start=-1,
        value_end=-1,
        entropy=4.0,
        matched_lines=multiline_key.split("\n"),
        context_before=["KEY = '''"],
        context_after=["'''"],
        repo_id="test-repo",
    )

    result = build_pseudonymised_context(candidate, [])

    assert "•" in result.sanitised_snippet
    assert "MIIB" not in result.sanitised_snippet


def test_build_metadata_only_context_masks_target_and_bystanders():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")

    result = build_metadata_only_context(github, [slack])

    assert result.treatment == Treatment.METADATA_ONLY
    assert GITHUB_SECRET not in result.sanitised_snippet
    assert SLACK_SECRET not in result.sanitised_snippet
    assert len(result.masked_spans) == 2


def test_build_metadata_only_context_reveals_no_length_signal():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")

    result = build_metadata_only_context(github, [])

    assert "•" * len(GITHUB_SECRET) not in result.sanitised_snippet
    assert "[REDACTED]" in result.sanitised_snippet
