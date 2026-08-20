from __future__ import annotations

# A small static lookup, not a live query against gitleaks itself -- gitleaks
# has no CLI for printing one rule's description, and the project only needs
# the rule types this dataset's candidates actually carry (see rule_id
# breakdown in the evaluation reports). Covers a few other common rule_ids
# too, for generality beyond this specific dataset.
_RULE_DESCRIPTIONS: dict[str, str] = {
    "aws-access-token": (
        "Matches AWS access key ID format (AKIA/ASIA prefix + 16 alphanumeric "
        "characters). Pattern-only -- does not verify the key is still active."
    ),
    "private-key": (
        "Matches PEM-style private key headers "
        "(-----BEGIN ... PRIVATE KEY-----). Flags the whole key block that "
        "follows the header, not just the header line itself."
    ),
    "generic-api-key": (
        "A broad, low-specificity heuristic: a variable name containing "
        "words like 'key'/'token'/'secret' assigned a sufficiently long, "
        "high-entropy-looking string. Prone to false positives on "
        "non-secret random-looking strings such as UUIDs, hashes, and test "
        "fixtures -- a match on this rule alone is weaker evidence than a "
        "match on a format-specific rule."
    ),
    "jwt": (
        "Matches the three-part, dot-separated base64 structure of a JSON "
        "Web Token (header.payload.signature). Does not decode or verify "
        "the token -- doesn't confirm it's unexpired or tied to a real "
        "session."
    ),
    "github-pat": (
        "Matches GitHub personal access token format (ghp_ prefix + 36 alphanumeric characters)."
    ),
    "slack-bot-token": "Matches Slack bot token format (xoxb- prefix).",
    "gcp-api-key": "Matches Google Cloud API key format (AIza prefix + 35 characters).",
}

_DEFAULT = (
    "No specific description available for this rule -- treat it as a "
    "generic pattern/entropy match with no further guarantee about what it "
    "actually found."
)


def get_gitleaks_rule(rule_id: str) -> str:
    """Looks up what a GitLeaks rule actually checks for, so the model can
    judge how much a match on this specific rule is worth trusting on its
    own (a format-specific match like aws-access-token is stronger evidence
    than a broad heuristic like generic-api-key)."""
    if not rule_id:
        return "get_gitleaks_rule error: rule_id must not be empty"
    return _RULE_DESCRIPTIONS.get(rule_id, f"{rule_id}: {_DEFAULT}")
