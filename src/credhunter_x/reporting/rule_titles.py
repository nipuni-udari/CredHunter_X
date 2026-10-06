from __future__ import annotations

# Short readable names for rule ids, used by the SARIF and markdown reports.
# Display only: the classifier's rule lookup (get_gitleaks_rule.py) is separate.
_RULE_TITLES: dict[str, str] = {
    "aws-access-token": "AWS access key ID",
    "generic-api-key": "Generic API key or secret",
    "private-key": "Private key material",
    "jwt": "JSON Web Token",
    "github-pat": "GitHub personal access token",
    "slack-bot-token": "Slack bot token",
    "gcp-api-key": "Google Cloud API key",
    "generic.secret": "Generic secret",
    "generic.password-in-url": "Password in a connection URL",
    "private.key": "Private key material",
    "high-entropy": "High-entropy string",
}


def rule_title(rule_id: str) -> str:
    """Falls back to the raw rule id, which is terse but never wrong."""
    return _RULE_TITLES.get(rule_id, rule_id)
