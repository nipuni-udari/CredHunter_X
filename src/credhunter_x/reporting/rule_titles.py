from __future__ import annotations

# Short, human names for detector rule ids, shared by the SARIF and markdown
# reporters. Display only -- the classifier's own rule lookup
# (classifiers/tools/get_gitleaks_rule.py) is deliberately separate, so
# changing a title here never changes what the model is told.
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
