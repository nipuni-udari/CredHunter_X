from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RemediationReference:
    rule_id: str
    required_elements: list[str]
    source: str


def load_remediation_reference(
    path: Path = Path("remediation_reference.yaml"),
) -> dict[str, RemediationReference]:
    """Loads the pre-registered reference standard RQ3 scores remediations
    against. Raises (doesn't default to empty) if the file is missing —
    a missing reference means RQ3 can't run at all, not "zero
    requirements for every rule"."""
    if not path.exists():
        raise FileNotFoundError(
            f"remediation reference not found at {path} — RQ3 needs a "
            "pre-registered standard before it can score anything"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        rule_id: RemediationReference(
            rule_id=rule_id,
            required_elements=list(entry.get("required_elements", [])),
            source=entry.get("source", ""),
        )
        for rule_id, entry in data.items()
    }
