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
    """Loads the pre-registered reference standard RQ3 scores against.
    Raises if missing -- no reference means RQ3 can't run, not "zero
    requirements"."""
    if not path.exists():
        raise FileNotFoundError(f"remediation reference not found at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        rule_id: RemediationReference(
            rule_id=rule_id,
            required_elements=list(entry.get("required_elements", [])),
            source=entry.get("source", ""),
        )
        for rule_id, entry in data.items()
    }
