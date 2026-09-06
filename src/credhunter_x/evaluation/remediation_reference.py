from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# v2 nests rule families under these; v1 put rule ids at the top level.
_RULE_SECTIONS = ("rules", "retained_zero_candidate_rules")


@dataclass(frozen=True)
class RemediationReference:
    rule_id: str
    required_elements: list[str]
    source: str
    optional_elements: list[str] = field(default_factory=list)
    # Aligned 1:1 with required_elements. None means always required.
    required_gates: list[str | None] = field(default_factory=list)


def _element_text(element: object) -> str:
    """v2 elements are dicts carrying id/description/accepted_phrasings;
    v1 elements were plain strings. Only `description` reaches the checker
    -- accepted_phrasings are hints for the human hand-labeller, and
    feeding them in would turn semantic matching into keyword matching,
    which the reference file explicitly says they are not."""
    if isinstance(element, str):
        return element.strip()
    if isinstance(element, dict):
        return str(element.get("description", "")).strip()
    raise TypeError(f"unsupported required_elements entry: {type(element).__name__}")


def _element_gate(element: object) -> str | None:
    """`waived_unless` turns a conditional element into one extra yes/no
    question. When the gate answers false the element is waived rather
    than failed -- "revoke the credential" is not a defect in advice that
    concluded the value was never a credential."""
    if isinstance(element, dict) and element.get("waived_unless"):
        return str(element["waived_unless"]).strip()
    return None


def load_remediation_reference(
    path: Path = Path("remediation_reference.yaml"),
) -> dict[str, RemediationReference]:
    """Loads the reference standard RQ3 scores against. Raises if missing
    -- no reference means RQ3 can't run, not "zero requirements"."""
    if not path.exists():
        raise FileNotFoundError(f"remediation reference not found at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    entries: dict[str, dict[str, Any]] = {}
    for section in _RULE_SECTIONS:
        entries.update(data.get(section) or {})
    if not entries:
        # v1 flat layout: rule ids at the top level, no _-prefixed metadata
        entries = {k: v for k, v in data.items() if not k.startswith("_")}

    return {
        rule_id: RemediationReference(
            rule_id=rule_id,
            required_elements=[_element_text(e) for e in entry.get("required_elements", [])],
            required_gates=[_element_gate(e) for e in entry.get("required_elements", [])],
            optional_elements=[_element_text(e) for e in entry.get("optional_elements", [])],
            source=str(entry.get("source") or entry.get("source_note") or ""),
        )
        for rule_id, entry in entries.items()
    }
