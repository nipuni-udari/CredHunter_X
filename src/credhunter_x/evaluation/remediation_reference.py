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
    """v2 elements are dicts with id, description and accepted_phrasings; v1
    elements were plain strings. Only the description goes to the checker;
    accepted_phrasings are hints for hand-labelling, not keywords."""
    if isinstance(element, str):
        return element.strip()
    if isinstance(element, dict):
        return str(element.get("description", "")).strip()
    raise TypeError(f"unsupported required_elements entry: {type(element).__name__}")


def _element_gate(element: object) -> str | None:
    """waived_unless adds one yes/no question. If the answer is no, the element
    is waived rather than failed: "revoke the credential" isn't needed if the
    advice concluded it was never a credential."""
    if isinstance(element, dict) and element.get("waived_unless"):
        return str(element["waived_unless"]).strip()
    return None


def load_remediation_reference(
    path: Path = Path("remediation_reference.yaml"),
) -> dict[str, RemediationReference]:
    """Loads the reference standard for RQ3. Raises if the file is missing,
    because RQ3 can't run without it."""
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
