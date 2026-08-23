from __future__ import annotations

_TASK_DESCRIPTION = """You are checking whether a piece of remediation \
advice for a leaked credential covers a specific set of required \
elements. For each numbered element below, decide whether the \
remediation text satisfies it — even if worded differently, as long as \
the same substantive action is covered."""


def build_element_check_prompt(remediation: str, required_elements: list[str]) -> str:
    """Numbers required_elements 1..N; asks for a JSON array of N booleans
    in that order, matching ElementCheckSchema."""
    numbered = "\n".join(f"{i}. {element}" for i, element in enumerate(required_elements, start=1))
    contract = (
        f'Respond with a JSON object: {{"element_present": [...]}} — an array of exactly '
        f"{len(required_elements)} booleans, in the same order as the numbered elements above, "
        "one per element."
    )
    return "\n\n".join(
        [
            _TASK_DESCRIPTION,
            f"Required elements:\n{numbered}",
            f'Remediation text to check:\n"""\n{remediation}\n"""',
            contract,
        ]
    )
