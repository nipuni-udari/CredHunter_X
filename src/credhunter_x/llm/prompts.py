from __future__ import annotations

from credhunter_x.models.candidate import Candidate
from credhunter_x.models.treatment import SanitisedContext, Treatment

_TASK_DESCRIPTION = """You are reviewing a candidate secret flagged by a static \
pattern-matching scanner (GitLeaks) in a Python source file. The scanner has \
no notion of true/false positive — it only reports pattern/entropy matches. \
Your job is to decide whether this candidate is an actual, live credential \
that would be a real security risk if exposed, or a false positive (for \
example a placeholder, example value, test fixture, or non-secret that \
happened to match the pattern)."""

_RESPONSE_CONTRACT = """Respond with a JSON object matching this contract:
- label: "true_secret", "false_positive", or "uncertain"
- confidence: a float between 0 and 1
- severity: "low", "medium", "high", or "critical" (impact if label is \
true_secret; "low" otherwise)
- explanation: a short, concrete justification citing what you observed
- remediation: a short, actionable next step"""

_AGENTIC_ADDENDUM = """You have access to tools that let you verify claims \
about this code before deciding:
- search_file: search a specific file in the repository for a text pattern, \
returning matching lines with a little surrounding context
- check_file_exists: check whether a file path actually exists in the \
repository
- get_gitleaks_rule: look up what the scanner rule that flagged this \
candidate actually checks for

Use these tools when they would change your judgement — for example, \
checking whether a suspicious value is only ever referenced from test \
code, or whether a file path implied by the code actually exists. You do \
not need to use every tool, or any tool, if the code context already \
makes the answer clear.

Once you have enough information and are ready to give your final answer, \
call the submit_classification tool with:
- label: "true_secret", "false_positive", or "uncertain"
- confidence: a float between 0 and 1
- severity: "low", "medium", "high", or "critical" (impact if label is \
true_secret; "low" otherwise)
- explanation: a short, concrete justification citing what you observed \
(including anything a tool told you)
- remediation: a short, actionable next step"""

_MASKED_NOTE = """Note: this code window has been sanitised. Every candidate \
secret value (this one and any others in the surrounding lines) has been \
replaced with a placeholder of bullet characters ("•") — you will never see \
the real value. The only signal available for a masked value is its \
metadata below: length, Shannon entropy, character set, and the scanner \
rule that flagged it. Base your judgement on the surrounding code (variable \
naming, file location, how the value is used) and that metadata, never on \
the placeholder's literal content."""

_RAW_WITH_MASKED_NEIGHBOURS_NOTE = """Note: the candidate under review is \
shown with its real, unmasked value below. One or more OTHER secret values \
that happen to share this code window have been replaced with placeholder \
bullet characters ("•") — those belong to unrelated candidates, not the one \
you are evaluating, and their real values were never sent to you. Base your \
judgement on the candidate under review, whose real value you can see \
directly."""

_PSEUDONYMISED_NOTE = """Note: this code window has been sanitised. Every \
candidate secret value (this one and any others in the surrounding lines) \
has been replaced with a FAKE value of the same shape and format as a real \
credential of that type — it looks like a real secret, but it is not the \
real one, and carries no information about the actual value. Do not treat \
the placeholder's specific characters as evidence of anything; base your \
judgement on the surrounding code (variable naming, file location, how the \
value is used) and the metadata below, never on the fake value's literal \
content."""

_METADATA_ONLY_NOTE = """Note: this code window has been sanitised. Every \
candidate secret value (this one and any others in the surrounding lines) \
has been replaced with a short fixed marker, "[REDACTED]" — this marker \
carries NO information about the real value's length, shape, or content; \
its length is not related to the real value's length. The only signal \
available for a redacted value is its metadata below: length, Shannon \
entropy, character set, and the scanner rule that flagged it. Base your \
judgement on the surrounding code (variable naming, file location, how the \
value is used) and that metadata."""

_TREATMENT_NOTES: dict[Treatment, str] = {
    Treatment.MASKED: _MASKED_NOTE,
    Treatment.RAW: _RAW_WITH_MASKED_NEIGHBOURS_NOTE,
    Treatment.PSEUDONYMISED: _PSEUDONYMISED_NOTE,
    Treatment.METADATA_ONLY: _METADATA_ONLY_NOTE,
}


def _context_sections(candidate: Candidate, context: SanitisedContext) -> list[str]:
    sections = []

    if context.masked_spans:
        sections.append(_TREATMENT_NOTES[context.treatment])
        for span in context.masked_spans:
            metadata = span.metadata
            sections.append(
                f"Sanitised span (lines {span.start}-{span.end}): "
                f"length={metadata.length}, entropy={metadata.entropy:.2f}, "
                f"charset={metadata.charset!r}, rule={metadata.prefix_hint!r}"
            )

    sections.append(
        f"Scanner rule: {candidate.rule_id}\n"
        f"Reported entropy: {candidate.entropy:.2f}\n"
        f"File: {candidate.file_path}, lines {candidate.line_start}-{candidate.line_end}"
    )
    sections.append(f"Code context:\n```python\n{context.sanitised_snippet}\n```")
    return sections


def build_classification_prompt(candidate: Candidate, context: SanitisedContext) -> str:
    sections = [_TASK_DESCRIPTION, _RESPONSE_CONTRACT, *_context_sections(candidate, context)]
    return "\n\n".join(sections)


def build_agentic_prompt(candidate: Candidate, context: SanitisedContext) -> str:
    """Arm B's initial prompt -- same framing as Arm A, but with the
    tool-usage addendum instead of an immediate response contract."""
    sections = [_TASK_DESCRIPTION, _AGENTIC_ADDENDUM, *_context_sections(candidate, context)]
    return "\n\n".join(sections)
