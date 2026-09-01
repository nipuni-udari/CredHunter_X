from __future__ import annotations

from credhunter_x.models.candidate import Candidate
from credhunter_x.models.treatment import MaskedSpan, SanitisedContext, Treatment

# ---------------------------------------------------------------------------
# FIX 1: "test fixture" removed from the false-positive examples, and the
# CredData labelling convention stated explicitly. CredData deliberately
# labels credentials in test/ directories as True ("to prevent the case of
# missing real usable credentials"), so instructing the model to treat test
# fixtures as false positives systematically penalises recall against the
# ground truth.
#
# FIX 2: the "judge from surrounding code" guidance now lives HERE, in the
# shared task description, so that every treatment (including RAW) receives
# identical judgement instructions. Previously only the redacted treatments
# were coached to use context, which confounded H4 -- a masked-vs-raw result
# could have been caused by the differing instructions rather than by where
# the discriminative signal actually lives.
#
# FIX 5: FIX 1 closed the *location* loophole but left the *authenticity*
# one open, and the model went straight through it. Measured on the dev
# split's misses, every single one reasoned that the value was "a synthetic
# fixture token, not an active credential" / "not evidenced as a live
# credential" -- never that its location made it safe. Asking for "an actual
# credential that would be a real security risk if exposed" invited exactly
# that, so the convention below now names the authenticity excuses too, and
# says plainly that absence of proof-of-liveness is not evidence of safety.
# Two further changes here: the scanner is no longer described as GitLeaks
# alone (trufflehog3's entropy rule produces most candidates, and telling
# the model its evidence came from pattern matching is simply false), and
# the false-positive examples -- previously the only concrete guidance in
# the prompt, all pointing one way -- are now stated as a closed list of
# things that genuinely are not credential material.
# ---------------------------------------------------------------------------
_TASK_DESCRIPTION = """You are reviewing a candidate secret flagged by \
automated secret scanners — pattern-matching rules and statistical entropy \
detection — in a Python source file. The scanners have no notion of \
true/false positive; they only report pattern and entropy matches. Your job \
is to decide whether this candidate is credential material, or a false \
positive: a value that merely resembles it.

Base your judgement on the evidence available in the code window: the \
variable name, the file's location and role, how the value is used, and the \
scanner metadata provided below.

Important labelling convention. A credential-shaped value counts as a true \
secret even when:
- it appears in test code, a fixture, a tests/ directory, or example code;
- it looks synthetic, non-functional, expired, revoked, or scoped to a local \
emulator, mock service or development account;
- you cannot confirm it was ever provisioned, or is currently usable.

Do not require evidence that the value is live, active or exploitable. That \
evidence is rarely visible in a single code window, and treating its absence \
as proof of harmlessness is exactly how real leaked credentials get \
dismissed.

Classify as a false positive only when the value is not credential material \
at all — for example a hash, digest or commit SHA, a UUID or other \
identifier, a public key, an encoded data payload, an obvious placeholder \
such as "xxx" or "<your-key-here>", or a reference to a variable rather \
than a literal value."""

# ---------------------------------------------------------------------------
# FIX 3: "uncertain" removed from the label set. Ground truth is binary
# (T/F), and McNemar's test requires paired binary outcomes; a third label
# leaves no principled scoring rule and makes arms non-comparable if they
# emit differing numbers of abstentions. Uncertainty is carried by the
# existing `confidence` float instead, which loses nothing.
#
# FIX 4: remediation now explicitly requests the multiple elements the RQ3
# reference standard grades (stop storing in source / rotation / repository
# changes), rather than "a short, actionable next step" (singular), which
# would have underproduced against an 80% multi-element pass threshold.
# ---------------------------------------------------------------------------
_RESPONSE_CONTRACT = """Respond with a JSON object matching this contract:
- label: "true_secret" or "false_positive"
- confidence: a float between 0 and 1 (use this to express uncertainty; do \
not abstain)
- severity: "low", "medium", "high", or "critical" (impact if label is \
true_secret; "low" otherwise)
- explanation: a short, concrete justification citing what you observed
- remediation: the concrete steps needed to resolve this, covering how to \
stop storing the value in source, whether the credential needs rotating or \
revoking, and any repository changes required"""

_AGENTIC_ADDENDUM = """You have access to tools that let you verify claims \
about this code before deciding:
- search_file: search a specific file in the repository for a text pattern, \
returning matching lines with a little surrounding context
- check_file_exists: check whether a file path actually exists in the \
repository
- get_gitleaks_rule: look up what the scanner rule that flagged this \
candidate actually checks for

Use these tools when they would change your judgement — for example, \
checking whether a value is referenced from application code as well as \
tests, or whether a file path implied by the code actually exists. You do \
not need to use every tool, or any tool, if the code context already \
makes the answer clear.

Once you have enough information and are ready to give your final answer, \
call the submit_classification tool with:
- label: "true_secret" or "false_positive"
- confidence: a float between 0 and 1 (use this to express uncertainty; do \
not abstain)
- severity: "low", "medium", "high", or "critical" (impact if label is \
true_secret; "low" otherwise)
- explanation: a short, concrete justification citing what you observed \
(including anything a tool told you)
- remediation: the concrete steps needed to resolve this, covering how to \
stop storing the value in source, whether the credential needs rotating or \
revoking, and any repository changes required"""

# ---------------------------------------------------------------------------
# Treatment notes now state ONLY the factual difference between treatments:
# what was replaced, with what, and what that placeholder does or does not
# tell you. All judgement guidance has moved to _TASK_DESCRIPTION so it is
# identical across arms. (FIX 2, continued.)
# ---------------------------------------------------------------------------
_MASKED_NOTE = """Note: this code window has been sanitised. Every candidate \
secret value (this one and any others in the surrounding lines) has been \
replaced with a placeholder of bullet characters ("•") — you will never see \
a real value. The placeholder's literal content carries no information; the \
only signal about a masked value is the metadata listed below."""

_RAW_WITH_MASKED_NEIGHBOURS_NOTE = """Note: the candidate under review is \
shown with its real, unmasked value below. Any OTHER secret values sharing \
this code window have been replaced with placeholder bullet characters \
("•") — those belong to unrelated candidates, not the one you are \
evaluating, and their real values were never sent to you."""

_PSEUDONYMISED_NOTE = """Note: this code window has been sanitised. Every \
candidate secret value (this one and any others in the surrounding lines) \
has been replaced with a FAKE value of the same shape and format as a real \
credential of that type — it looks like a real secret, but it is not the \
real one and carries no information about the actual value. The fake \
value's literal characters are not evidence of anything; the only signal \
about a replaced value is the metadata listed below."""

_METADATA_ONLY_NOTE = """Note: this code window has been sanitised. Every \
candidate secret value (this one and any others in the surrounding lines) \
has been replaced with a short fixed marker, "[REDACTED]". This marker \
carries NO information about the real value's length, shape, or content — \
its length is unrelated to the real value's length. The only signal about a \
redacted value is the metadata listed below."""

_TREATMENT_NOTES: dict[Treatment, str] = {
    Treatment.MASKED: _MASKED_NOTE,
    Treatment.RAW: _RAW_WITH_MASKED_NEIGHBOURS_NOTE,
    Treatment.PSEUDONYMISED: _PSEUDONYMISED_NOTE,
    Treatment.METADATA_ONLY: _METADATA_ONLY_NOTE,
}


def _format_span_metadata(span: MaskedSpan) -> str:
    """Render one sanitised span's metadata.

    Spans are labelled as either the candidate under review or a neighbour,
    so the model is never left guessing which metadata block belongs to the
    value it is judging. The role comes from span.is_target, set when the
    span was built: deriving it here from line ranges instead labelled every
    secret sharing the target's line as the candidate under review, which
    handed the model several contradictory metadata blocks (a fifth of the
    corpus sits on a shared line). `prefix_hint` is rendered as "prefix"
    rather than "rule" to avoid collision with the actual scanner rule line
    below.
    """
    metadata = span.metadata
    role = "candidate under review" if span.is_target else "neighbouring candidate"
    return (
        f"Sanitised span ({role}, lines {span.start}-{span.end}): "
        f"length={metadata.length}, entropy={metadata.entropy:.2f}, "
        f"charset={metadata.charset!r}, prefix={metadata.prefix_hint!r}"
    )


def _context_sections(candidate: Candidate, context: SanitisedContext) -> list[str]:
    sections = []

    # The treatment note is emitted unconditionally, not only when
    # masked_spans is non-empty. Previously a RAW candidate with no
    # neighbours produced a structurally different prompt (no note, no
    # metadata block) from a RAW candidate that happened to sit near
    # another secret -- prompt shape varied within a single treatment.
    sections.append(_TREATMENT_NOTES[context.treatment])

    if context.masked_spans:
        for span in context.masked_spans:
            sections.append(_format_span_metadata(span))

    sections.append(
        f"Scanner rule: {candidate.rule_id}\n"
        f"Reported entropy: {candidate.entropy:.2f}\n"
        f"File: {candidate.file_path}, lines {candidate.line_start}-{candidate.line_end}"
    )
    sections.append(f"Code context:\n```python\n{context.sanitised_snippet}\n```")
    return sections


def build_classification_prompt(candidate: Candidate, context: SanitisedContext) -> str:
    """Arm A -- single call, immediate structured answer."""
    sections = [_TASK_DESCRIPTION, _RESPONSE_CONTRACT, *_context_sections(candidate, context)]
    return "\n\n".join(sections)


def build_agentic_prompt(candidate: Candidate, context: SanitisedContext) -> str:
    """Arm B's initial prompt -- same framing as Arm A, but with the
    tool-usage addendum instead of an immediate response contract."""
    sections = [_TASK_DESCRIPTION, _AGENTIC_ADDENDUM, *_context_sections(candidate, context)]
    return "\n\n".join(sections)
