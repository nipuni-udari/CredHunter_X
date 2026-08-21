from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from credhunter_x.classifiers.agentic import AgenticClassifier
from credhunter_x.classifiers.single_prompt import SinglePromptClassifier
from credhunter_x.config.settings import Mode, ScanConfig, Settings
from credhunter_x.gitleaks.parser import parse_gitleaks_report
from credhunter_x.gitleaks.runner import run_gitleaks
from credhunter_x.guard.errors import LeakError
from credhunter_x.guard.leak_guard import LeakGuard
from credhunter_x.llm.guarded_client import GuardedLLMClient
from credhunter_x.llm.litellm_client import LiteLLMClient
from credhunter_x.llm.parsing import LLMParsingError
from credhunter_x.masking.masker import (
    build_metadata_only_context,
    build_pseudonymised_context,
    build_raw_context,
    mask_context_window,
)
from credhunter_x.masking.secret_registry import SecretRegistry
from credhunter_x.models.candidate import Candidate
from credhunter_x.models.classification import ClassificationResult
from credhunter_x.models.treatment import SanitisedContext, Treatment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanResult:
    candidate: Candidate
    classification: ClassificationResult


def scan_repository(
    source: Path,
    *,
    settings: Settings,
    scan_config: ScanConfig,
    repo_id: str = "",
) -> list[ScanResult]:
    """Single wiring point: gitleaks -> masking -> classifier. Both the CLI
    and scripts/run_evaluation.py call this rather than duplicating the
    wiring. LiteLLMClient/GuardedLLMClient are constructed only here,
    always wrapped together — no caller can obtain an unguarded client."""
    findings = run_gitleaks(source, gitleaks_binary=settings.gitleaks_binary_path)
    candidates = parse_gitleaks_report(findings, source, repo_id=repo_id)
    return classify_candidates(
        candidates, settings=settings, scan_config=scan_config, source_root=source
    )


def classify_candidates(
    candidates: list[Candidate],
    *,
    settings: Settings,
    scan_config: ScanConfig,
    source_root: Path,
    delay_seconds: float = 0.0,
    skip_candidate_on_error: bool = False,
) -> list[ScanResult]:
    """Masking + classification over an already-discovered candidate list —
    the shared tail of scan_repository(), factored out so
    scripts/run_evaluation.py can classify a pre-filtered CredData slice
    without re-running gitleaks or duplicating the guard/registry wiring.

    source_root is only actually used by Arm B (its tools read files from
    disk) but is required unconditionally, since which classifier gets
    built is decided by scan_config.mode inside this function, not by the
    caller.

    delay_seconds paces successive LLM calls (default 0, so scan_repository/
    the CLI's normal single-repo scan is unaffected) — bulk evaluation runs
    over many candidates can exceed a provider's tokens-per-minute budget
    faster than the client's own retry/backoff can recover from, so
    scripts/run_evaluation.py passes a nonzero value there.

    skip_candidate_on_error (default False, so scan_repository/the CLI's
    normal behaviour is unchanged) lets scripts/run_evaluation.py opt in to
    excluding just the offending candidate from the returned results
    instead of letting the whole batch abort. Two error types are caught:

    - LeakError: does not weaken the guard itself — the call that would
      have leaked a fragment still aborts, nothing is ever sent, and the
      block is still logged at ERROR by LeakGuard.check() before this
      catches it. It only changes what happens *after* that abort — move
      on to the next candidate instead of crashing the process — which
      matters for Arm B, where a tool call can legitimately pull in a
      paired public key/certificate's overlapping bytes from outside a
      candidate's fixed context window (see scripts/run_evaluation.py).
    - LLMParsingError: no safety concern at all, just a malformed/
      unparseable final answer from the model for that one candidate
      (e.g. a stray chat-template token instead of JSON) — excluding it
      is a pure availability trade-off, not a guard decision."""
    if not candidates:
        return []

    registry = SecretRegistry()
    registry.register_candidates(candidates)
    guard = LeakGuard(registry)
    llm_client = LiteLLMClient(model=settings.llm_model, api_key=settings.llm_api_key)
    client = GuardedLLMClient(llm_client, guard)

    classifier: SinglePromptClassifier | AgenticClassifier
    if scan_config.mode == Mode.SINGLE:
        classifier = SinglePromptClassifier(client)
    elif scan_config.mode == Mode.AGENTIC:
        classifier = AgenticClassifier(client, source_root, candidates)
    else:
        raise NotImplementedError(f"scan mode {scan_config.mode!r} is not implemented yet")

    results = []
    for i, candidate in enumerate(candidates):
        if i > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)
        others = [c for c in candidates if c.id != candidate.id]
        context = _build_context(candidate, others, scan_config.treatment)
        if skip_candidate_on_error:
            try:
                classification = classifier.classify(candidate, context)
            except LeakError:
                logger.warning(
                    "Excluding candidate %s from results after a blocked call "
                    "(skip_candidate_on_error=True)",
                    candidate.id,
                )
                continue
            except LLMParsingError as exc:
                logger.warning(
                    "Excluding candidate %s from results after unparseable model "
                    "output (skip_candidate_on_error=True): %s",
                    candidate.id,
                    exc,
                )
                continue
        else:
            classification = classifier.classify(candidate, context)
        results.append(ScanResult(candidate, classification))
    return results


_CONTEXT_BUILDERS: dict[Treatment, Callable[[Candidate, list[Candidate]], SanitisedContext]] = {
    Treatment.RAW: build_raw_context,
    Treatment.MASKED: mask_context_window,
    Treatment.PSEUDONYMISED: build_pseudonymised_context,
    Treatment.METADATA_ONLY: build_metadata_only_context,
}


def _build_context(
    candidate: Candidate, others: list[Candidate], treatment: Treatment
) -> SanitisedContext:
    builder = _CONTEXT_BUILDERS.get(treatment)
    if builder is None:
        raise NotImplementedError(f"treatment {treatment!r} is not implemented yet")
    return builder(candidate, others)
