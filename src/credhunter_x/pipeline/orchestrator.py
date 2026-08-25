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


@dataclass(frozen=True)
class ScanOutcome:
    """scan_repository()'s return type -- lets a caller tell "gitleaks
    found nothing" apart from "found something but couldn't classify it,"
    so a CI check never reports an incomplete scan as clean."""

    results: list[ScanResult]
    skipped_count: int


def scan_repository(
    source: Path,
    *,
    settings: Settings,
    scan_config: ScanConfig,
    repo_id: str = "",
    skip_candidate_on_error: bool = True,
) -> ScanOutcome:
    """Single wiring point: gitleaks -> masking -> classifier. This is what
    the CLI calls; run_evaluation.py calls classify_candidates() directly
    to reuse one gitleaks pass across treatments. LiteLLMClient/
    GuardedLLMClient are always constructed together here -- no caller can
    get an unguarded client.

    skip_candidate_on_error defaults True here (unlike classify_candidates'
    False) since this is the unattended CI path -- one bad model response
    shouldn't crash the whole scan. Pass False for a hard failure instead."""
    findings = run_gitleaks(source, gitleaks_binary=settings.gitleaks_binary_path)
    candidates = parse_gitleaks_report(findings, source, repo_id=repo_id)
    results = classify_candidates(
        candidates,
        settings=settings,
        scan_config=scan_config,
        source_root=source,
        skip_candidate_on_error=skip_candidate_on_error,
    )
    return ScanOutcome(results=results, skipped_count=len(candidates) - len(results))


def classify_candidates(
    candidates: list[Candidate],
    *,
    settings: Settings,
    scan_config: ScanConfig,
    source_root: Path,
    delay_seconds: float = 0.0,
    skip_candidate_on_error: bool = False,
) -> list[ScanResult]:
    """Masking + classification over an already-discovered candidate list
    -- the shared tail of scan_repository(), factored out so
    run_evaluation.py can classify a pre-filtered slice without re-running
    gitleaks.

    The guard's secret registry holds only the candidate currently being
    classified, not its neighbours -- a wider registry would raise LeakError
    (and, with skip_candidate_on_error, silently drop the candidate from
    the report) whenever one candidate's context happens to contain another
    candidate's registered value, real secret or not. Two candidates in the
    same repo can legitimately share a value that isn't even a secret (a
    hash, a boilerplate test fixture) and dropping either of them from the
    report over that is wrong -- every gitleaks/trufflehog finding should
    reach the LLM and show up in the report on its own merits. This only
    narrows the guard's own blocking check; bystander masking in the
    context builders and in the agentic classifier's tool-result masking
    (see AgenticClassifier) is unrelated and still sees every other
    candidate in the same repo, since that only blanks out text and never
    causes a candidate to be dropped.

    Results are returned in the same order as `candidates` regardless of
    internal per-repo grouping.

    delay_seconds paces LLM calls for bulk runs against rate limits
    (default 0, no effect on a normal single-repo scan).

    skip_candidate_on_error (default False) excludes just the offending
    candidate instead of aborting the whole batch, for two error types:
    LeakError (the call still aborted and got logged -- this only decides
    what happens after, not whether the guard fires) and LLMParsingError
    (a malformed final answer, a pure availability trade-off, no safety
    angle)."""
    if not candidates:
        return []

    llm_client = LiteLLMClient(model=settings.llm_model, api_key=settings.llm_api_key)

    by_repo: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_repo.setdefault(candidate.repo_id, []).append(candidate)

    results_by_id: dict[str, ScanResult] = {}
    call_index = 0
    for repo_candidates in by_repo.values():
        for candidate in repo_candidates:
            if call_index > 0 and delay_seconds > 0:
                time.sleep(delay_seconds)
            call_index += 1

            registry = SecretRegistry()
            registry.register_candidates([candidate])
            guard = LeakGuard(registry)
            client = GuardedLLMClient(llm_client, guard)

            classifier: SinglePromptClassifier | AgenticClassifier
            if scan_config.mode == Mode.SINGLE:
                classifier = SinglePromptClassifier(client)
            elif scan_config.mode == Mode.AGENTIC:
                classifier = AgenticClassifier(client, source_root, repo_candidates)
            else:
                raise NotImplementedError(f"scan mode {scan_config.mode!r} is not implemented yet")

            others = [c for c in repo_candidates if c.id != candidate.id]
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
            results_by_id[candidate.id] = ScanResult(candidate, classification)

    return [results_by_id[c.id] for c in candidates if c.id in results_by_id]


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
