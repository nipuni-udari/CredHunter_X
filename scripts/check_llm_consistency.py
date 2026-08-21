"""Repeatedly re-classifies a fixed sample of candidates under masked
treatment (the shipped default) to measure the model's OWN stability
across identical, repeated input — RQ3's consistency check. Fully
automated, needs no human judgement: label/severity agreement across the
repeats is measured with Fleiss' kappa, and — for candidates that landed
on true_secret in every repeat and whose rule has a filled-in
remediation_reference.yaml entry — remediation element-presence agreement
too. Manual, costs real LLM quota — not part of CI.

Usage:
    uv run python scripts/check_llm_consistency.py --arm single --split all --n 10 --repeats 5
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from credhunter_x.config.settings import Mode, ScanConfig, Settings
from credhunter_x.dataset.creddata import load_creddata_labels
from credhunter_x.dataset.split import split_by_repo
from credhunter_x.evaluation.metrics import fleiss_kappa
from credhunter_x.evaluation.remediation_reference import load_remediation_reference
from credhunter_x.evaluation.remediation_scoring import ElementCheckParsingError, score_remediation
from credhunter_x.gitleaks.parser import parse_gitleaks_report
from credhunter_x.gitleaks.runner import run_gitleaks
from credhunter_x.guard.leak_guard import LeakGuard
from credhunter_x.llm.guarded_client import GuardedLLMClient
from credhunter_x.llm.litellm_client import LiteLLMClient
from credhunter_x.masking.secret_registry import SecretRegistry
from credhunter_x.models.candidate import Candidate
from credhunter_x.models.treatment import Treatment
from credhunter_x.pipeline.orchestrator import classify_candidates

CREDDATA_ROOT = Path("data/creddata_raw/CredData")


def _repo_id_of(file_path: str) -> str:
    parts = file_path.split("/")
    return parts[1] if len(parts) > 1 else ""


def _element_stability_kappa(
    complete_ids: list[str],
    remediations_by_candidate: dict[str, list[str]],
    candidates_by_id: dict[str, Candidate],
    repeats: int,
) -> float | None:
    """For every candidate that landed on true_secret in every repeat and
    has a non-empty reference entry, scores all `repeats` remediations and
    treats each repeat's pass/fail as one rater. Returns None if there's
    nothing scoreable yet (e.g. remediation_reference.yaml is still the
    placeholder scaffold)."""
    reference = load_remediation_reference()
    registry = SecretRegistry()
    registry.register_candidates(list(candidates_by_id.values()))
    guard = LeakGuard(registry)
    settings = Settings()
    llm_client = LiteLLMClient(model=settings.llm_model, api_key=settings.llm_api_key)
    client = GuardedLLMClient(llm_client, guard)

    pass_ratings: list[list[bool]] = []
    for cid in complete_ids:
        candidate = candidates_by_id[cid]
        entry = reference.get(candidate.rule_id)
        if entry is None or not entry.required_elements:
            continue
        remediations = remediations_by_candidate[cid]
        if len(remediations) != repeats:
            continue

        passes = []
        try:
            for remediation in remediations:
                result = score_remediation(
                    client,
                    candidate_id=cid,
                    rule_id=candidate.rule_id,
                    remediation=remediation,
                    required_elements=entry.required_elements,
                )
                passes.append(result.pass_rate == 1.0)
        except ElementCheckParsingError:
            continue
        pass_ratings.append(passes)

    if len(pass_ratings) < 2:
        return None
    return fleiss_kappa(pass_ratings)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Measure classifier stability across repeated runs"
    )
    parser.add_argument("--arm", choices=["single", "agentic"], default="single")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    mode = Mode.SINGLE if args.arm == "single" else Mode.AGENTIC

    print("Loading ground truth labels...", flush=True)
    all_rows = load_creddata_labels()
    dev_rows, test_rows = split_by_repo(all_rows)
    if args.split == "dev":
        repo_ids = {r.repo_id for r in dev_rows}
    elif args.split == "test":
        repo_ids = {r.repo_id for r in test_rows}
    else:
        repo_ids = {r.repo_id for r in dev_rows} | {r.repo_id for r in test_rows}

    print("Running gitleaks against the full materialized corpus...", flush=True)
    findings = run_gitleaks(CREDDATA_ROOT)
    all_candidates = parse_gitleaks_report(findings, CREDDATA_ROOT)
    candidates = [
        c
        for c in all_candidates
        if _repo_id_of(c.file_path) in repo_ids and c.file_path.endswith(".py")
    ]
    candidates_by_id = {c.id: c for c in candidates}

    rng = random.Random(args.seed)
    sample = rng.sample(candidates, min(args.n, len(candidates)))
    print(
        f"Sampling {len(sample)} candidates, {args.repeats} repeats each "
        f"({args.arm}, masked treatment) -- this costs real API quota",
        flush=True,
    )

    settings = Settings()
    scan_config = ScanConfig(mode=mode, treatment=Treatment.MASKED)

    labels_by_candidate: dict[str, list[str]] = defaultdict(list)
    severities_by_candidate: dict[str, list[str]] = defaultdict(list)
    remediations_by_candidate: dict[str, list[str]] = defaultdict(list)

    for repeat in range(1, args.repeats + 1):
        print(f"  repeat {repeat}/{args.repeats}...", flush=True)
        scan_results = classify_candidates(
            sample,
            settings=settings,
            scan_config=scan_config,
            source_root=CREDDATA_ROOT,
            delay_seconds=8.0,
            skip_candidate_on_error=True,
        )
        for result in scan_results:
            cid = result.candidate.id
            labels_by_candidate[cid].append(result.classification.label.value)
            severities_by_candidate[cid].append(result.classification.severity.value)
            if result.classification.label.value == "true_secret":
                remediations_by_candidate[cid].append(result.classification.remediation)

    complete_ids = [
        cid for cid, labels in labels_by_candidate.items() if len(labels) == args.repeats
    ]
    incomplete = len(sample) - len(complete_ids)
    if incomplete:
        print(
            f"  {incomplete} candidate(s) excluded from the kappa calculation "
            "(didn't get all repeats -- a guard-trip or parse error hit at least one run)",
            flush=True,
        )
    if len(complete_ids) < 2:
        print("not enough complete candidates to compute Fleiss' kappa")
        return

    label_ratings = [labels_by_candidate[cid] for cid in complete_ids]
    severity_ratings = [severities_by_candidate[cid] for cid in complete_ids]
    element_kappa = _element_stability_kappa(
        complete_ids, remediations_by_candidate, candidates_by_id, args.repeats
    )

    print()
    print("=" * 60)
    print(f"n candidates (complete)  : {len(complete_ids)}")
    print(f"repeats per candidate    : {args.repeats}")
    print(f"label fleiss' kappa      : {fleiss_kappa(label_ratings):.3f}")
    print(f"severity fleiss' kappa   : {fleiss_kappa(severity_ratings):.3f}")
    if element_kappa is None:
        print("element-presence kappa   : n/a (no rule had a filled-in reference entry)")
    else:
        print(f"element-presence kappa   : {element_kappa:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
