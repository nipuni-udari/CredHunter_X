"""Research evaluation script: scores GitLeaks alone, or GitLeaks + an LLM
classifier (Arm A single-prompt, or Arm B agentic), against CredData's
ground truth. Costs real LLM quota when --arm single/agentic is used — not
part of CI, run manually.

Usage:
    uv run python scripts/run_evaluation.py --split dev --arm gitleaks_only
    uv run python scripts/run_evaluation.py --split all --arm single --treatment raw
    uv run python scripts/run_evaluation.py --split all --arm single --treatment masked
    uv run python scripts/run_evaluation.py --split all --arm single --treatment pseudonymised
    uv run python scripts/run_evaluation.py --split all --arm single --treatment metadata_only
    uv run python scripts/run_evaluation.py --split all --arm agentic --treatment raw
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from credhunter_x.config.settings import Mode, ScanConfig, Settings
from credhunter_x.dataset.creddata import GroundTruthRow, load_creddata_labels
from credhunter_x.dataset.split import split_by_repo
from credhunter_x.evaluation.labeler import (
    MatchOutcome,
    index_ground_truth,
    match_candidate,
)
from credhunter_x.evaluation.metrics import compute_metrics, mcnemar_test
from credhunter_x.gitleaks.parser import parse_gitleaks_report
from credhunter_x.gitleaks.runner import run_gitleaks
from credhunter_x.models.candidate import Candidate
from credhunter_x.models.classification import Label
from credhunter_x.models.evaluation import MetricReport
from credhunter_x.models.treatment import Treatment
from credhunter_x.pipeline.orchestrator import ScanResult, classify_candidates

CREDDATA_ROOT = Path("data/creddata_raw/CredData")
RESULTS_DIR = Path("results")


def _repo_id_of(file_path: str) -> str:
    # CredData's own path convention: "data/{RepoID}/{src|test|other}/{FileID}.py"
    parts = file_path.split("/")
    return parts[1] if len(parts) > 1 else ""


def _select_candidates(candidates: list[Candidate], repo_ids: set[str]) -> list[Candidate]:
    return [
        c
        for c in candidates
        if _repo_id_of(c.file_path) in repo_ids and c.file_path.endswith(".py")
    ]


def _print_report(report: MetricReport) -> None:
    print()
    print("=" * 60)
    print(f"arm          : {report.arm}")
    print(f"treatment    : {report.treatment}")
    print(f"n_candidates : {report.n_candidates}")
    print(f"lost_count   : {report.lost_count}")
    print(f"precision    : {report.precision:.3f}")
    print(f"recall       : {report.recall:.3f}")
    print(f"f1           : {report.f1:.3f}")
    if report.mcnemar_stat is not None:
        print(f"mcnemar stat : {report.mcnemar_stat:.3f}")
        print(f"mcnemar p    : {report.mcnemar_p_value:.4f}")
    print("=" * 60)


def _write_results(
    *,
    arm: str,
    treatment: str,
    split: str,
    model: str,
    scan_results: list[ScanResult],
    excluded: list[Candidate],
    gt_rows: list[GroundTruthRow],
    report: MetricReport,
) -> None:
    """Persists a run's full per-candidate output plus its summary metrics
    to results/, gitignored (raw treatment's explanations can quote real
    secret values verbatim, same as they do on stdout -- inherent to raw
    treatment, not new exposure here)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"{arm}_{treatment}_{split}"
    gt_index = index_ground_truth(gt_rows)

    jsonl_path = RESULTS_DIR / f"{stem}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for result in scan_results:
            c, cl = result.candidate, result.classification
            row = {
                "candidate_id": c.id,
                "file_path": c.file_path,
                "line_start": c.line_start,
                "line_end": c.line_end,
                "rule_id": c.rule_id,
                "real_secret_length": len(c.matched_value),
                "ground_truth_outcome": match_candidate(c, gt_index).value,
                "label": cl.label.value,
                "confidence": cl.confidence,
                "severity": cl.severity.value,
                "explanation": cl.explanation,
                "remediation": cl.remediation,
                "turns": cl.turns,
                "input_tokens": cl.input_tokens,
                "output_tokens": cl.output_tokens,
                "latency_ms": cl.latency_ms,
                "tool_calls": [
                    {
                        "tool_name": tc.tool_name,
                        "arguments": tc.arguments,
                        "result_summary": tc.result_summary,
                    }
                    for tc in cl.tool_calls
                ],
            }
            f.write(json.dumps(row) + "\n")

    summary_path = RESULTS_DIR / f"{stem}_summary.json"
    summary = {
        "arm": arm,
        "treatment": treatment,
        "split": split,
        "model": model,
        "generated_at": datetime.now(UTC).isoformat(),
        "n_candidates_evaluated": len(scan_results),
        "excluded_candidate_ids": [c.id for c in excluded],
        "metrics": {
            "n_candidates": report.n_candidates,
            "lost_count": report.lost_count,
            "precision": report.precision,
            "recall": report.recall,
            "f1": report.f1,
            "mcnemar_stat": report.mcnemar_stat,
            "mcnemar_p_value": report.mcnemar_p_value,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  results written to {jsonl_path} and {summary_path}", flush=True)


def _write_gitleaks_results(
    *,
    split: str,
    candidates: list[Candidate],
    gt_rows: list[GroundTruthRow],
    report: MetricReport,
) -> None:
    """Persists the GitLeaks-only baseline the same way _write_results does
    for the LLM arms, so every arm has a comparable results/ file. No
    classification fields here — GitLeaks itself has no label/confidence/
    explanation, just a flagged file+line."""
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"gitleaks_only_raw_{split}"
    gt_index = index_ground_truth(gt_rows)

    jsonl_path = RESULTS_DIR / f"{stem}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for c in candidates:
            row = {
                "candidate_id": c.id,
                "file_path": c.file_path,
                "line_start": c.line_start,
                "line_end": c.line_end,
                "rule_id": c.rule_id,
                "ground_truth_outcome": match_candidate(c, gt_index).value,
            }
            f.write(json.dumps(row) + "\n")

    summary_path = RESULTS_DIR / f"{stem}_summary.json"
    summary = {
        "arm": "gitleaks_only",
        "treatment": "raw",
        "split": split,
        "generated_at": datetime.now(UTC).isoformat(),
        "n_candidates_evaluated": len(candidates),
        "metrics": {
            "n_candidates": report.n_candidates,
            "lost_count": report.lost_count,
            "precision": report.precision,
            "recall": report.recall,
            "f1": report.f1,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  results written to {jsonl_path} and {summary_path}", flush=True)


def _run_gitleaks_only(
    candidates: list[Candidate], gt_rows: list[GroundTruthRow]
) -> tuple[MetricReport, list[bool]]:
    gt_index = index_ground_truth(gt_rows)
    total_true_count = sum(1 for r in gt_rows if r.ground_truth)
    outcomes = [match_candidate(c, gt_index) for c in candidates]
    correct = [o == MatchOutcome.TRUE_POSITIVE for o in outcomes]
    report = compute_metrics(
        outcomes, total_true_count=total_true_count, arm="gitleaks_only", treatment="raw"
    )
    return report, correct


def _run_llm_arm(
    candidates: list[Candidate],
    gt_rows: list[GroundTruthRow],
    *,
    mode: Mode,
    treatment: Treatment,
    split: str,
) -> tuple[MetricReport, MetricReport]:
    """Returns (gitleaks_only_report, llm_report), both computed over the
    same surviving candidate set so McNemar pairing stays valid even when
    the guard blocks a candidate at runtime.

    Excluded candidates can't be known in advance: whether a call trips the
    guard on a real-but-harmless byte overlap (e.g. a certificate/public
    key sharing key material with its own private key) depends on whether
    Arm B's model happens to call a tool that pulls in that overlapping
    content from outside the candidate's fixed context window -- so the
    survivor set is only known after classify_candidates() returns, not
    filterable up front. See classify_candidates' skip_candidate_on_error
    docstring for what does and doesn't change about the guard itself."""
    gt_index = index_ground_truth(gt_rows)
    total_true_count = sum(1 for r in gt_rows if r.ground_truth)

    settings = Settings()
    scan_config = ScanConfig(mode=mode, treatment=treatment)
    arm_name = "single" if mode == Mode.SINGLE else "agentic"

    print(
        f"Calling the LLM for {len(candidates)} candidates (this costs real API quota)...",
        flush=True,
    )
    # Groq's free-tier TPM budget (8000 tokens/min on this model) can't absorb
    # 87 back-to-back calls, some carrying large multi-line private-key
    # context windows -- pace calls so we stay under budget instead of
    # relying solely on the client's own retry/backoff to recover.
    scan_results = classify_candidates(
        candidates,
        settings=settings,
        scan_config=scan_config,
        source_root=CREDDATA_ROOT,
        delay_seconds=8.0,
        skip_candidate_on_error=True,
    )

    survived_ids = {r.candidate.id for r in scan_results}
    excluded = [c for c in candidates if c.id not in survived_ids]
    if excluded:
        print(
            f"  {len(excluded)} candidate(s) excluded: either the guard blocked a "
            "real-but-harmless byte overlap, or the model produced unparseable output "
            "(see _run_llm_arm docstring / the WARNING-level log line for each)",
            flush=True,
        )
        for c in excluded:
            print(f"    - {c.id}", flush=True)

    survived_candidates = [c for c in candidates if c.id in survived_ids]
    gitleaks_report, gitleaks_correct = _run_gitleaks_only(survived_candidates, gt_rows)

    # Only a TRUE_SECRET verdict means the pipeline would surface this candidate
    # to a user at all -- FALSE_POSITIVE/UNCERTAIN verdicts mean it suppresses
    # the candidate, so it's excluded from precision's numerator/denominator,
    # same as a scanner that never generated a candidate here in the first place.
    outcomes = []
    correct = []
    for result in scan_results:
        flagged = result.classification.label == Label.TRUE_SECRET
        outcome = match_candidate(result.candidate, gt_index)
        if flagged:
            outcomes.append(outcome)
        correct.append(flagged and outcome == MatchOutcome.TRUE_POSITIVE)

    report = compute_metrics(
        outcomes, total_true_count=total_true_count, arm=arm_name, treatment=treatment.value
    )

    stat, p_value = mcnemar_test(gitleaks_correct, correct)
    report = MetricReport(
        arm=report.arm,
        treatment=report.treatment,
        n_candidates=report.n_candidates,
        precision=report.precision,
        recall=report.recall,
        f1=report.f1,
        lost_count=report.lost_count,
        mcnemar_stat=stat,
        mcnemar_p_value=p_value,
    )

    _write_results(
        arm=arm_name,
        treatment=treatment.value,
        split=split,
        model=settings.llm_model,
        scan_results=scan_results,
        excluded=excluded,
        gt_rows=gt_rows,
        report=report,
    )
    return gitleaks_report, report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run CredHunter-X evaluation against CredData")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument(
        "--arm", choices=["gitleaks_only", "single", "agentic"], default="gitleaks_only"
    )
    parser.add_argument(
        "--treatment",
        choices=["raw", "masked", "pseudonymised", "metadata_only"],
        default="raw",
    )
    args = parser.parse_args()
    treatment = Treatment(args.treatment)
    mode = Mode.SINGLE if args.arm == "single" else Mode.AGENTIC

    print("Loading ground truth labels...", flush=True)
    all_rows = load_creddata_labels()
    dev_rows, test_rows = split_by_repo(all_rows)
    dev_repo_ids = {r.repo_id for r in dev_rows}
    test_repo_ids = {r.repo_id for r in test_rows}

    if args.split == "dev":
        gt_rows, repo_ids = dev_rows, dev_repo_ids
    elif args.split == "test":
        gt_rows, repo_ids = test_rows, test_repo_ids
    else:
        gt_rows, repo_ids = all_rows, dev_repo_ids | test_repo_ids
    print(
        f"  split={args.split}: {len(gt_rows)} ground-truth rows across {len(repo_ids)} repos",
        flush=True,
    )

    print("Running gitleaks against the full materialized corpus...", flush=True)
    findings = run_gitleaks(CREDDATA_ROOT)
    all_candidates = parse_gitleaks_report(findings, CREDDATA_ROOT)
    candidates = _select_candidates(all_candidates, repo_ids)
    print(f"  {len(candidates)} .py candidates fall within the '{args.split}' split", flush=True)

    gitleaks_report, _ = _run_gitleaks_only(candidates, gt_rows)
    _write_gitleaks_results(
        split=args.split, candidates=candidates, gt_rows=gt_rows, report=gitleaks_report
    )

    if args.arm == "gitleaks_only":
        _print_report(gitleaks_report)
        return

    gitleaks_report, llm_report = _run_llm_arm(
        candidates, gt_rows, mode=mode, treatment=treatment, split=args.split
    )

    print()
    print("--- gitleaks_only (for comparison / McNemar pairing) ---")
    _print_report(gitleaks_report)
    print()
    print(f"--- {args.arm} ---")
    _print_report(llm_report)


if __name__ == "__main__":
    main()
