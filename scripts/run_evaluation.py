"""Research evaluation script: scores a detector alone, or a detector + an
LLM classifier (Arm A single-prompt, or Arm B agentic), against CredData's
ground truth. Costs real LLM quota when --arm single/agentic is used — not
part of CI, run manually.

--source picks which scanner(s) generate candidates (default gitleaks, for
backwards compatibility); "combined" runs gitleaks and trufflehog3
independently and merges out any secret both flag (see
pipeline/candidate_merge.py) so it isn't sent to the LLM twice. --arm
gitleaks_only skips the LLM entirely regardless of --source -- it just
reports whichever source was configured.

--candidates-cache freezes the scan so every arm and treatment is scored on
one identical candidate set (see _load_candidates_cache). It is off by
default: re-scanning is the correct behaviour for a real repository, which
gains new secrets between runs.

Usage:
    uv run python scripts/run_evaluation.py --split dev --arm gitleaks_only
    uv run python scripts/run_evaluation.py --split dev --arm gitleaks_only --source trufflehog
    uv run python scripts/run_evaluation.py --split dev --arm gitleaks_only --source combined
    uv run python scripts/run_evaluation.py --split all --arm single --treatment raw
    uv run python scripts/run_evaluation.py --split all --arm single --treatment masked
    uv run python scripts/run_evaluation.py --split all --arm single --treatment pseudonymised
    uv run python scripts/run_evaluation.py --split all --arm single --treatment metadata_only
    uv run python scripts/run_evaluation.py --split all --arm agentic --treatment raw
    uv run python scripts/run_evaluation.py --split all --arm agentic --source combined

    # scan once, then score every arm/treatment on that same frozen set:
    uv run python scripts/run_evaluation.py --split all --arm gitleaks_only --source combined \
        --candidates-cache data/processed/all_combined.candidates.json
    uv run python scripts/run_evaluation.py --split all --arm single --treatment masked \
        --source combined --candidates-cache data/processed/all_combined.candidates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
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
from credhunter_x.evaluation.result_files import result_stem
from credhunter_x.gitleaks.parser import parse_gitleaks_report
from credhunter_x.gitleaks.runner import run_gitleaks
from credhunter_x.models.candidate import Candidate
from credhunter_x.models.classification import Label
from credhunter_x.models.evaluation import MetricReport
from credhunter_x.models.treatment import Treatment
from credhunter_x.pipeline.candidate_merge import merge_candidates
from credhunter_x.pipeline.orchestrator import ScanResult, classify_candidates
from credhunter_x.trufflehog.parser import parse_trufflehog_report
from credhunter_x.trufflehog.runner import run_trufflehog

CREDDATA_ROOT = Path("data/creddata_raw/CredData")
RESULTS_DIR = Path("results")

# Bumped whenever Candidate's fields change, so an old cache is rejected
# rather than blowing up inside Candidate(**row).
_CACHE_FORMAT = 1


def _cache_key(source: str) -> dict[str, object]:
    """What a cached candidate set is valid for. Reusing a gitleaks-only
    cache on a --source combined run would score the wrong candidate set
    without any visible sign, so the key is checked on load."""
    return {"format": _CACHE_FORMAT, "source": source, "corpus": str(CREDDATA_ROOT)}


def _load_candidates_cache(path: Path, source: str) -> list[Candidate]:
    """Loads a frozen candidate set instead of re-running the detectors.

    Why this exists: the detectors are re-run on every invocation, and
    trufflehog3 emits its findings in a different order each time (it scans
    in parallel). merge_candidates absorbs the FIRST trufflehog finding that
    matches a given gitleaks one, and a multi-line private key legitimately
    matches many of them -- so a different one survives each run. The
    candidate count is stable; which candidates they are is not. Measured
    across two --split all runs: 6 of 517 differed.

    That is harmless for a single run, but RQ4 compares treatments to each
    other, and "what does masking cost?" only means something if every
    treatment judged the same candidates. Freezing the set makes that
    guarantee explicit rather than hoping the scan repeats itself."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("key") != _cache_key(source):
        raise SystemExit(
            f"{path} was built for {payload.get('key')}, but this run needs "
            f"{_cache_key(source)}. Delete it, point --candidates-cache elsewhere, "
            "or pass --refresh-candidates to rebuild it."
        )
    return [Candidate(**row) for row in payload["candidates"]]


def _save_candidates_cache(path: Path, source: str, candidates: list[Candidate]) -> None:
    """Holds real secret values (matched_value, matched_lines, context) --
    keep it gitignored and never ship it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": _cache_key(source),
        "generated_at": datetime.now(UTC).isoformat(),
        "candidates": [asdict(c) for c in candidates],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"  cached {len(candidates)} candidates to {path}", flush=True)


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


def _generate_candidates(source: str) -> list[Candidate]:
    """Runs the requested detector(s) over the full materialized CredData
    corpus. "combined" runs both independently, then merges out any secret
    both flag so it's counted -- and sent to the LLM -- once, not twice.

    Both parsers are pointed at CREDDATA_ROOT (all 71 repos at once, not
    one repo at a time), so neither has a single repo_id to stamp on its
    candidates -- it's derived here from each candidate's own file_path
    instead, same convention as _repo_id_of/_select_candidates below."""
    gitleaks_candidates: list[Candidate] = []
    trufflehog_candidates: list[Candidate] = []

    if source in ("gitleaks", "combined"):
        print("Running gitleaks against the full materialized corpus...", flush=True)
        findings = run_gitleaks(CREDDATA_ROOT)
        gitleaks_candidates = parse_gitleaks_report(findings, CREDDATA_ROOT)

    if source in ("trufflehog", "combined"):
        print("Running trufflehog3 against the full materialized corpus...", flush=True)
        # .py-only: a single pathological non-Python file elsewhere in the
        # corpus is known to crash trufflehog3's own entropy scan (see
        # run_trufflehog's include_extensions docstring), and every
        # candidate outside a .py file gets discarded by _select_candidates
        # below regardless.
        findings = run_trufflehog(CREDDATA_ROOT, include_extensions=(".py",))
        trufflehog_candidates = parse_trufflehog_report(findings, CREDDATA_ROOT)

    if source == "gitleaks":
        candidates = gitleaks_candidates
    elif source == "trufflehog":
        candidates = trufflehog_candidates
    else:
        candidates = merge_candidates(gitleaks_candidates, trufflehog_candidates)

    return [replace(c, repo_id=_repo_id_of(c.file_path)) for c in candidates]


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
    source: str,
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
    stem = result_stem(arm=arm, treatment=treatment, split=split, source=source, model=model)
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
        "source": source,
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


def _write_detector_results(
    *,
    arm_name: str,
    split: str,
    candidates: list[Candidate],
    gt_rows: list[GroundTruthRow],
    report: MetricReport,
) -> None:
    """Persists a detector-only baseline (gitleaks, trufflehog, or their
    merged combination) the same way _write_results does for the LLM arms,
    so every arm has a comparable results/ file. No classification fields
    here — a scanner alone has no label/confidence/explanation, just a
    flagged file+line."""
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"{arm_name}_raw_{split}"
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
        "arm": arm_name,
        "treatment": "raw",
        "split": split,
        # arm_name already carries the source ("combined_only"), but record
        # it as its own field so every summary is queryable the same way.
        "source": arm_name.removesuffix("_only"),
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


def _run_detector_only(
    candidates: list[Candidate], gt_rows: list[GroundTruthRow], *, arm_name: str
) -> tuple[MetricReport, list[bool]]:
    gt_index = index_ground_truth(gt_rows)
    total_true_count = sum(1 for r in gt_rows if r.ground_truth)
    outcomes = [match_candidate(c, gt_index) for c in candidates]
    correct = [o == MatchOutcome.TRUE_POSITIVE for o in outcomes]
    report = compute_metrics(
        outcomes, total_true_count=total_true_count, arm=arm_name, treatment="raw"
    )
    return report, correct


def _run_llm_arm(
    candidates: list[Candidate],
    gt_rows: list[GroundTruthRow],
    *,
    mode: Mode,
    treatment: Treatment,
    split: str,
    source: str,
    source_label: str,
    delay_seconds: float,
) -> tuple[MetricReport, MetricReport]:
    """Returns (detector_only_report, llm_report), both computed over the
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
    # Pacing is a per-provider concern, so it's a flag rather than a
    # constant: the 8s that a free tier needed costs over an hour of pure
    # sleeping across a 517-candidate run on a paid endpoint. LiteLLMClient
    # already retries rate limits with exponential backoff, so this only
    # reduces how often that has to fire -- see --delay-seconds.
    scan_results = classify_candidates(
        candidates,
        settings=settings,
        scan_config=scan_config,
        source_root=CREDDATA_ROOT,
        delay_seconds=delay_seconds,
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
    detector_report, detector_correct = _run_detector_only(
        survived_candidates, gt_rows, arm_name=source_label
    )

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

    stat, p_value = mcnemar_test(detector_correct, correct)
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
        source=source,
        model=settings.llm_model,
        scan_results=scan_results,
        excluded=excluded,
        gt_rows=gt_rows,
        report=report,
    )
    return detector_report, report


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
    parser.add_argument(
        "--source",
        choices=["gitleaks", "trufflehog", "combined"],
        default="gitleaks",
        help="which scanner(s) generate candidates; 'combined' merges gitleaks + trufflehog3",
    )
    parser.add_argument(
        "--candidates-cache",
        type=Path,
        default=None,
        help=(
            "reuse one frozen candidate set across runs so every arm and treatment is "
            "scored on identical input (built on first use, loaded thereafter). Off by "
            "default -- a scan of a real repository must re-run the detectors to catch "
            "secrets added since. Contains real secret values: keep it gitignored."
        ),
    )
    parser.add_argument(
        "--refresh-candidates",
        action="store_true",
        help="re-scan and overwrite --candidates-cache, e.g. after the corpus changes",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help=(
            "pause between LLM calls, to stay under a provider's rate limit "
            "(default 1.0; raise it for a constrained free tier, 0 to go flat out -- "
            "the client retries rate limits with backoff either way)"
        ),
    )
    args = parser.parse_args()
    if args.delay_seconds < 0:
        parser.error("--delay-seconds cannot be negative")
    if args.refresh_candidates and args.candidates_cache is None:
        parser.error("--refresh-candidates only means something with --candidates-cache")
    treatment = Treatment(args.treatment)
    mode = Mode.SINGLE if args.arm == "single" else Mode.AGENTIC
    source_label = f"{args.source}_only"

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

    cache = args.candidates_cache
    if cache is not None and cache.exists() and not args.refresh_candidates:
        print(f"Loading cached candidates from {cache} (skipping the scan)...", flush=True)
        all_candidates = _load_candidates_cache(cache, args.source)
        print(f"  {len(all_candidates)} candidates loaded", flush=True)
    else:
        all_candidates = _generate_candidates(args.source)
        if cache is not None:
            _save_candidates_cache(cache, args.source, all_candidates)

    candidates = _select_candidates(all_candidates, repo_ids)
    print(f"  {len(candidates)} .py candidates fall within the '{args.split}' split", flush=True)

    detector_report, _ = _run_detector_only(candidates, gt_rows, arm_name=source_label)
    _write_detector_results(
        arm_name=source_label,
        split=args.split,
        candidates=candidates,
        gt_rows=gt_rows,
        report=detector_report,
    )

    if args.arm == "gitleaks_only":
        _print_report(detector_report)
        return

    detector_report, llm_report = _run_llm_arm(
        candidates,
        gt_rows,
        mode=mode,
        treatment=treatment,
        split=args.split,
        source=args.source,
        source_label=source_label,
        delay_seconds=args.delay_seconds,
    )

    print()
    print(f"--- {source_label} (for comparison / McNemar pairing) ---")
    _print_report(detector_report)
    print()
    print(f"--- {args.arm} ---")
    _print_report(llm_report)


if __name__ == "__main__":
    main()
