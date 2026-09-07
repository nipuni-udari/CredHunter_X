"""Scores every true_secret remediation in a results/*.jsonl file against
remediation_reference.yaml's pre-registered required elements, reporting a
pass rate (all required elements present) with a Wilson 95% CI per rule
and overall. Manual, costs real LLM quota — not part of CI.

Nothing to score until remediation_reference.yaml's placeholder
required_elements are actually filled in (see that file's own header
comment) — that content is a real methodological step, not this script's
job to generate.

Usage:
    uv run python scripts/score_remediation_quality.py --arm single --treatment raw --split all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from credhunter_x.config.settings import Settings
from credhunter_x.evaluation.metrics import wilson_score_interval
from credhunter_x.evaluation.remediation_reference import load_remediation_reference
from credhunter_x.evaluation.remediation_scoring import (
    ElementCheckParsingError,
    ElementCheckResult,
    score_remediation,
)
from credhunter_x.evaluation.result_files import result_stem
from credhunter_x.guard.errors import LeakError
from credhunter_x.guard.leak_guard import LeakGuard
from credhunter_x.llm.guarded_client import GuardedLLMClient
from credhunter_x.llm.litellm_client import LiteLLMClient, LiteLLMClientError
from credhunter_x.masking.secret_registry import SecretRegistry
from credhunter_x.models.candidate import Candidate

CREDDATA_ROOT = Path("data/creddata_raw/CredData")
RESULTS_DIR = Path("results")
CANDIDATES_CACHE = Path("data/processed/all_combined.candidates.json")


def _print_report(by_rule: dict[str, list[bool]], all_passes: list[bool]) -> None:
    print()
    print("=" * 60)
    for rule_id, passes in sorted(by_rule.items()):
        n, n_pass = len(passes), sum(passes)
        lower, upper = wilson_score_interval(n_pass, n)
        print(f"{rule_id:<20} {n_pass}/{n} pass  (95% CI: {lower:.3f}-{upper:.3f})")

    n, n_pass = len(all_passes), sum(all_passes)
    lower, upper = wilson_score_interval(n_pass, n)
    verdict = "CLEARS" if lower >= 0.8 else "DOES NOT CLEAR"
    print("-" * 60)
    print(f"{'OVERALL':<20} {n_pass}/{n} pass  (95% CI: {lower:.3f}-{upper:.3f})")
    print(f"{'vs 80% target':<20} {verdict} (CI lower bound {lower:.3f} vs 0.80)")
    print("=" * 60)


DENOMINATOR_NOTE = (
    "per-arm pass rate: over each arm's full scoreable set. "
    "arm comparison: paired on the candidates BOTH arms flagged -- rows only "
    "one arm flagged have nothing to pair against and are excluded there."
)


def _row(result: ElementCheckResult) -> dict[str, object]:
    """One scored remediation. Shared by the live sidecar and the summary
    file so the durable copy and the reported copy cannot drift apart."""
    return {
        "candidate_id": result.candidate_id,
        "rule_id": result.rule_id,
        "required_elements": result.required_elements,
        "element_present": result.element_present,
        "optional_elements": result.optional_elements,
        "optional_present": result.optional_present,
        "gate_answers": result.gate_answers,
        "pass_rate": result.pass_rate,
        "passed": result.pass_rate == 1.0,
        "raw_model_output": result.raw_model_output,
    }


def _open_sidecar(out_path: Path) -> TextIO:
    """Rows land here as they complete. The summary JSON is written once at
    the end, so without this a kill at row 400 of 459 loses all 400."""
    path = out_path.with_suffix(".rows.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"streaming rows to {path}", flush=True)
    return path.open("w", encoding="utf-8")


def _write_rows(
    path: Path,
    results: list[ElementCheckResult],
    *,
    stem: str,
    args: argparse.Namespace,
    blocked: list[str] | None = None,
    failed: list[str] | None = None,
) -> None:
    """One record per scored remediation, plus the run's own settings.

    The checker is an LLM and does not repeat itself exactly, so these
    verdicts cannot be regenerated -- a reported pass rate is only auditable
    if the rows behind it were written down at the time. Cohen's kappa also
    needs the per-row answers to line up against the hand-labels.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_run": stem,
        "arm": args.arm,
        "treatment": args.treatment,
        "checker_model": args.model,
        "limit": args.limit,
        "excluded_rules": args.exclude_rule,
        "n_scored": len(results),
        "n_passed": sum(1 for r in results if r.pass_rate == 1.0),
        "guard_blocked": blocked or [],
        "client_failed": failed or [],
        "denominators": DENOMINATOR_NOTE,
        "rows": [_row(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n{len(results)} row(s) written to {path}")


def _print_optional_report(results: list[ElementCheckResult]) -> None:
    """Optional elements are reported, never required to pass -- the
    reference file's own grading_policy. They carry the finer signal once
    the required pair saturates."""
    tallies: dict[str, list[bool]] = {}
    for result in results:
        for element, present in zip(result.optional_elements, result.optional_present, strict=True):
            tallies.setdefault(element, []).append(present)
    if not tallies:
        return

    print()
    print("optional elements (reported, never gating a pass)")
    print("-" * 60)
    for element, presences in sorted(tallies.items(), key=lambda kv: -len(kv[1])):
        n, n_present = len(presences), sum(presences)
        label = element if len(element) <= 46 else element[:43] + "..."
        print(f"  {label:<48} {n_present}/{n}  ({n_present / n:.0%})")

    waived = sum(1 for r in results if r.gate_answers and not all(r.gate_answers))
    if waived:
        print(f"\n  {waived} row(s) had a gated element waived")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Score remediation quality against the pre-registered reference"
    )
    parser.add_argument("--arm", choices=["single", "agentic"], required=True)
    parser.add_argument(
        "--treatment",
        choices=["raw", "masked", "pseudonymised", "metadata_only"],
        default="raw",
    )
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument(
        "--source", choices=["gitleaks", "trufflehog", "combined"], default="combined"
    )
    parser.add_argument("--model", default=None, help="defaults to LLM_MODEL from .env")
    parser.add_argument("--limit", type=int, default=None, help="score at most N rows (pilot runs)")
    parser.add_argument(
        "--exclude-rule",
        action="append",
        default=[],
        metavar="RULE_ID",
        help="skip a rule family; repeatable",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help=(
            "write per-row verdicts to a JSON file. Use it on every real run: "
            "the checker is not deterministic, so a run scored without --out "
            "cannot be reconstructed, and Cohen's kappa needs the per-row answers"
        ),
    )
    args = parser.parse_args()

    settings = Settings()
    stem = result_stem(
        arm=args.arm,
        treatment=args.treatment,
        split=args.split,
        source=args.source,
        model=args.model or settings.llm_model,
    )
    jsonl_path = RESULTS_DIR / f"{stem}.jsonl"
    if not jsonl_path.exists():
        print(f"missing {jsonl_path} -- run run_evaluation.py for this arm/treatment/split first")
        return

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line]
    scoreable_rows = [r for r in rows if r["label"] == "true_secret"]
    print(
        f"{len(scoreable_rows)} true_secret remediations to consider (of {len(rows)} total rows)",
        flush=True,
    )

    reference = load_remediation_reference()

    # The frozen candidate set, not a fresh gitleaks scan: re-scanning
    # would rebuild a registry from gitleaks alone, and every trufflehog
    # candidate would then fail the id lookup below and be skipped.
    if not CANDIDATES_CACHE.exists():
        print(f"missing {CANDIDATES_CACHE} -- build it with run_evaluation.py --candidates-cache")
        return
    payload = json.loads(CANDIDATES_CACHE.read_text(encoding="utf-8"))
    candidates = [Candidate(**row) for row in payload["candidates"]]
    candidates_by_id = {c.id: c for c in candidates}
    print(f"registry built from {len(candidates)} frozen candidates", flush=True)

    registry = SecretRegistry()
    registry.register_candidates(candidates)
    guard = LeakGuard(registry)
    llm_client = LiteLLMClient(model=settings.llm_model, api_key=settings.llm_api_key)
    client = GuardedLLMClient(llm_client, guard)

    results: list[ElementCheckResult] = []
    skipped = 0
    blocked: list[str] = []
    failed: list[str] = []
    sidecar = _open_sidecar(Path(args.out)) if args.out else None
    for row in scoreable_rows:
        if args.limit is not None and len(results) >= args.limit:
            break
        if row["rule_id"] in args.exclude_rule:
            skipped += 1
            continue
        entry = reference.get(row["rule_id"])
        if entry is None or not entry.required_elements:
            skipped += 1
            continue
        if row["candidate_id"] not in candidates_by_id:
            skipped += 1
            continue

        try:
            result = score_remediation(
                client,
                candidate_id=row["candidate_id"],
                rule_id=row["rule_id"],
                remediation=row["remediation"],
                required_elements=entry.required_elements,
                optional_elements=entry.optional_elements,
                required_gates=entry.required_gates,
            )
        except ElementCheckParsingError as exc:
            print(f"  skipping {row['candidate_id']}: {exc}", flush=True)
            skipped += 1
            continue
        except LeakError:
            # The guard already refused the call -- nothing left the machine.
            # Dropping the row rather than aborting mirrors the orchestrator's
            # skip_candidate_on_error, which is why the classification runs
            # survived the same block. Counted separately from `skipped`
            # because an unscoreable row is a finding, not a technicality.
            print(f"  guard blocked {row['candidate_id']} -- excluded", flush=True)
            blocked.append(row["candidate_id"])
            continue
        except LiteLLMClientError as exc:
            # Same policy as the orchestrator: a transport failure drops the
            # row, it does not end a 459-call run.
            print(f"  client error on {row['candidate_id']}: {exc}", flush=True)
            failed.append(row["candidate_id"])
            continue
        results.append(result)
        if sidecar is not None:
            sidecar.write(json.dumps(_row(result)) + "\n")
            sidecar.flush()

    if sidecar is not None:
        sidecar.close()
    if failed:
        print(f"  {len(failed)} row(s) lost to client errors:", flush=True)
        for cid in failed:
            print(f"    {cid}", flush=True)
    if blocked:
        print(
            f"  {len(blocked)} row(s) blocked by the egress guard and excluded from scoring:",
            flush=True,
        )
        for cid in blocked:
            print(f"    {cid}", flush=True)
    if skipped:
        print(
            f"  {skipped} row(s) skipped (no reference entry, no candidate match, "
            "or unparseable answer)",
            flush=True,
        )
    if not results:
        print("no scoreable remediations -- nothing to report")
        return

    # "pass" = every required element present, matching "remediations match
    # a pre-registered reference standard" literally, not partial credit.
    by_rule: dict[str, list[bool]] = {}
    for result in results:
        by_rule.setdefault(result.rule_id, []).append(result.pass_rate == 1.0)
    all_passes = [result.pass_rate == 1.0 for result in results]

    _print_report(by_rule, all_passes)
    _print_optional_report(results)

    print(f"\n{DENOMINATOR_NOTE}")

    if args.out:
        _write_rows(
            Path(args.out), results, stem=stem, args=args, blocked=blocked, failed=failed
        )


if __name__ == "__main__":
    main()
