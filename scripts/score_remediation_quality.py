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

from credhunter_x.config.settings import Settings
from credhunter_x.evaluation.metrics import wilson_score_interval
from credhunter_x.evaluation.remediation_reference import load_remediation_reference
from credhunter_x.evaluation.remediation_scoring import (
    ElementCheckParsingError,
    ElementCheckResult,
    score_remediation,
)
from credhunter_x.gitleaks.parser import parse_gitleaks_report
from credhunter_x.gitleaks.runner import run_gitleaks
from credhunter_x.guard.leak_guard import LeakGuard
from credhunter_x.llm.guarded_client import GuardedLLMClient
from credhunter_x.llm.litellm_client import LiteLLMClient
from credhunter_x.masking.secret_registry import SecretRegistry

CREDDATA_ROOT = Path("data/creddata_raw/CredData")
RESULTS_DIR = Path("results")


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
    args = parser.parse_args()

    stem = f"{args.arm}_{args.treatment}_{args.split}"
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

    print("Re-running gitleaks to populate the leak guard's registry...", flush=True)
    findings = run_gitleaks(CREDDATA_ROOT)
    candidates = parse_gitleaks_report(findings, CREDDATA_ROOT)
    candidates_by_id = {c.id: c for c in candidates}

    registry = SecretRegistry()
    registry.register_candidates(candidates)
    guard = LeakGuard(registry)
    settings = Settings()
    llm_client = LiteLLMClient(model=settings.llm_model, api_key=settings.llm_api_key)
    client = GuardedLLMClient(llm_client, guard)

    results: list[ElementCheckResult] = []
    skipped = 0
    for row in scoreable_rows:
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
            )
        except ElementCheckParsingError as exc:
            print(f"  skipping {row['candidate_id']}: {exc}", flush=True)
            skipped += 1
            continue
        results.append(result)

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


if __name__ == "__main__":
    main()
