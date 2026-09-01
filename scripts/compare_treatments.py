"""Prints the RQ4 four-treatment comparison table (precision, recall, F1,
Δf1 vs raw, bytes of real secret sent) for one arm, from results already on
disk — no LLM calls, pure aggregation over results/*_summary.json and
results/*.jsonl. Run scripts/run_evaluation.py for all four treatments
first (same --arm/--split) before this has anything to read.

Usage:
    uv run python scripts/compare_treatments.py --arm single --split all
    uv run python scripts/compare_treatments.py --arm agentic --split all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from credhunter_x.evaluation.result_files import result_stem

RESULTS_DIR = Path("results")
_TREATMENTS = ["raw", "masked", "pseudonymised", "metadata_only"]


def _bytes_of_real_secret_sent(jsonl_path: Path, treatment: str) -> int:
    if treatment != "raw":
        return 0
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line]
    if any("real_secret_length" not in row for row in rows):
        sys.exit(
            f"{jsonl_path} predates the real_secret_length field -- "
            "re-run this treatment before comparing"
        )
    return sum(row["real_secret_length"] for row in rows)


def _discover_run(arm: str, split: str) -> tuple[str, str]:
    """Which (source, model) run to report on, read from the summaries'
    own contents rather than guessed from a default.

    A wrong guess here is silent: it reports a different run's numbers, or
    prints "missing" for results that are sitting right there. Every
    summary records its own source and model, so infer when there is only
    one candidate run and make the user choose when there is more than
    one."""
    found: set[tuple[str, str]] = set()
    for path in RESULTS_DIR.glob(f"{arm}_*_{split}_*_summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("arm") != arm or summary.get("split") != split:
            continue
        # Summaries written before source/model were recorded can't identify
        # a run, and their files no longer follow the current naming either.
        source, model = summary.get("source"), summary.get("model")
        if source and model:
            found.add((source, model))
    if not found:
        sys.exit(f"no {arm}/{split} results in {RESULTS_DIR}/ -- run run_evaluation.py first")
    if len(found) > 1:
        listing = "\n".join(f"  --source {s} --model {m}" for s, m in sorted(found))
        sys.exit(f"several {arm}/{split} runs on disk; pick one with:\n{listing}")
    return found.pop()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Compare all four RQ4 treatments for one arm")
    parser.add_argument("--arm", choices=["single", "agentic"], required=True)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument(
        "--source",
        choices=["gitleaks", "trufflehog", "combined"],
        default=None,
        help="which scanner produced the candidates; inferred when only one run exists",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id, e.g. openai/gpt-5.6-luna; inferred when only one run exists",
    )
    args = parser.parse_args()

    source, model = args.source, args.model
    if source is None or model is None:
        discovered_source, discovered_model = _discover_run(args.arm, args.split)
        source = source or discovered_source
        model = model or discovered_model
    print(f"arm={args.arm}  split={args.split}  source={source}  model={model}")

    rows = []
    raw_f1 = None
    for treatment in _TREATMENTS:
        stem = result_stem(
            arm=args.arm, treatment=treatment, split=args.split, source=source, model=model
        )
        summary_path = RESULTS_DIR / f"{stem}_summary.json"
        jsonl_path = RESULTS_DIR / f"{stem}.jsonl"
        if not summary_path.exists():
            print(f"  missing: {summary_path} -- run run_evaluation.py for this treatment first")
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary["metrics"]
        f1 = metrics["f1"]
        if treatment == "raw":
            raw_f1 = f1
        delta = "--" if raw_f1 is None else f"{f1 - raw_f1:+.3f}"
        secret_bytes = _bytes_of_real_secret_sent(jsonl_path, treatment)
        rows.append((treatment, metrics["precision"], metrics["recall"], f1, delta, secret_bytes))

    print()
    header = (
        f"{'treatment':<14}{'precision':>10}{'recall':>10}{'f1':>10}"
        f"{'Δf1 vs raw':>12}{'bytes sent':>12}"
    )
    print(header)
    print("-" * len(header))
    for treatment, precision, recall, f1, delta, secret_bytes in rows:
        print(
            f"{treatment:<14}{precision:>10.3f}{recall:>10.3f}{f1:>10.3f}"
            f"{delta:>12}{secret_bytes:>12}"
        )


if __name__ == "__main__":
    main()
