"""Exports a sample of real true_secret remediations for hand-labeling —
the manual step behind RQ3's Cohen's kappa validation of the automated
element-checker. One row per (candidate, required element) pair, with a
blank human_present column for the student to fill in by hand (e.g. in a
spreadsheet). No LLM calls, no fabricated labels — pure sampling.

Requires remediation_reference.yaml to actually have required_elements
filled in; nothing to export from the placeholder scaffold. Not part of
CI, run manually.

Usage:
    uv run python scripts/export_for_hand_labeling.py --arm single --treatment raw --split all
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from credhunter_x.evaluation.remediation_reference import load_remediation_reference

RESULTS_DIR = Path("results")
_DEFAULT_SAMPLE_SIZE = 45


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Export a sample of remediations for hand-labeling"
    )
    parser.add_argument("--arm", choices=["single", "agentic"], required=True)
    parser.add_argument(
        "--treatment",
        choices=["raw", "masked", "pseudonymised", "metadata_only"],
        default="raw",
    )
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--n", type=int, default=_DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results/hand_labeling_sample.csv"))
    args = parser.parse_args()

    stem = f"{args.arm}_{args.treatment}_{args.split}"
    jsonl_path = RESULTS_DIR / f"{stem}.jsonl"
    if not jsonl_path.exists():
        print(f"missing {jsonl_path} -- run run_evaluation.py for this arm/treatment/split first")
        return

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line]
    reference = load_remediation_reference()
    scoreable = [
        r
        for r in rows
        if r["label"] == "true_secret"
        and r["rule_id"] in reference
        and reference[r["rule_id"]].required_elements
    ]
    if not scoreable:
        print("nothing to sample -- remediation_reference.yaml has no filled-in elements yet")
        return

    # Stratify by rule_id: sample an even share per rule, falling back to
    # whatever's available if a rule has fewer candidates than its share.
    by_rule: dict[str, list[dict]] = {}
    for row in scoreable:
        by_rule.setdefault(row["rule_id"], []).append(row)

    rng = random.Random(args.seed)
    per_rule_target = max(1, args.n // len(by_rule))
    sampled_rows = []
    for rule_rows in by_rule.values():
        rng.shuffle(rule_rows)
        sampled_rows.extend(rule_rows[:per_rule_target])
    rng.shuffle(sampled_rows)
    sampled_rows = sampled_rows[: args.n]

    args.out.parent.mkdir(exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rule_id", "remediation", "element_text", "human_present"])
        for row in sampled_rows:
            for element in reference[row["rule_id"]].required_elements:
                writer.writerow(
                    [row["candidate_id"], row["rule_id"], row["remediation"], element, ""]
                )

    n_element_rows = sum(len(reference[r["rule_id"]].required_elements) for r in sampled_rows)
    print(f"exported {len(sampled_rows)} remediations ({n_element_rows} rows) to {args.out}")
    print(
        "fill in the human_present column by hand (TRUE/FALSE), then run "
        "compare_hand_labels_to_checker.py"
    )


if __name__ == "__main__":
    main()
