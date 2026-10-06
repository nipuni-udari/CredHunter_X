"""Exports a sample of true_secret remediations for hand-labelling, the
manual step behind the RQ3 kappa check. One row per (candidate, required
element), with an empty human_present column to fill in by hand.
No LLM calls; it only samples.

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

from credhunter_x.config.settings import Settings
from credhunter_x.evaluation.remediation_reference import load_remediation_reference
from credhunter_x.evaluation.result_files import result_stem

RESULTS_DIR = Path("results")
_DEFAULT_SAMPLE_SIZE = 45


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Export a sample of remediations for hand-labeling"
    )
    parser.add_argument(
        "--arm",
        choices=["single", "agentic", "both"],
        required=True,
        help="'both' splits the sample evenly, so kappa validates the checker "
        "on each arm's writing style rather than only one",
    )
    parser.add_argument(
        "--treatment",
        choices=["raw", "masked", "pseudonymised", "metadata_only"],
        default="raw",
    )
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--n", type=int, default=_DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source", choices=["gitleaks", "trufflehog", "combined"], default="combined"
    )
    parser.add_argument("--model", default=None, help="defaults to LLM_MODEL from .env")
    parser.add_argument("--out", type=Path, default=Path("results/hand_labeling_sample.csv"))
    args = parser.parse_args()

    settings = Settings()
    reference = load_remediation_reference()
    rng = random.Random(args.seed)
    arms = ["single", "agentic"] if args.arm == "both" else [args.arm]
    per_arm = args.n // len(arms)

    sampled_rows: list[dict] = []
    taken: set[str] = set()
    for arm in arms:
        # result_stem gives the actual file name.
        stem = result_stem(
            arm=arm,
            treatment=args.treatment,
            split=args.split,
            source=args.source,
            model=args.model or settings.llm_model,
        )
        jsonl_path = RESULTS_DIR / f"{stem}.jsonl"
        if not jsonl_path.exists():
            print(f"missing {jsonl_path} -- run run_evaluation.py for this arm first")
            return

        rows = [json.loads(x) for x in jsonl_path.read_text(encoding="utf-8").splitlines() if x]
        scoreable = [
            r
            for r in rows
            if r["label"] == "true_secret"
            and r["rule_id"] in reference
            and reference[r["rule_id"]].required_elements
            # Skip a file:line already sampled, so one candidate isn't labelled twice.
            and r["candidate_id"] not in taken
        ]
        if not scoreable:
            print(f"nothing to sample for {arm} -- reference has no filled-in elements yet")
            return

        # Stratify by rule_id: an equal share per rule, or everything a rule has
        # if it has fewer.
        by_rule: dict[str, list[dict]] = {}
        for row in scoreable:
            by_rule.setdefault(row["rule_id"], []).append(row)

        picked: list[dict] = []
        per_rule_target = max(1, per_arm // len(by_rule))
        for rule_rows in by_rule.values():
            rng.shuffle(rule_rows)
            picked.extend(rule_rows[:per_rule_target])
        rng.shuffle(picked)
        picked = picked[:per_arm]

        for row in picked:
            row["_arm"] = arm
            taken.add(row["candidate_id"])
        sampled_rows.extend(picked)
        print(f"  {arm}: {len(picked)} remediations sampled")

    # Interleave the arms so the labeller sees them mixed.
    rng.shuffle(sampled_rows)

    args.out.parent.mkdir(exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Keep arm and treatment on each row so the kappa script reads the right run.
        writer.writerow(
            [
                "candidate_id",
                "rule_id",
                "arm",
                "treatment",
                "element_no",
                "remediation",
                "element_text",
                "human_present",
            ]
        )
        for row in sampled_rows:
            # element_no tells the two rows of one remediation apart at a glance.
            for i, element in enumerate(reference[row["rule_id"]].required_elements, 1):
                writer.writerow(
                    [
                        row["candidate_id"],
                        row["rule_id"],
                        row["_arm"],
                        args.treatment,
                        i,
                        row["remediation"],
                        element,
                        "",
                    ]
                )

    n_element_rows = sum(len(reference[r["rule_id"]].required_elements) for r in sampled_rows)
    print(f"exported {len(sampled_rows)} remediations ({n_element_rows} rows) to {args.out}")
    print(
        "fill in the human_present column by hand (TRUE/FALSE), then run "
        "compare_hand_labels_to_checker.py"
    )


if __name__ == "__main__":
    main()
