"""Compares the student's hand-filled labels (from
export_for_hand_labeling.py) against the element-checker's own verdicts,
reporting Cohen's kappa — the validation step that decides whether the
automated checker is trustworthy at scale (the scope document's own rule
of thumb: >= ~0.85 raw agreement). No labels are read from anywhere but
the CSV the student filled in by hand. No LLM calls, run manually.

Verdicts are READ from the scored run's --out file, never re-generated.
Re-scoring would compare the hand labels against a fresh set of checker
answers rather than the ones the dissertation reports, and the checker
flips ~4.7% of verdicts between runs (rq3_checker_stability.json), so the
kappa would not apply to the reported pass rates.

Usage:
    uv run python scripts/compare_hand_labels_to_checker.py \\
        --labels results/hand_labeling_sample.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from credhunter_x.evaluation.metrics import cohens_kappa
from credhunter_x.evaluation.remediation_reference import load_remediation_reference

RESULTS_DIR = Path("results")


def _parse_bool(value: str) -> bool | None:
    v = value.strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    return None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Validate the automated checker against hand labels"
    )
    parser.add_argument("--labels", type=Path, default=Path("results/hand_labeling_sample.csv"))
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="scored run to validate against; defaults to the arm/treatment named in the CSV",
    )
    args = parser.parse_args()

    if not args.labels.exists():
        print(
            f"missing {args.labels} -- run export_for_hand_labeling.py first, "
            "then fill it in by hand"
        )
        return

    rows_by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.labels.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_candidate[row["candidate_id"]].append(row)

    unfilled = [
        row
        for rows in rows_by_candidate.values()
        for row in rows
        if _parse_bool(row["human_present"]) is None
    ]
    if unfilled:
        print(
            f"{len(unfilled)} row(s) still have a blank/unrecognised human_present value "
            f"-- fill in every row of {args.labels} before running this"
        )
        return

    scores_path = args.scores
    if scores_path is None:
        first = next(iter(rows_by_candidate.values()))[0]
        arm, treatment = first.get("arm"), first.get("treatment")
        if not arm or not treatment:
            print("CSV has no arm/treatment columns -- pass --scores explicitly")
            return
        scores_path = RESULTS_DIR / f"rq3_scores_{arm}_{treatment}.json"
    if not scores_path.exists():
        print(f"missing {scores_path} -- score that run with --out first")
        return

    scored = json.loads(scores_path.read_text(encoding="utf-8"))
    verdicts = {r["candidate_id"]: r for r in scored["rows"]}
    print(f"validating against {scores_path} ({len(verdicts)} scored rows, no LLM calls)")

    reference = load_remediation_reference()

    human_labels: list[bool] = []
    checker_labels: list[bool] = []
    skipped = 0
    for candidate_id, rows in rows_by_candidate.items():
        rule_id = rows[0]["rule_id"]
        required_elements = [row["element_text"] for row in rows]
        entry = reference.get(rule_id)
        if entry is None or entry.required_elements != required_elements:
            print(f"  skipping {candidate_id}: remediation_reference.yaml has changed since export")
            skipped += 1
            continue

        scored_row = verdicts.get(candidate_id)
        if scored_row is None:
            print(f"  skipping {candidate_id}: not in {scores_path.name}")
            skipped += 1
            continue
        # Match on element text, not position -- a reordered reference would
        # otherwise silently pair a human answer with the wrong verdict.
        by_element = dict(
            zip(scored_row["required_elements"], scored_row["element_present"], strict=True)
        )

        for row in rows:
            checker_value = by_element.get(row["element_text"])
            if checker_value is None:
                print(f"  skipping one element of {candidate_id}: text not in the scored run")
                skipped += 1
                continue
            human_value = _parse_bool(row["human_present"])
            assert human_value is not None  # already validated above
            human_labels.append(human_value)
            checker_labels.append(checker_value)

    if skipped:
        print(f"  {skipped} row(s)/candidate(s) skipped", flush=True)
    if len(human_labels) < 2:
        print("not enough comparable labels to compute kappa")
        return

    kappa = cohens_kappa(human_labels, checker_labels)
    agreement = sum(1 for h, c in zip(human_labels, checker_labels, strict=True) if h == c) / len(
        human_labels
    )

    print()
    print("=" * 60)
    print(f"n element judgements   : {len(human_labels)}")
    print(f"raw agreement          : {agreement:.3f}")
    print(f"cohen's kappa          : {kappa:.3f}")
    trustworthy = "yes" if agreement >= 0.85 else "no"
    print(f"trustworthy at scale?  : {trustworthy} (>= 0.85 raw agreement)")
    print("=" * 60)


if __name__ == "__main__":
    main()
