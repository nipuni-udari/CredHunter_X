"""Compares the student's hand-filled labels (from
export_for_hand_labeling.py) against the automated element-checker's own
output on the same candidates, reporting Cohen's kappa — the validation
step that decides whether the automated checker is trustworthy at scale
(the scope document's own rule of thumb: >= ~0.85 raw agreement). No
labels are read from anywhere but the CSV the student filled in by hand.
Costs real LLM quota (re-scores every sampled candidate). Not part of CI,
run manually.

Usage:
    uv run python scripts/compare_hand_labels_to_checker.py \\
        --labels results/hand_labeling_sample.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from credhunter_x.config.settings import Settings
from credhunter_x.evaluation.metrics import cohens_kappa
from credhunter_x.evaluation.remediation_reference import load_remediation_reference
from credhunter_x.evaluation.remediation_scoring import ElementCheckParsingError, score_remediation
from credhunter_x.gitleaks.parser import parse_gitleaks_report
from credhunter_x.gitleaks.runner import run_gitleaks
from credhunter_x.guard.leak_guard import LeakGuard
from credhunter_x.llm.guarded_client import GuardedLLMClient
from credhunter_x.llm.litellm_client import LiteLLMClient
from credhunter_x.masking.secret_registry import SecretRegistry

CREDDATA_ROOT = Path("data/creddata_raw/CredData")


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

    print("Re-running gitleaks to populate the leak guard's registry...", flush=True)
    findings = run_gitleaks(CREDDATA_ROOT)
    all_candidates = parse_gitleaks_report(findings, CREDDATA_ROOT)

    registry = SecretRegistry()
    registry.register_candidates(all_candidates)
    guard = LeakGuard(registry)
    settings = Settings()
    llm_client = LiteLLMClient(model=settings.llm_model, api_key=settings.llm_api_key)
    client = GuardedLLMClient(llm_client, guard)

    reference = load_remediation_reference()

    human_labels: list[bool] = []
    checker_labels: list[bool] = []
    skipped = 0
    for candidate_id, rows in rows_by_candidate.items():
        rule_id = rows[0]["rule_id"]
        remediation = rows[0]["remediation"]
        required_elements = [row["element_text"] for row in rows]
        entry = reference.get(rule_id)
        if entry is None or entry.required_elements != required_elements:
            print(f"  skipping {candidate_id}: remediation_reference.yaml has changed since export")
            skipped += 1
            continue

        try:
            result = score_remediation(
                client,
                candidate_id=candidate_id,
                rule_id=rule_id,
                remediation=remediation,
                required_elements=required_elements,
            )
        except ElementCheckParsingError as exc:
            print(f"  skipping {candidate_id}: {exc}")
            skipped += 1
            continue

        for row, checker_value in zip(rows, result.element_present, strict=True):
            human_value = _parse_bool(row["human_present"])
            assert human_value is not None  # already validated above
            human_labels.append(human_value)
            checker_labels.append(checker_value)

    if skipped:
        print(f"  {skipped} candidate(s) skipped", flush=True)
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
