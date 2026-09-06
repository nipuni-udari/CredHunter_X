"""Computes E1's full metric set for every run already on disk.

The eight runs were scored before candidate-level F1, FPR, MCC, bootstrap
CIs and the per-rule breakdown existed in the harness. All of them are
derivable from the stored rows, so this recomputes them: read-only, no LLM
calls, no re-scan, and the existing *_summary.json files are untouched.

Usage:
    uv run python scripts/rescore_e1_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from credhunter_x.evaluation.labeler import MatchOutcome
from credhunter_x.evaluation.metrics import (
    bootstrap_ci,
    candidate_confusion,
    false_positive_rate,
    matthews_corrcoef,
    stratify_by_rule,
)

RESULTS_DIR = Path("results")
OUT_PATH = RESULTS_DIR / "rq1_e1_metrics.json"
TOTAL_TRUE = 663  # every GroundTruth=='T' row in the Python-filtered corpus


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _score(rows: list[dict], *, flag_everything: bool = False) -> dict:
    outcomes = [MatchOutcome(r["ground_truth_outcome"]) for r in rows]
    flagged = [True if flag_everything else r["label"] == "true_secret" for r in rows]
    rule_ids = [r["rule_id"] for r in rows]

    counts = candidate_confusion(outcomes, flagged)
    ci_low, ci_high = bootstrap_ci(outcomes, flagged, lambda c: c.precision)
    return {
        "n_rows": len(rows),
        "n_scored": counts.true_positive
        + counts.false_positive
        + counts.false_negative
        + counts.true_negative,
        "true_positive": counts.true_positive,
        "false_positive": counts.false_positive,
        "false_negative": counts.false_negative,
        "true_negative": counts.true_negative,
        "precision": counts.precision,
        "precision_ci_low": ci_low,
        "precision_ci_high": ci_high,
        "candidate_recall": counts.recall,
        "candidate_f1": counts.f1,
        "corpus_recall": counts.true_positive / TOTAL_TRUE,
        "false_positive_rate": false_positive_rate(counts),
        "mcc": matthews_corrcoef(counts),
        "by_rule": {
            rule: {
                "n": c.true_positive + c.false_positive + c.false_negative + c.true_negative,
                "true_positive": c.true_positive,
                "false_positive": c.false_positive,
                "false_negative": c.false_negative,
                "true_negative": c.true_negative,
                "precision": c.precision,
                "recall": c.recall,
            }
            for rule, c in sorted(stratify_by_rule(rule_ids, outcomes, flagged).items())
        },
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    paths = sorted(
        p
        for p in RESULTS_DIR.glob("*.jsonl")
        if p.stem.startswith(("single_", "agentic_")) and "_summary" not in p.stem
    )
    if not paths:
        print("no arm results found in results/ -- run run_evaluation.py first")
        return

    out: dict[str, dict] = {}
    # The detector flags every candidate it generated, so its own row set
    # is any arm's rows with flagged forced true.
    out["detector_combined"] = _score(_load(paths[0]), flag_everything=True)

    header = (
        f"{'run':<26}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}"
        f"{'prec':>8}{'95% CI':>18}{'cand F1':>9}{'FPR':>7}{'MCC':>8}"
    )
    print(header)
    print("-" * len(header))

    def _row(name: str, s: dict) -> None:
        ci = f"[{s['precision_ci_low']:.3f}, {s['precision_ci_high']:.3f}]"
        print(
            f"{name:<26}{s['true_positive']:>5}{s['false_positive']:>5}"
            f"{s['false_negative']:>5}{s['true_negative']:>5}"
            f"{s['precision']:>8.3f}{ci:>18}{s['candidate_f1']:>9.3f}"
            f"{s['false_positive_rate']:>7.3f}{s['mcc']:>+8.3f}"
        )

    _row("detector (flags all)", out["detector_combined"])
    for path in paths:
        name = path.stem.replace("_all_combined_gpt-5.6-luna", "")
        out[path.stem] = _score(_load(path))
        _row(name, out[path.stem])

    print()
    print("per-rule, single_raw vs the detector (precision):")
    print(f"  {'rule':<26}{'n':>5}{'detector':>10}{'ArmA':>8}")
    detector_rules = out["detector_combined"]["by_rule"]
    arm_a = next((k for k in out if k.startswith("single_raw")), None)
    if arm_a:
        for rule, stats in sorted(out[arm_a]["by_rule"].items(), key=lambda kv: -kv[1]["n"]):
            det = detector_rules.get(rule, {}).get("precision", 0.0)
            print(f"  {rule:<26}{stats['n']:>5}{det:>10.3f}{stats['precision']:>8.3f}")

    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print("Candidate-level, LOST excluded. corpus_recall (denominator 663) is in the JSON.")
    print(f"written to {OUT_PATH} -- no *_summary.json was modified")


if __name__ == "__main__":
    main()
