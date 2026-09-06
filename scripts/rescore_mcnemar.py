"""Recomputes RQ1's detector-vs-LLM McNemar for every run already on disk,
under both definitions of "correct", straight from results/*.jsonl.

Read-only over the stored rows: no LLM calls, no re-scan, and the existing
*_summary.json files are left exactly as they were generated. The rescore
lands in its own file so the original numbers stay auditable.

The runs on disk were scored with only the first definition, which cannot
favour the LLM by construction -- see metrics.agreement_vectors.

Usage:
    uv run python scripts/rescore_mcnemar.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from credhunter_x.evaluation.labeler import MatchOutcome
from credhunter_x.evaluation.metrics import agreement_vectors, mcnemar_test

RESULTS_DIR = Path("results")
OUT_PATH = RESULTS_DIR / "rq1_mcnemar_rescore.json"


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _rescore(rows: list[dict]) -> dict:
    outcomes = [MatchOutcome(r["ground_truth_outcome"]) for r in rows]
    flagged = [r["label"] == "true_secret" for r in rows]

    # as run_evaluation scored it: correct = flagged AND really a secret
    detector_old = [o is MatchOutcome.TRUE_POSITIVE for o in outcomes]
    llm_old = [
        f and o is MatchOutcome.TRUE_POSITIVE for f, o in zip(flagged, outcomes, strict=True)
    ]
    old_stat, old_p = mcnemar_test(detector_old, llm_old)

    detector_new, llm_new = agreement_vectors(outcomes, flagged)
    new_stat, new_p = mcnemar_test(detector_new, llm_new)

    def _cells(a: list[bool], b: list[bool]) -> tuple[int, int]:
        return (
            sum(1 for x, y in zip(a, b, strict=True) if x and not y),
            sum(1 for x, y in zip(a, b, strict=True) if not x and y),
        )

    old_det_only, old_llm_only = _cells(detector_old, llm_old)
    new_det_only, new_llm_only = _cells(detector_new, llm_new)

    return {
        "n_rows": len(rows),
        "n_scored": len(llm_new),
        "flagged_and_real": {
            "detector_only_correct": old_det_only,
            "llm_only_correct": old_llm_only,
            "statistic": old_stat,
            "p_value": old_p,
        },
        "agrees_with_truth": {
            "detector_only_correct": new_det_only,
            "llm_only_correct": new_llm_only,
            "statistic": new_stat,
            "p_value": new_p,
        },
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    paths = sorted(
        p
        for p in RESULTS_DIR.glob("*.jsonl")
        if (p.stem.startswith(("single_", "agentic_")) and "_summary" not in p.stem)
    )
    if not paths:
        print("no arm results found in results/ -- run run_evaluation.py first")
        return

    header = (
        f"{'run':<34}{'n':>5}{'scored':>8}"
        f"{'old chi2':>10}{'old p':>9}{'old fav':>9}"
        f"{'new chi2':>10}{'new p':>9}{'new fav':>9}"
    )
    print(header)
    print("-" * len(header))

    out: dict[str, dict] = {}
    for path in paths:
        scores = _rescore(_load(path))
        out[path.stem] = scores
        old, new = scores["flagged_and_real"], scores["agrees_with_truth"]
        old_fav = "detector" if old["detector_only_correct"] > old["llm_only_correct"] else "LLM"
        new_fav = "detector" if new["detector_only_correct"] > new["llm_only_correct"] else "LLM"
        name = path.stem.replace("_all_combined_gpt-5.6-luna", "")
        print(
            f"{name:<34}{scores['n_rows']:>5}{scores['n_scored']:>8}"
            f"{old['statistic']:>10.3f}{old['p_value']:>9.4f}{old_fav:>9}"
            f"{new['statistic']:>10.3f}{new['p_value']:>9.4f}{new_fav:>9}"
        )

    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print("old = correct is 'flagged AND really a secret' (cannot favour the LLM)")
    print("new = correct is 'agrees with ground truth', LOST excluded")
    print(f"written to {OUT_PATH} -- no *_summary.json was modified")


if __name__ == "__main__":
    main()
