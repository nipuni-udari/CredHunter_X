"""Exploratory: is the agentic arm's narrower confidence gap more than noise?

Bootstraps the difference in (mean confidence when right - mean confidence
when wrong) between the two raw runs, over flagged, scored candidates only.
Not part of the pre-registered test plan -- report it as exploratory.

Usage:
    uv run python scripts/rq2_separation_ci.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

RESULTS = Path("results")
SEED, DRAWS = 20260903, 4000


def _flagged_scored(arm: str) -> list[tuple[float, bool]]:
    path = RESULTS / f"{arm}_raw_all_combined_gpt-5.6-luna.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [
        (r["confidence"], r["ground_truth_outcome"] == "true_positive")
        for r in rows
        if r["label"] == "true_secret"
        and r["ground_truth_outcome"] in ("true_positive", "false_positive")
    ]


def _separation(sample: list[tuple[float, bool]]) -> float:
    right = [c for c, ok in sample if ok]
    wrong = [c for c, ok in sample if not ok]
    return sum(right) / len(right) - sum(wrong) / len(wrong)


def main() -> None:
    single, agentic = _flagged_scored("single"), _flagged_scored("agentic")
    rng = random.Random(SEED)
    diffs = sorted(
        _separation(rng.choices(single, k=len(single)))
        - _separation(rng.choices(agentic, k=len(agentic)))
        for _ in range(DRAWS)
    )
    out = {
        "seed": SEED,
        "draws": DRAWS,
        "n_single": len(single),
        "n_agentic": len(agentic),
        "separation_single": _separation(single),
        "separation_agentic": _separation(agentic),
        "difference": _separation(single) - _separation(agentic),
        "ci_low": diffs[int(0.025 * DRAWS)],
        "ci_high": diffs[int(0.975 * DRAWS) - 1],
    }
    (RESULTS / "rq2_separation_ci.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
