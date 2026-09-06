"""Paired bootstrap CI on the precision gain over the detector -- H1's
decision statistic.

metrics.bootstrap_ci gives a CI on one run's precision. H1 asks something
different: is the DIFFERENCE against the detector reliably above zero? That
needs a paired resample -- both systems scored on the same drawn candidates,
so the shared sampling variation cancels.

The scope quotes [+0.013, +0.084] for Arm A; this regenerates that number
from the stored rows. Read-only, no API calls.

Usage:
    uv run python scripts/rq1_delta_precision_ci.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

RESULTS = Path("results")
SEED = 20260903
DRAWS = 4000  # matches what the status page reports
LOST = "lost"


def _load(stem: str) -> list[dict[str, Any]]:
    path = RESULTS / f"{stem}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _precision(pairs: list[tuple[bool, bool]]) -> float:
    """pairs are (is_really_a_secret, the system flagged it)."""
    tp = sum(1 for real, flagged in pairs if flagged and real)
    fp = sum(1 for real, flagged in pairs if flagged and not real)
    return tp / (tp + fp) if tp + fp else 0.0


def delta_ci(rows: list[dict[str, Any]]) -> tuple[float, float, float, float, float]:
    """Returns (detector P, llm P, delta, ci_low, ci_high) over the scored
    candidates. LOST rows have no ground-truth row at that file:line and are
    excluded from precision by CredData convention."""
    scored = [r for r in rows if r["ground_truth_outcome"] != LOST]
    real = [r["ground_truth_outcome"] == "true_positive" for r in scored]
    llm = [r["label"] == "true_secret" for r in scored]

    # The detector flags every candidate it generated -- that is the baseline.
    det_pairs = list(zip(real, [True] * len(real), strict=True))
    llm_pairs = list(zip(real, llm, strict=True))
    det_p, llm_p = _precision(det_pairs), _precision(llm_pairs)

    rng = random.Random(SEED)
    n = len(scored)
    deltas = []
    for _ in range(DRAWS):
        idx = [rng.randrange(n) for _ in range(n)]
        d = _precision([det_pairs[i] for i in idx])
        m = _precision([llm_pairs[i] for i in idx])
        deltas.append(m - d)
    deltas.sort()
    return det_p, llm_p, llm_p - det_p, deltas[int(0.025 * DRAWS)], deltas[int(0.975 * DRAWS)]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    stems = sorted(p.stem for p in RESULTS.glob("*_all_combined_gpt-5.6-luna.jsonl"))
    print(f"Paired bootstrap, {DRAWS} draws, seed {SEED}")
    print(f"{'run':<26}{'detector P':>12}{'LLM P':>9}{'delta':>9}{'95% CI on delta':>22}{'>0?':>6}")
    print("-" * 84)

    out = {}
    for stem in stems:
        rows = _load(stem)
        det, llm, delta, lo, hi = delta_ci(rows)
        name = stem.replace("_all_combined_gpt-5.6-luna", "")
        excludes_zero = lo > 0
        out[name] = {
            "detector_precision": det,
            "llm_precision": llm,
            "delta": delta,
            "ci_low": lo,
            "ci_high": hi,
            "excludes_zero": excludes_zero,
            "n_scored": len([r for r in rows if r["ground_truth_outcome"] != LOST]),
        }
        print(
            f"{name:<26}{det:>12.3f}{llm:>9.3f}{delta:>+9.3f}"
            f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}{'YES' if excludes_zero else 'no':>6}"
        )

    path = RESULTS / "rq1_delta_precision_ci.json"
    path.write_text(
        json.dumps({"seed": SEED, "draws": DRAWS, "runs": out}, indent=2), encoding="utf-8"
    )
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
