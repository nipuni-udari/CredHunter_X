"""Paired bootstrap CI for the F1 cost of redaction, the statistic for H4.

H4 asks whether removing the credential costs less than 5 F1 points. Both
treatments scored the same candidates, so the resampling is paired, by
candidate_id rather than position (agentic_metadata_only has 516 rows and
raw has 517). Read-only, no API calls.

Usage:
    uv run python scripts/rq4_treatment_equivalence_ci.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

RESULTS = Path("results")
SEED = 20260903  # same seed and number of draws as rq1_delta_precision_ci.py
DRAWS = 4000
LOST = "lost"
TOTAL_TRUE = 663  # every GroundTruth=='T' row in the Python-filtered corpus
MARGIN = 0.05  # H4 margin: F1 within 5 points of raw
SUFFIX = "_all_combined_gpt-5.6-luna"

ARMS = ("single", "agentic")
REDACTED = ("masked", "pseudonymised", "metadata_only")


def _load(arm: str, treatment: str) -> dict[str, dict[str, Any]]:
    """Rows keyed by candidate_id, so two runs can be joined rather than zipped."""
    path = RESULTS / f"{arm}_{treatment}{SUFFIX}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    keyed = {r["candidate_id"]: r for r in rows}
    if len(keyed) != len(rows):
        raise SystemExit(f"{path} has duplicate candidate_ids -- cannot pair")
    return keyed


def _metrics(pairs: list[tuple[bool, bool]]) -> tuple[float, float]:
    """(corpus F1, candidate F1) for one bootstrap sample. Corpus recall
    divides by all 663 true rows in the corpus; candidate recall divides by the
    true rows among the candidates."""
    tp = sum(1 for real, flagged in pairs if flagged and real)
    fp = sum(1 for real, flagged in pairs if flagged and not real)
    fn = sum(1 for real, flagged in pairs if real and not flagged)

    precision = tp / (tp + fp) if tp + fp else 0.0
    corpus_recall = tp / TOTAL_TRUE
    candidate_recall = tp / (tp + fn) if tp + fn else 0.0

    def f1(p: float, r: float) -> float:
        return 2 * p * r / (p + r) if p + r else 0.0

    return f1(precision, corpus_recall), f1(precision, candidate_recall)


def compare(arm: str, treatment: str) -> dict[str, Any]:
    raw_rows = _load(arm, "raw")
    red_rows = _load(arm, treatment)

    shared = sorted(set(raw_rows) & set(red_rows))
    dropped = sorted((set(raw_rows) | set(red_rows)) - set(shared))

    # LOST belongs to the candidate, not the treatment. If the two runs
    # disagree on it, the pairing is broken.
    for cid in shared:
        if (raw_rows[cid]["ground_truth_outcome"] == LOST) != (
            red_rows[cid]["ground_truth_outcome"] == LOST
        ):
            raise SystemExit(f"{arm}/{treatment}: {cid} is LOST in one run but not the other")

    scored = [cid for cid in shared if raw_rows[cid]["ground_truth_outcome"] != LOST]
    real = [raw_rows[cid]["ground_truth_outcome"] == "true_positive" for cid in scored]
    raw_flag = [raw_rows[cid]["label"] == "true_secret" for cid in scored]
    red_flag = [red_rows[cid]["label"] == "true_secret" for cid in scored]

    raw_pairs = list(zip(real, raw_flag, strict=True))
    red_pairs = list(zip(real, red_flag, strict=True))

    raw_corpus, raw_cand = _metrics(raw_pairs)
    red_corpus, red_cand = _metrics(red_pairs)

    discordant = sum(1 for a, b in zip(raw_flag, red_flag, strict=True) if a != b)

    rng = random.Random(SEED)
    n = len(scored)
    corpus_deltas: list[float] = []
    cand_deltas: list[float] = []
    for _ in range(DRAWS):
        idx = [rng.randrange(n) for _ in range(n)]
        rc, rcand = _metrics([raw_pairs[i] for i in idx])
        dc, dcand = _metrics([red_pairs[i] for i in idx])
        corpus_deltas.append(dc - rc)
        cand_deltas.append(dcand - rcand)
    corpus_deltas.sort()
    cand_deltas.sort()

    lo_i, hi_i = int(0.025 * DRAWS), int(0.975 * DRAWS)
    c_lo, c_hi = corpus_deltas[lo_i], corpus_deltas[hi_i]
    k_lo, k_hi = cand_deltas[lo_i], cand_deltas[hi_i]

    return {
        "arm": arm,
        "treatment": treatment,
        "n_paired": len(shared),
        "n_scored": n,
        "dropped_candidate_ids": dropped,
        "discordant_labels": discordant,
        "corpus_f1_raw": raw_corpus,
        "corpus_f1_treatment": red_corpus,
        "corpus_f1_delta": red_corpus - raw_corpus,
        "corpus_f1_ci_low": c_lo,
        "corpus_f1_ci_high": c_hi,
        "corpus_within_margin": abs(c_lo) < MARGIN and abs(c_hi) < MARGIN,
        "candidate_f1_raw": raw_cand,
        "candidate_f1_treatment": red_cand,
        "candidate_f1_delta": red_cand - raw_cand,
        "candidate_f1_ci_low": k_lo,
        "candidate_f1_ci_high": k_hi,
        "candidate_within_margin": abs(k_lo) < MARGIN and abs(k_hi) < MARGIN,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"Paired bootstrap on delta F1 vs raw, {DRAWS} draws, seed {SEED}")
    print(f"Equivalence margin: +/-{MARGIN:.2f} F1 (H4, fixed before data collection)\n")

    results = [compare(arm, t) for arm in ARMS for t in REDACTED]

    header = (
        f"{'comparison':<30}{'n':>5}{'disc':>6}"
        f"{'corpus dF1':>12}{'95% CI':>20}{'in margin':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        name = f"{r['arm']} x {r['treatment']}"
        ci = f"[{r['corpus_f1_ci_low']:+.4f}, {r['corpus_f1_ci_high']:+.4f}]"
        print(
            f"{name:<30}{r['n_scored']:>5}{r['discordant_labels']:>6}"
            f"{r['corpus_f1_delta']:>+12.4f}{ci:>20}"
            f"{'YES' if r['corpus_within_margin'] else 'NO':>11}"
        )

    print()
    print(f"{'comparison':<30}{'cand. dF1':>12}{'95% CI':>20}{'in margin':>11}")
    print("-" * 73)
    for r in results:
        name = f"{r['arm']} x {r['treatment']}"
        ci = f"[{r['candidate_f1_ci_low']:+.4f}, {r['candidate_f1_ci_high']:+.4f}]"
        print(
            f"{name:<30}{r['candidate_f1_delta']:>+12.4f}{ci:>20}"
            f"{'YES' if r['candidate_within_margin'] else 'NO':>11}"
        )

    for r in results:
        if r["dropped_candidate_ids"]:
            print(
                f"\nnote: {r['arm']} x {r['treatment']} paired on {r['n_paired']} candidates; "
                f"dropped {len(r['dropped_candidate_ids'])} present in only one run:"
            )
            for cid in r["dropped_candidate_ids"]:
                print(f"  {cid}")

    worst_low = min(r["corpus_f1_ci_low"] for r in results)
    print(
        f"\nWorst-case corpus F1 cost across all six comparisons: {worst_low:+.4f} "
        f"(margin {-MARGIN:+.2f})"
    )
    print("All six within margin" if all(r["corpus_within_margin"] for r in results) else
          "AT LEAST ONE COMPARISON FALLS OUTSIDE THE MARGIN")

    path = RESULTS / "rq4_treatment_equivalence_ci.json"
    path.write_text(
        json.dumps(
            {"seed": SEED, "draws": DRAWS, "margin": MARGIN, "comparisons": results}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
