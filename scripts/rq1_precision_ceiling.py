"""How much of the achievable precision gain did the classifier actually capture?

A gain from 0.577 to 0.624 reads as small against a ceiling of 1.000. But
1.000 was never available: section 6.2.3 shows that the labels on
generic.password-in-url track nothing visible in the code, so no
context-based filter can resolve them. The right denominator is the gain
that was reachable, not the gain that is arithmetically possible.

This computes that denominator under assumptions stated in the output rather
than buried, because the number is only as defensible as they are. The
irreducible family is a parameter, not a constant, so the sensitivity is
visible.

Read-only, no API calls. Reproduces the published per-run precision before
reporting anything derived from it.

Usage:
    uv run python scripts/rq1_precision_ceiling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

RESULTS = Path("results")
SUFFIX = "_all_combined_gpt-5.6-luna"
LOST = "lost"

# The family section 6.2.3 argues is unresolvable from visible context. Held
# as a parameter so the claim can be checked against a different assumption
# rather than taken on trust.
IRREDUCIBLE_RULES = ("generic.password-in-url",)

ARMS = ("single", "agentic")


def _load(arm: str, treatment: str = "raw") -> list[dict[str, Any]]:
    path = RESULTS / f"{arm}_{treatment}{SUFFIX}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if r["ground_truth_outcome"] != LOST]


def _precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if tp + fp else 0.0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    scored = _load("single")
    real = [r for r in scored if r["ground_truth_outcome"] == "true_positive"]
    not_real = [r for r in scored if r["ground_truth_outcome"] == "false_positive"]

    # The detector flags everything it generated, so its confusion is the
    # composition of the scored set itself.
    det_tp, det_fp = len(real), len(not_real)
    det_p = _precision(det_tp, det_fp)

    irreducible_fp = [r for r in not_real if r["rule_id"] in IRREDUCIBLE_RULES]
    irreducible_tp = [r for r in real if r["rule_id"] in IRREDUCIBLE_RULES]

    print(f"scored candidates          : {len(scored)}")
    print(f"detector TP / FP           : {det_tp} / {det_fp}")
    print(f"detector precision         : {det_p:.4f}")
    print(f"irreducible family         : {', '.join(IRREDUCIBLE_RULES)}")
    print(f"  its false positives      : {len(irreducible_fp)}")
    print(f"  its true positives       : {len(irreducible_tp)}")

    # Ceiling A -- a perfect context-based filter: suppresses every false
    # positive whose label context can resolve, keeps every true positive, and
    # is left with the irreducible family, on which it can do no better than
    # the detector.
    ceil_a = _precision(det_tp, len(irreducible_fp))

    # Ceiling B -- the same filter, but permitted to discard the irreducible
    # family wholesale. Higher precision, but it throws away that family's real
    # credentials, so it is a different product rather than a better classifier.
    ceil_b_tp = det_tp - len(irreducible_tp)
    ceil_b = _precision(ceil_b_tp, 0)

    print(f"\nCeiling A (keeps every true positive)  : {ceil_a:.4f}  <- the reported ceiling")
    print(f"Ceiling B (discards the whole family)  : {ceil_b:.4f}, but loses "
          f"{len(irreducible_tp)} of {det_tp} recoverable true positives.")
    print("  Ceiling B is degenerate: under the 'perfect elsewhere' assumption it removes")
    print("  every remaining false positive, so it says more about the assumption than")
    print("  about any achievable system. It is reported for completeness, not used.")

    print(f"\n{'arm':<10}{'TP':>5}{'FP':>5}{'FN':>5}{'precision':>11}"
          f"{'gain':>9}{'headroom':>10}{'captured':>10}")
    print("-" * 65)

    out: dict[str, Any] = {
        "n_scored": len(scored),
        "detector_tp": det_tp,
        "detector_fp": det_fp,
        "detector_precision": det_p,
        "irreducible_rules": list(IRREDUCIBLE_RULES),
        "irreducible_false_positives": len(irreducible_fp),
        "irreducible_true_positives": len(irreducible_tp),
        "ceiling_keeping_all_true_positives": ceil_a,
        "ceiling_discarding_family": ceil_b,
        "arms": {},
    }

    for arm in ARMS:
        rows = _load(arm)
        tp = sum(
            1
            for r in rows
            if r["label"] == "true_secret" and r["ground_truth_outcome"] == "true_positive"
        )
        fp = sum(
            1
            for r in rows
            if r["label"] == "true_secret" and r["ground_truth_outcome"] == "false_positive"
        )
        fn = sum(
            1
            for r in rows
            if r["label"] != "true_secret" and r["ground_truth_outcome"] == "true_positive"
        )
        p = _precision(tp, fp)
        # The detector baseline for this arm's own scored set -- agentic
        # metadata_only aside, these are the same 182 rows, but recomputing
        # keeps the comparison like for like.
        arm_det = _precision(
            sum(1 for r in rows if r["ground_truth_outcome"] == "true_positive"),
            sum(1 for r in rows if r["ground_truth_outcome"] == "false_positive"),
        )
        gain = p - arm_det
        headroom = ceil_a - arm_det
        captured = gain / headroom if headroom else 0.0
        print(
            f"{arm:<10}{tp:>5}{fp:>5}{fn:>5}{p:>11.4f}"
            f"{gain:>+9.4f}{headroom:>10.4f}{captured:>9.1%}"
        )
        out["arms"][arm] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": p,
            "detector_precision": arm_det,
            "gain": gain,
            "headroom_to_ceiling": headroom,
            "fraction_of_headroom_captured": captured,
        }

    print(
        "\nheadroom = Ceiling A - detector precision; captured = gain / headroom.\n"
        "Ceiling A assumes only that the irreducible family cannot be resolved from\n"
        "context. Every other false positive is assumed perfectly suppressible, which\n"
        "is generous to the ceiling and therefore conservative about the share captured."
    )

    path = RESULTS / "rq1_precision_ceiling.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
