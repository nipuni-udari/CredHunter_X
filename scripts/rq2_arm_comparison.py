"""E2: single-prompt vs multi-step, on one frozen candidate set.

Everything RQ2 asks for, from the stored rows -- paired McNemar, quality
metrics, token and latency cost, money per 1,000 candidates, and the
calibration data behind the confidence-vs-correctness figure. No LLM calls.

Prices default to GPT-5.6 Luna's September 2026 rates. They are flags, not
constants, because the rate dropped 80% five weeks before these runs and
will move again -- pass the rate that applied, and the JSON records it.

Usage:
    uv run python scripts/rq2_arm_comparison.py
    uv run python scripts/rq2_arm_comparison.py --treatment masked --fx 0.75
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scipy.stats import chi2

from credhunter_x.evaluation.cost import compute_cost_summary, load_cost_rows

RESULTS = Path("results")
LOST = "lost"
CONFIDENT = 0.9


def _rows(arm: str, treatment: str, model: str) -> list[dict[str, Any]]:
    path = RESULTS / f"{arm}_{treatment}_all_combined_{model}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _agrees(row: dict[str, Any]) -> bool:
    """Correct = the label matches ground truth. Not "flagged and real" --
    that definition makes a correctly suppressed false positive count as
    wrong for both arms; see metrics.agreement_vectors."""
    return bool((row["label"] == "true_secret") == (row["ground_truth_outcome"] == "true_positive"))


def _mcnemar(a_rows: dict[str, Any], b_rows: dict[str, Any]) -> dict[str, Any]:
    only_a = only_b = 0
    for cid, a in a_rows.items():
        b = b_rows.get(cid)
        if b is None or a["ground_truth_outcome"] == LOST:
            continue
        ax, bx = _agrees(a), _agrees(b)
        only_a += ax and not bx
        only_b += bx and not ax
    n = only_a + only_b
    stat = (abs(only_a - only_b) - 1) ** 2 / n if n else 0.0
    return {
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "statistic": stat,
        "p_value": float(1 - chi2.cdf(stat, 1)) if n else 1.0,
    }


def _calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Confidence split by whether the call was right. If the two
    distributions sit on top of each other, the score carries no signal."""
    scored = [r for r in rows if r["ground_truth_outcome"] != LOST]
    right = [r["confidence"] for r in scored if _agrees(r)]
    wrong = [r["confidence"] for r in scored if not _agrees(r)]

    # LOST rows have no ground-truth entry at that file:line, so a flag on one
    # cannot be called wrong -- CredData convention keeps them out of
    # precision, and they are kept out here too. Counting them as wrong (as an
    # earlier hand-analysis did) inflates the confident-and-wrong count.
    flagged = [r for r in rows if r["label"] == "true_secret"]
    flagged_scored = [r for r in flagged if r["ground_truth_outcome"] != LOST]
    high = [r for r in flagged_scored if r["confidence"] >= CONFIDENT]
    high_wrong = [r for r in high if r["ground_truth_outcome"] == "false_positive"]
    high_incl_lost = [r for r in flagged if r["confidence"] >= CONFIDENT]
    high_wrong_incl_lost = [
        r for r in high_incl_lost if r["ground_truth_outcome"] != "true_positive"
    ]

    edges = [0.70 + 0.02 * i for i in range(16)]

    def hist(vals: list[float]) -> list[int]:
        out = [0] * (len(edges) - 1)
        for v in vals:
            for i in range(len(edges) - 1):
                if edges[i] <= v < edges[i + 1] or (i == len(edges) - 2 and v >= edges[i + 1]):
                    out[i] += 1
                    break
        return out

    mean_right = sum(right) / len(right) if right else 0.0
    mean_wrong = sum(wrong) / len(wrong) if wrong else 0.0

    # Two defensible denominators, so neither is lost. "all scored" uses the
    # corrected notion of correct (agrees with truth, so a suppressed false
    # positive counts as right); "flagged only" asks the narrower question --
    # when it raised an alarm, was it confident about the good ones?
    f_right = [r["confidence"] for r in flagged if r["ground_truth_outcome"] == "true_positive"]
    f_wrong = [r["confidence"] for r in flagged if r["ground_truth_outcome"] == "false_positive"]
    fm_right = sum(f_right) / len(f_right) if f_right else 0.0
    fm_wrong = sum(f_wrong) / len(f_wrong) if f_wrong else 0.0

    return {
        "flagged_only": {
            "mean_confidence_when_right": fm_right,
            "mean_confidence_when_wrong": fm_wrong,
            "separation": fm_right - fm_wrong,
            "n_right": len(f_right),
            "n_wrong": len(f_wrong),
        },
        "n_scored": len(scored),
        "n_correct": len(right),
        "n_wrong": len(wrong),
        "mean_confidence_when_right": mean_right,
        "mean_confidence_when_wrong": mean_wrong,
        "separation": mean_right - mean_wrong,
        "flagged": len(flagged),
        "flagged_scored": len(flagged_scored),
        "flagged_confident": len(high),
        "confident_and_wrong": len(high_wrong),
        "confident_and_wrong_rate": len(high_wrong) / len(high) if high else 0.0,
        "confident_and_wrong_incl_lost": len(high_wrong_incl_lost),
        "confident_and_wrong_rate_incl_lost": (
            len(high_wrong_incl_lost) / len(high_incl_lost) if high_incl_lost else 0.0
        ),
        "bin_edges": edges,
        "hist_right": hist(right),
        "hist_wrong": hist(wrong),
    }


def _cost(arm: str, treatment: str, model: str, args: argparse.Namespace) -> dict[str, Any]:
    path = RESULTS / f"{arm}_{treatment}_all_combined_{model}.jsonl"
    summary = compute_cost_summary(load_cost_rows(path))
    usd = (
        summary.total_input_tokens * args.price_in / 1e6
        + summary.total_output_tokens * args.price_out / 1e6
    )
    per_1k = usd / summary.n_candidates * 1000
    return {
        "n_candidates": summary.n_candidates,
        "input_tokens": summary.total_input_tokens,
        "output_tokens": summary.total_output_tokens,
        "total_tokens": summary.total_input_tokens + summary.total_output_tokens,
        "mean_turns": summary.mean_turns,
        "mean_latency_ms": summary.mean_latency_ms,
        "usd_total": usd,
        "usd_per_1k_candidates": per_1k,
        "gbp_per_1k_candidates": per_1k * args.fx,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description="RQ2 / E2 -- single vs agentic")
    p.add_argument("--treatment", default="raw")
    p.add_argument("--model", default="gpt-5.6-luna")
    p.add_argument("--price-in", type=float, default=0.20, help="USD per 1M input tokens")
    p.add_argument("--price-out", type=float, default=1.20, help="USD per 1M output tokens")
    p.add_argument("--fx", type=float, default=0.741, help="GBP per USD")
    p.add_argument("--out", default="results/rq2_arm_comparison.json")
    args = p.parse_args()

    a = _rows("single", args.treatment, args.model)
    b = _rows("agentic", args.treatment, args.model)
    a_by = {r["candidate_id"]: r for r in a}
    b_by = {r["candidate_id"]: r for r in b}
    shared = [c for c in a_by if c in b_by]
    disagreements = sum(1 for c in shared if a_by[c]["label"] != b_by[c]["label"])

    payload = {
        "treatment": args.treatment,
        "model": args.model,
        "pricing": {
            "usd_per_1m_input": args.price_in,
            "usd_per_1m_output": args.price_out,
            "gbp_per_usd": args.fx,
        },
        "n_shared_candidates": len(shared),
        "label_disagreements": disagreements,
        "label_disagreement_rate": disagreements / len(shared) if shared else 0.0,
        "mcnemar": _mcnemar(a_by, b_by),
        "single": {
            "calibration": _calibration(a),
            "cost": _cost("single", args.treatment, args.model, args),
        },
        "agentic": {
            "calibration": _calibration(b),
            "cost": _cost("agentic", args.treatment, args.model, args),
        },
    }

    m, sc, ac = payload["mcnemar"], payload["single"], payload["agentic"]
    print(
        f"RQ2 / E2  --  single vs agentic, {args.treatment} treatment, "
        f"{len(shared)} shared candidates"
    )
    print()
    print(
        f"  label disagreements   : {disagreements}/{len(shared)} "
        f"({payload['label_disagreement_rate']:.1%})"
    )
    print(
        f"  paired McNemar        : only-single={m['only_a_correct']} "
        f"only-agentic={m['only_b_correct']}  chi2={m['statistic']:.3f}  p={m['p_value']:.3f}"
    )
    print()
    print(f"  {'':<24}{'single':>14}{'agentic':>14}{'ratio':>9}")
    for label, key, fmt in (
        ("total tokens", "total_tokens", ",.0f"),
        ("mean latency (ms)", "mean_latency_ms", ",.0f"),
        ("mean turns", "mean_turns", ".2f"),
        ("USD / 1k candidates", "usd_per_1k_candidates", ".3f"),
        ("GBP / 1k candidates", "gbp_per_1k_candidates", ".3f"),
    ):
        s_v, a_v = sc["cost"][key], ac["cost"][key]
        print(f"  {label:<24}{format(s_v, fmt):>14}{format(a_v, fmt):>14}{a_v / s_v:>8.2f}x")
    print()
    print(f"  {'':<24}{'single':>14}{'agentic':>14}")
    for label, key, fmt in (
        ("confident & wrong", "confident_and_wrong", "d"),
        ("  as a rate", "confident_and_wrong_rate", ".3f"),
        ("mean conf. when right", "mean_confidence_when_right", ".3f"),
        ("mean conf. when wrong", "mean_confidence_when_wrong", ".3f"),
        ("separation", "separation", ".3f"),
    ):
        print(
            f"  {label:<24}{format(sc['calibration'][key], fmt):>14}"
            f"{format(ac['calibration'][key], fmt):>14}"
        )
    print(f"  {'-- flagged rows only --':<24}")
    for label, key in (
        ("mean conf. when right", "mean_confidence_when_right"),
        ("mean conf. when wrong", "mean_confidence_when_wrong"),
        ("separation", "separation"),
    ):
        print(
            f"  {label:<24}{sc['calibration']['flagged_only'][key]:>14.3f}"
            f"{ac['calibration']['flagged_only'][key]:>14.3f}"
        )

    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
