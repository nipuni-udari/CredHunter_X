"""Does the agentic arm give BETTER remediation advice than the single arm?

RQ1/RQ2 compared the arms on labels and found a tie; that says nothing about
advice quality, which only RQ3 measures. No LLM calls -- pure aggregation
over the two scored files.

Per-arm pass rates use each arm's own full scoreable set. The arm COMPARISON
is paired on the candidates both arms flagged: rows only one arm flagged have
nothing to pair against, and including them would mix "is the advice better?"
with "did the arms flag different things?".

Usage:
    uv run python scripts/rq3_arm_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from credhunter_x.evaluation.metrics import mcnemar_test, wilson_score_interval

SINGLE = Path("results/rq3_scores_single_raw.json")
AGENTIC = Path("results/rq3_scores_agentic_raw.json")
OUT = Path("results/rq3_arm_comparison.json")
FLIP = 0.047  # checker's own verdict flip rate, results/rq3_checker_stability.json


def load(path: Path) -> dict[str, dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {r["candidate_id"]: r for r in d["rows"]}


def rate(k: int, n: int) -> str:
    lo, hi = wilson_score_interval(k, n)
    return f"{k}/{n} = {k / n:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    s, a = load(SINGLE), load(AGENTIC)
    shared = sorted(set(s) & set(a))

    print("=" * 68)
    print("PER-ARM pass rate -- each arm's own full scoreable set")
    print("=" * 68)
    sk, ak = sum(r["passed"] for r in s.values()), sum(r["passed"] for r in a.values())
    print(f"  single   {rate(sk, len(s))}")
    print(f"  agentic  {rate(ak, len(a))}")

    print()
    print("=" * 68)
    print(f"PAIRED comparison -- the {len(shared)} candidates BOTH arms flagged")
    print("=" * 68)
    sp = [s[c]["passed"] for c in shared]
    ap = [a[c]["passed"] for c in shared]
    print(f"  single   {rate(sum(sp), len(shared))}")
    print(f"  agentic  {rate(sum(ap), len(shared))}")

    only_s = [c for c in shared if s[c]["passed"] and not a[c]["passed"]]
    only_a = [c for c in shared if a[c]["passed"] and not s[c]["passed"]]
    stat, p = mcnemar_test(sp, ap)
    disc = len(only_s) + len(only_a)
    print(f"  passed under single only : {len(only_s)}")
    print(f"  passed under agentic only: {len(only_a)}")
    print(f"  McNemar chi2 = {stat:.3f}   p = {p:.4f}")
    print()
    print(f"  discordant {disc}/{len(shared)} = {disc / len(shared):.3f}  "
          f"vs the checker's own flip rate {FLIP:.3f}")
    verdict = (
        "DIFFERENCE (p < 0.05)" if p < 0.05 else "NO DIFFERENCE ESTABLISHED (p >= 0.05)"
    )
    print(f"  -> {verdict}")

    print()
    print("=" * 68)
    print("PER-RULE, on the shared set")
    print("=" * 68)
    rules: dict[str, list[str]] = {}
    for c in shared:
        rules.setdefault(s[c]["rule_id"], []).append(c)
    print(f"  {'rule':26s} {'single':>10s} {'agentic':>10s}   n")
    for rid, cs in sorted(rules.items(), key=lambda kv: -len(kv[1])):
        k_s = sum(s[c]["passed"] for c in cs)
        k_a = sum(a[c]["passed"] for c in cs)
        flag = "  <-- differs" if k_s != k_a else ""
        print(f"  {rid:26s} {k_s:5d}/{len(cs):<4d} {k_a:5d}/{len(cs):<4d} {len(cs):3d}{flag}")

    payload = {
        "shared_candidates": len(shared),
        "single_full": {"n": len(s), "passed": sk},
        "agentic_full": {"n": len(a), "passed": ak},
        "paired": {
            "single_passed": sum(sp),
            "agentic_passed": sum(ap),
            "pass_only_single": only_s,
            "pass_only_agentic": only_a,
            "mcnemar_chi2": stat,
            "mcnemar_p": p,
            "discordant_rate": disc / len(shared),
            "checker_flip_rate": FLIP,
        },
        "per_rule_shared": {
            rid: {
                "n": len(cs),
                "single_passed": sum(s[c]["passed"] for c in cs),
                "agentic_passed": sum(a[c]["passed"] for c in cs),
            }
            for rid, cs in rules.items()
        },
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
