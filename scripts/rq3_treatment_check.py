"""Does redaction change remediation QUALITY, not just wording? (RQ3)

E3 has no independent variable, so RQ3 scores one condition -- but which one
needs an argument, not a default. `raw` is the best case (the model sees
everything); `masked` is the configuration RQ4 says you should actually
deploy. This settles it empirically instead.

Paired by candidate: the same file:line scored under both treatments, so
differences cannot come from scoring different rows. Unpaired samples would
confound treatment with which candidates each treatment happened to flag.

Read the result against the checker's own 0.047 verdict flip rate
(scripts/check_checker_stability.py) -- a difference of one or two rows out
of thirty is inside that noise and means nothing.

Usage:
    uv run python scripts/rq3_treatment_check.py --arm single --n 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from credhunter_x.config.settings import Settings
from credhunter_x.evaluation.metrics import wilson_score_interval
from credhunter_x.evaluation.remediation_reference import load_remediation_reference
from credhunter_x.evaluation.remediation_scoring import (
    ElementCheckParsingError,
    score_remediation,
)
from credhunter_x.evaluation.result_files import result_stem
from credhunter_x.guard.errors import LeakError
from credhunter_x.guard.leak_guard import LeakGuard
from credhunter_x.llm.guarded_client import GuardedLLMClient
from credhunter_x.llm.litellm_client import LiteLLMClient
from credhunter_x.masking.secret_registry import SecretRegistry
from credhunter_x.models.candidate import Candidate

RESULTS_DIR = Path("results")
CANDIDATES_CACHE = Path("data/processed/all_combined.candidates.json")


def _rows(arm: str, treatment: str, model: str) -> dict[str, dict[str, Any]]:
    stem = result_stem(arm=arm, treatment=treatment, split="all", source="combined", model=model)
    path = RESULTS_DIR / f"{stem}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        r = json.loads(line)
        if r["label"] == "true_secret" and r["remediation"].strip():
            out[r["candidate_id"]] = r
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description="Paired raw-vs-masked remediation quality check")
    p.add_argument("--arm", choices=["single", "agentic"], default="single")
    p.add_argument("--baseline", default="raw")
    p.add_argument("--compare", default="masked")
    p.add_argument("--model", default=None)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/rq3_treatment_check.json")
    args = p.parse_args()

    settings = Settings()
    model = args.model or settings.llm_model
    base = _rows(args.arm, args.baseline, model)
    comp = _rows(args.arm, args.compare, model)
    reference = load_remediation_reference()

    payload = json.loads(CANDIDATES_CACHE.read_text(encoding="utf-8"))
    candidates = [Candidate(**r) for r in payload["candidates"]]
    by_id = {c.id: c for c in candidates}

    shared = sorted(
        cid
        for cid in set(base) & set(comp)
        if reference.get(base[cid]["rule_id"]) is not None and cid in by_id
    )
    print(
        f"{args.arm}: {len(base)} true_secret under {args.baseline}, "
        f"{len(comp)} under {args.compare}, {len(shared)} scoreable in both",
        flush=True,
    )

    rng = random.Random(args.seed)
    sample = rng.sample(shared, min(args.n, len(shared)))
    print(f"sampling {len(sample)} paired candidates = {len(sample) * 2} checker calls", flush=True)
    print(
        f"  rule mix: {dict(Counter(base[c]['rule_id'] for c in sample).most_common())}", flush=True
    )

    registry = SecretRegistry()
    registry.register_candidates(candidates)
    guard = LeakGuard(registry)
    client = GuardedLLMClient(
        LiteLLMClient(model=settings.llm_model, api_key=settings.llm_api_key), guard
    )

    verdicts: dict[str, dict[str, Any]] = {}
    errors = 0
    blocked: list[str] = []
    for treatment, rows in ((args.baseline, base), (args.compare, comp)):
        print(f"  scoring {treatment}...", flush=True)
        for cid in sample:
            row = rows[cid]
            entry = reference[row["rule_id"]]
            try:
                res = score_remediation(
                    client,
                    candidate_id=cid,
                    rule_id=row["rule_id"],
                    remediation=row["remediation"],
                    required_elements=entry.required_elements,
                    optional_elements=entry.optional_elements,
                    required_gates=entry.required_gates,
                )
            except ElementCheckParsingError as exc:
                print(f"    parse error {cid}: {exc}", flush=True)
                errors += 1
                continue
            except LeakError:
                # The guard already refused the call -- the payload never
                # left. This only decides what happens after: drop the row
                # rather than kill the run, same policy as the orchestrator's
                # skip_candidate_on_error. Recorded, never silently dropped.
                print(f"    guard blocked {cid} ({treatment}) -- excluded", flush=True)
                blocked.append(f"{treatment}:{cid}")
                continue
            verdicts.setdefault(cid, {"rule_id": row["rule_id"]})[treatment] = {
                "passed": res.pass_rate == 1.0,
                "element_present": list(res.element_present),
                "gate_answers": list(res.gate_answers),
            }

    paired = [c for c, v in verdicts.items() if args.baseline in v and args.compare in v]
    b_pass = sum(1 for c in paired if verdicts[c][args.baseline]["passed"])
    c_pass = sum(1 for c in paired if verdicts[c][args.compare]["passed"])
    only_b = [
        c
        for c in paired
        if verdicts[c][args.baseline]["passed"] and not verdicts[c][args.compare]["passed"]
    ]
    only_c = [
        c
        for c in paired
        if verdicts[c][args.compare]["passed"] and not verdicts[c][args.baseline]["passed"]
    ]

    b_lo, b_hi = wilson_score_interval(b_pass, len(paired))
    c_lo, c_hi = wilson_score_interval(c_pass, len(paired))

    print()
    print("=" * 66)
    print(f"paired candidates        : {len(paired)}   parse errors: {errors}")
    if blocked:
        print(f"guard-blocked (excluded) : {len(blocked)}  {blocked}")
    print(
        f"{args.baseline:<8} pass rate       : {b_pass}/{len(paired)} = "
        f"{b_pass / len(paired):.3f}   95% CI [{b_lo:.3f}, {b_hi:.3f}]"
    )
    print(
        f"{args.compare:<8} pass rate       : {c_pass}/{len(paired)} = "
        f"{c_pass / len(paired):.3f}   95% CI [{c_lo:.3f}, {c_hi:.3f}]"
    )
    print(
        f"discordant pairs         : {len(only_b)} pass only under {args.baseline}, "
        f"{len(only_c)} only under {args.compare}"
    )
    print("=" * 66)
    print("Compare the gap against the checker's own 0.047 verdict flip rate:")
    print(
        f"  {len(only_b) + len(only_c)} of {len(paired)} pairs disagree "
        f"({(len(only_b) + len(only_c)) / len(paired):.3f})"
    )

    out = {
        "arm": args.arm,
        "baseline": args.baseline,
        "compare": args.compare,
        "n_paired": len(paired),
        "parse_errors": errors,
        "guard_blocked": blocked,
        f"{args.baseline}_passed": b_pass,
        f"{args.compare}_passed": c_pass,
        f"{args.baseline}_ci": [b_lo, b_hi],
        f"{args.compare}_ci": [c_lo, c_hi],
        "pass_only_baseline": only_b,
        "pass_only_compare": only_c,
        "verdicts": verdicts,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
