"""Does the element checker give the same verdict twice? (RQ3 step 3)

RQ1/RQ2/RQ4 are measured against CredData's fixed labels, so only the model
under test can wobble. RQ3 has no fixed answer sheet -- an LLM decides
pass/fail -- so the measuring instrument can move too, and that has to be
quantified before any pass rate is reported.

Method mirrors check_llm_consistency.py, pointed at the checker instead of
the classifier: hold the remediation text fixed, ask the same questions N
times, count how often the verdict changes.

Manual, costs real LLM quota -- not part of CI.

Usage:
    uv run python scripts/check_checker_stability.py --arm single --n 30 --repeats 5
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
from credhunter_x.evaluation.metrics import fleiss_kappa, wilson_score_interval
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


def _pairwise_flip_rate(verdicts: list[bool]) -> tuple[int, int]:
    """Disagreeing pairs out of all pairs, for one item."""
    pairs = disagree = 0
    for i in range(len(verdicts)):
        for j in range(i + 1, len(verdicts)):
            pairs += 1
            disagree += verdicts[i] != verdicts[j]
    return disagree, pairs


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description="Measure checker stability across repeat scorings")
    p.add_argument("--arm", choices=["single", "agentic"], default="single")
    p.add_argument("--treatment", default="raw")
    p.add_argument("--split", default="all")
    p.add_argument("--source", default="combined")
    p.add_argument("--model", default=None)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/rq3_checker_stability.json")
    args = p.parse_args()

    settings = Settings()
    stem = result_stem(
        arm=args.arm,
        treatment=args.treatment,
        split=args.split,
        source=args.source,
        model=args.model or settings.llm_model,
    )
    jsonl_path = RESULTS_DIR / f"{stem}.jsonl"
    if not jsonl_path.exists():
        raise SystemExit(f"missing {jsonl_path}")

    rows = [json.loads(x) for x in jsonl_path.read_text(encoding="utf-8").splitlines() if x]
    reference = load_remediation_reference()
    payload = json.loads(CANDIDATES_CACHE.read_text(encoding="utf-8"))
    candidates = [Candidate(**r) for r in payload["candidates"]]
    by_id = {c.id: c for c in candidates}

    scoreable = [
        r
        for r in rows
        if r["label"] == "true_secret"
        and reference.get(r["rule_id"]) is not None
        and r["candidate_id"] in by_id
        and r["remediation"].strip()
    ]
    print(f"{len(scoreable)} scoreable remediations in {stem}", flush=True)

    rng = random.Random(args.seed)
    sample = rng.sample(scoreable, min(args.n, len(scoreable)))
    print(
        f"sampling {len(sample)}, scoring each {args.repeats}x "
        f"= {len(sample) * args.repeats} checker calls",
        flush=True,
    )
    print(f"  rule mix: {dict(Counter(r['rule_id'] for r in sample).most_common())}", flush=True)

    registry = SecretRegistry()
    registry.register_candidates(candidates)
    guard = LeakGuard(registry)
    client = GuardedLLMClient(
        LiteLLMClient(model=settings.llm_model, api_key=settings.llm_api_key), guard
    )

    verdicts: dict[str, list[bool]] = {}
    elements: dict[str, list[list[bool]]] = {}
    gates: dict[str, list[list[bool]]] = {}
    errors = 0
    blocked: set[str] = set()

    for rep in range(1, args.repeats + 1):
        print(f"  repeat {rep}/{args.repeats}...", flush=True)
        for row in sample:
            entry = reference[row["rule_id"]]
            try:
                res = score_remediation(
                    client,
                    candidate_id=row["candidate_id"],
                    rule_id=row["rule_id"],
                    remediation=row["remediation"],
                    required_elements=entry.required_elements,
                    optional_elements=entry.optional_elements,
                    required_gates=entry.required_gates,
                )
            except ElementCheckParsingError as exc:
                print(f"    parse error on {row['candidate_id']}: {exc}", flush=True)
                errors += 1
                continue
            except LeakError:
                # Guard already refused the call; drop the row instead of
                # killing the run. Such a row loses a repeat and is excluded
                # from the kappa by the completeness filter below.
                print(f"    guard blocked {row['candidate_id']} -- excluded", flush=True)
                blocked.add(row["candidate_id"])
                continue
            cid = row["candidate_id"]
            verdicts.setdefault(cid, []).append(res.pass_rate == 1.0)
            elements.setdefault(cid, []).append(list(res.element_present))
            gates.setdefault(cid, []).append(list(res.gate_answers))

    complete = [c for c, v in verdicts.items() if len(v) == args.repeats]
    if len(complete) < 2:
        raise SystemExit("not enough complete rows to measure stability")

    # verdict level -- the number that actually gates a reported pass rate
    flipped = [c for c in complete if len(set(verdicts[c])) > 1]
    d_total = p_total = 0
    for c in complete:
        d, pr = _pairwise_flip_rate(verdicts[c])
        d_total += d
        p_total += pr
    verdict_flip = d_total / p_total if p_total else 0.0
    kappa = fleiss_kappa([["P" if x else "F" for x in verdicts[c]] for c in complete])

    # element level -- where the instability actually lives
    el_d = el_p = 0
    unstable_elements: Counter[str] = Counter()
    for c in complete:
        runs = elements[c]
        for idx in range(len(runs[0])):
            col = [r[idx] for r in runs]
            d, pr = _pairwise_flip_rate(col)
            el_d += d
            el_p += pr
            if len(set(col)) > 1:
                rid = next(r["rule_id"] for r in sample if r["candidate_id"] == c)
                unstable_elements[f"{rid}[{idx}]"] += 1
    element_flip = el_d / el_p if el_p else 0.0

    lo, hi = wilson_score_interval(len(flipped), len(complete))

    print()
    print("=" * 66)
    print(f"arm / treatment          : {args.arm} / {args.treatment}")
    print(f"remediations (complete)  : {len(complete)}  x {args.repeats} repeats")
    print(f"parse errors             : {errors}")
    if blocked:
        print(f"guard-blocked (excluded) : {len(blocked)}  {sorted(blocked)}")
    print(f"VERDICT flip rate        : {verdict_flip:.3f}  ({d_total}/{p_total} pairs)")
    print(f"  rows that ever flipped : {len(flipped)}/{len(complete)}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  verdict Fleiss kappa   : {kappa:.3f}")
    print(f"ELEMENT flip rate        : {element_flip:.3f}  ({el_d}/{el_p} pairs)")
    if unstable_elements:
        print(f"  least stable elements  : {dict(unstable_elements.most_common(6))}")
    print("=" * 66)
    if verdict_flip > 0.15:
        print("HIGH -- score each remediation 3x and take the majority in the full run.")
    elif verdict_flip > 0.05:
        print("MODERATE -- usable, but report the flip rate alongside any pass rate.")
    else:
        print("LOW -- single-pass scoring is defensible; still report the flip rate.")

    out: dict[str, Any] = {
        "source_run": stem,
        "arm": args.arm,
        "treatment": args.treatment,
        "n_remediations": len(complete),
        "repeats": args.repeats,
        "seed": args.seed,
        "parse_errors": errors,
        "guard_blocked": sorted(blocked),
        "verdict_flip_rate": verdict_flip,
        "verdict_rows_flipped": len(flipped),
        "verdict_flip_ci": [lo, hi],
        "verdict_fleiss_kappa": kappa,
        "element_flip_rate": element_flip,
        "unstable_elements": dict(unstable_elements),
        "verdicts_by_candidate": {c: verdicts[c] for c in complete},
        "elements_by_candidate": {c: elements[c] for c in complete},
        "gates_by_candidate": {c: gates[c] for c in complete},
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
