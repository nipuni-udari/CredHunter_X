"""Cluster bootstrap confidence interval on the grader-validation kappa.

compare_hand_labels_to_checker.py reports kappa = 0.680 over 142 element
judgements. Those judgements are NOT independent: they come from 71
remediations at two elements each, and two elements of one remediation are
graded from the same text by the same grader. A kappa computed as though all
142 were independent is optimistic about the effective sample size.

This resamples the 71 REMEDIATIONS with replacement -- never the individual
judgements -- so the dependence inside a remediation is preserved in every
draw. The naive judgement-level interval is computed alongside it, not
because it is defensible, but because the gap between the two is the
evidence that clustering mattered.

Verdicts are read from the stored scored runs, exactly as
compare_hand_labels_to_checker.py does. Re-scoring would validate against a
different set of grader answers from the ones the dissertation reports --
the grader flips ~4.7% of verdicts between runs.

No API calls. Reproduces the published point estimate before reporting
anything new.

Usage:
    uv run python scripts/rq3_kappa_cluster_ci.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from credhunter_x.evaluation.metrics import cohens_kappa
from credhunter_x.evaluation.remediation_reference import load_remediation_reference

RESULTS = Path("results")
LABELS = RESULTS / "hand_labeling_nipuni.csv"
SEED = 20260903
DRAWS = 4000

# The published figures this script must reproduce before its own numbers mean
# anything. From compare_hand_labels_to_checker.py on the same inputs.
PUBLISHED_KAPPA = 0.680
PUBLISHED_AGREEMENT = 0.901

Judgements = list[tuple[bool, bool]]  # (human said present, grader said present)


@dataclass(frozen=True)
class BootstrapResult:
    unit: str
    n_units: int
    kappa_ci_low: float
    kappa_ci_high: float
    agreement_ci_low: float
    agreement_ci_high: float
    degenerate_draws: int

    @property
    def kappa_width(self) -> float:
        return self.kappa_ci_high - self.kappa_ci_low


def _parse_bool(value: str) -> bool | None:
    v = value.strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    return None


def load_clusters() -> dict[tuple[str, str], Judgements]:
    """Paired human/grader judgements grouped by remediation -- the cluster.

    Keyed by (arm, candidate_id): 211 candidates appear in both scored files,
    so candidate_id alone could hand one arm's labels the other arm's
    verdicts."""
    rows_by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    with LABELS.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_candidate[row["candidate_id"]].append(row)

    pairs = {(r["arm"], r["treatment"]) for rows in rows_by_candidate.values() for r in rows}
    verdicts: dict[tuple[str, str], dict[str, Any]] = {}
    for arm, treatment in sorted(pairs):
        path = RESULTS / f"rq3_scores_{arm}_{treatment}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        for r in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            verdicts[(arm, r["candidate_id"])] = r

    reference = load_remediation_reference()
    clusters: dict[tuple[str, str], Judgements] = {}
    for candidate_id, rows in rows_by_candidate.items():
        arm = rows[0]["arm"]
        entry = reference.get(rows[0]["rule_id"])
        if entry is None or entry.required_elements != [r["element_text"] for r in rows]:
            raise SystemExit(f"{candidate_id}: remediation_reference.yaml has changed since export")
        scored = verdicts.get((arm, candidate_id))
        if scored is None:
            raise SystemExit(f"{candidate_id}: no {arm} verdict")

        # Match on element text, not position -- a reordered reference would
        # otherwise pair a human answer with the wrong verdict.
        by_element = dict(
            zip(scored["required_elements"], scored["element_present"], strict=True)
        )
        judgements: Judgements = []
        for row in rows:
            human = _parse_bool(row["human_present"])
            grader = by_element.get(row["element_text"])
            if human is None or grader is None:
                raise SystemExit(f"{candidate_id}: unfilled label or unmatched element text")
            judgements.append((human, grader))
        clusters[(arm, candidate_id)] = judgements
    return clusters


def _kappa_and_agreement(judgements: Judgements) -> tuple[float, float]:
    human: list[object] = [h for h, _ in judgements]
    grader: list[object] = [g for _, g in judgements]
    agreement = sum(1 for h, g in judgements if h == g) / len(judgements)
    return cohens_kappa(human, grader), agreement


def _percentile_ci(values: list[float]) -> tuple[float, float]:
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values))]


def bootstrap(units: list[Judgements], *, label: str) -> BootstrapResult:
    """Percentile bootstrap resampling whole units with replacement.

    A unit is a remediation for the cluster bootstrap and a single judgement
    for the naive one -- the only difference between the two."""
    rng = random.Random(SEED)
    n = len(units)
    kappas: list[float] = []
    agreements: list[float] = []
    degenerate = 0
    for _ in range(DRAWS):
        drawn: Judgements = []
        for _ in range(n):
            drawn.extend(units[rng.randrange(n)])
        human = {h for h, _ in drawn}
        grader = {g for _, g in drawn}
        # cohens_kappa returns 1.0 when both raters use one category -- real
        # behaviour, but it inflates the upper tail, so it is counted.
        if len(human) == 1 and len(grader) == 1:
            degenerate += 1
        k, a = _kappa_and_agreement(drawn)
        kappas.append(k)
        agreements.append(a)

    k_lo, k_hi = _percentile_ci(kappas)
    a_lo, a_hi = _percentile_ci(agreements)
    return BootstrapResult(
        unit=label,
        n_units=n,
        kappa_ci_low=k_lo,
        kappa_ci_high=k_hi,
        agreement_ci_low=a_lo,
        agreement_ci_high=a_hi,
        degenerate_draws=degenerate,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    clusters = load_clusters()
    all_judgements = [j for group in clusters.values() for j in group]
    kappa, agreement = _kappa_and_agreement(all_judgements)

    print(f"remediations (clusters) : {len(clusters)}")
    print(f"element judgements      : {len(all_judgements)}")
    print(f"kappa                   : {kappa:.3f}")
    print(f"raw agreement           : {agreement:.3f}")

    # Nothing derived is trustworthy if the inputs no longer reproduce the
    # figure the dissertation already reports.
    ok = (
        abs(kappa - PUBLISHED_KAPPA) < 0.0005
        and abs(agreement - PUBLISHED_AGREEMENT) < 0.0005
    )
    print(
        f"reproduces published {PUBLISHED_KAPPA} / {PUBLISHED_AGREEMENT}: "
        f"{'YES' if ok else 'NO -- STOP, inputs have changed'}"
    )
    if not ok:
        raise SystemExit(1)

    cluster_units = list(clusters.values())
    naive_units: list[Judgements] = [[j] for j in all_judgements]

    print(f"\nPercentile bootstrap, {DRAWS} draws, seed {SEED}\n")
    header = f"{'resampling unit':<26}{'n':>5}{'95% CI on kappa':>24}{'95% CI on agreement':>26}"
    print(header)
    print("-" * len(header))

    out = []
    passes = ((cluster_units, "remediation (correct)"), (naive_units, "judgement (naive)"))
    for units, label in passes:
        r = bootstrap(units, label=label)
        out.append(r)
        k = f"[{r.kappa_ci_low:.3f}, {r.kappa_ci_high:.3f}]"
        a = f"[{r.agreement_ci_low:.3f}, {r.agreement_ci_high:.3f}]"
        print(f"{label:<26}{r.n_units:>5}{k:>24}{a:>26}")

    cluster, naive = out
    widening = cluster.kappa_width / naive.kappa_width
    print(
        f"\nClustering widens the kappa interval by {widening:.2f}x -- "
        "the cost of 142 judgements not being 142 independent observations."
    )
    lower = cluster.kappa_ci_low
    print(
        f"Cluster lower bound {lower:.3f} vs the conventional 0.60 threshold for "
        f"substantial agreement: {'CLEARS' if lower >= 0.60 else 'DOES NOT CLEAR'}"
    )
    for r in out:
        if r.degenerate_draws:
            print(
                f"note: {r.degenerate_draws} of {DRAWS} {r.unit} draws were degenerate "
                "(one category only); cohens_kappa returns 1.0 for those"
            )

    # Per-arm, reported because the sample splits 36/35 and a reader will ask.
    per_arm: dict[str, dict[str, object]] = {}
    for arm in ("single", "agentic"):
        units = [j for (a, _), j in clusters.items() if a == arm]
        flat = [j for group in units for j in group]
        arm_kappa, arm_agreement = _kappa_and_agreement(flat)
        r = bootstrap(units, label=f"remediation ({arm})")
        per_arm[arm] = {
            "n_clusters": len(units),
            "kappa": arm_kappa,
            "agreement": arm_agreement,
            "bootstrap": asdict(r),
        }
        print(
            f"\n{arm:<8} {len(units)} remediations, kappa {arm_kappa:.3f}, "
            f"95% CI [{r.kappa_ci_low:.3f}, {r.kappa_ci_high:.3f}]"
        )

    path = RESULTS / "rq3_kappa_cluster_ci.json"
    path.write_text(
        json.dumps(
            {
                "seed": SEED,
                "draws": DRAWS,
                "n_clusters": len(clusters),
                "n_judgements": len(all_judgements),
                "kappa": kappa,
                "raw_agreement": agreement,
                "bootstraps": [asdict(r) for r in out],
                "per_arm": per_arm,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
