from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CostRow:
    candidate_id: str
    turns: int
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass(frozen=True)
class CostSummary:
    n_candidates: int
    total_input_tokens: int
    total_output_tokens: int
    mean_input_tokens: float
    mean_output_tokens: float
    mean_turns: float
    mean_latency_ms: float
    total_latency_ms: float


def load_cost_rows(jsonl_path: Path) -> list[CostRow]:
    """Reads the per-candidate turns/input_tokens/output_tokens/latency_ms
    fields already written by scripts/run_evaluation.py's _write_results —
    the same results/*.jsonl files, read back here for cost analysis
    instead of re-running the LLM."""
    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        rows.append(
            CostRow(
                candidate_id=data["candidate_id"],
                turns=data["turns"],
                input_tokens=data["input_tokens"],
                output_tokens=data["output_tokens"],
                latency_ms=data["latency_ms"],
            )
        )
    return rows


def compute_cost_summary(rows: list[CostRow]) -> CostSummary:
    """Arm A is always 1 turn/call; Arm B's turns/tokens vary per candidate
    depending on how much tool use the model chose to do — this is what
    lets the two arms' cost be compared on the same terms."""
    if not rows:
        raise ValueError("cannot summarise cost over an empty result set")

    n = len(rows)
    total_input = sum(r.input_tokens for r in rows)
    total_output = sum(r.output_tokens for r in rows)
    total_latency = sum(r.latency_ms for r in rows)
    total_turns = sum(r.turns for r in rows)

    return CostSummary(
        n_candidates=n,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        mean_input_tokens=total_input / n,
        mean_output_tokens=total_output / n,
        mean_turns=total_turns / n,
        mean_latency_ms=total_latency / n,
        total_latency_ms=total_latency,
    )
