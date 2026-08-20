from __future__ import annotations

import json
from pathlib import Path

import pytest

from credhunter_x.evaluation.cost import CostRow, compute_cost_summary, load_cost_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_load_cost_rows_reads_the_relevant_fields(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    _write_jsonl(
        path,
        [
            {
                "candidate_id": "a",
                "turns": 1,
                "input_tokens": 500,
                "output_tokens": 100,
                "latency_ms": 900.0,
                "label": "true_secret",  # extra fields are ignored
            },
            {
                "candidate_id": "b",
                "turns": 3,
                "input_tokens": 1200,
                "output_tokens": 300,
                "latency_ms": 2400.0,
            },
        ],
    )

    rows = load_cost_rows(path)

    assert rows == [
        CostRow("a", turns=1, input_tokens=500, output_tokens=100, latency_ms=900.0),
        CostRow("b", turns=3, input_tokens=1200, output_tokens=300, latency_ms=2400.0),
    ]


def test_load_cost_rows_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    row = {
        "candidate_id": "a",
        "turns": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "latency_ms": 1.0,
    }
    path.write_text(json.dumps(row) + "\n\n", encoding="utf-8")

    rows = load_cost_rows(path)

    assert len(rows) == 1


def test_compute_cost_summary_from_hand_computed_totals():
    rows = [
        CostRow("a", turns=1, input_tokens=500, output_tokens=100, latency_ms=1000.0),
        CostRow("b", turns=3, input_tokens=1500, output_tokens=300, latency_ms=3000.0),
    ]

    summary = compute_cost_summary(rows)

    assert summary.n_candidates == 2
    assert summary.total_input_tokens == 2000
    assert summary.total_output_tokens == 400
    assert summary.mean_input_tokens == 1000.0
    assert summary.mean_output_tokens == 200.0
    assert summary.mean_turns == 2.0
    assert summary.mean_latency_ms == 2000.0
    assert summary.total_latency_ms == 4000.0


def test_compute_cost_summary_raises_on_empty_input():
    with pytest.raises(ValueError):
        compute_cost_summary([])
