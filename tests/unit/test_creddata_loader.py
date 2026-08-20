from __future__ import annotations

from pathlib import Path

import pytest

from credhunter_x.dataset.creddata import load_creddata_labels

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_CSV = FIXTURES_DIR / "creddata_labels_sample.csv"


def test_loads_all_rows():
    rows = load_creddata_labels(SAMPLE_CSV)
    assert len(rows) == 8


def test_ground_truth_is_true_only_for_t_label():
    rows = load_creddata_labels(SAMPLE_CSV)
    true_rows = [r for r in rows if r.ground_truth]
    false_rows = [r for r in rows if not r.ground_truth]

    assert {r.id for r in true_rows} == {"29502", "11503182", "11519093", "32209"}
    # both F and X collapse to ground_truth=False
    assert {r.id for r in false_rows} == {"830", "35963", "36233", "1979"}


def test_missing_value_start_end_normalise_to_negative_one():
    rows = load_creddata_labels(SAMPLE_CSV)
    f_row = next(r for r in rows if r.id == "830")

    assert f_row.value_start == -1
    assert f_row.value_end == -1


def test_present_value_start_end_parse_as_int():
    rows = load_creddata_labels(SAMPLE_CSV)
    t_row = next(r for r in rows if r.id == "29502")

    assert t_row.value_start == 25
    assert t_row.value_end == 41


def test_fields_map_correctly_for_a_known_row():
    rows = load_creddata_labels(SAMPLE_CSV)
    row = next(r for r in rows if r.id == "11503182")

    assert row.file_id == "188e3140"
    assert row.repo_id == "057480bf"
    assert row.file_path == "data/057480bf/test/tool/188e3140.py"
    assert row.line_start == 9
    assert row.line_end == 9
    assert row.category == "Salt"


def test_raises_a_clear_error_when_file_is_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_creddata_labels(tmp_path / "does_not_exist.csv")
