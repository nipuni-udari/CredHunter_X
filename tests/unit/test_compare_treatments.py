from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compare_treatments.py"


def _load_module():
    """scripts/ isn't a package, so load the file directly."""
    spec = importlib.util.spec_from_file_location("compare_treatments", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_treatments"] = module
    spec.loader.exec_module(module)
    return module


compare_treatments = _load_module()


def _write_summary(directory: Path, stem: str, **fields) -> None:
    payload = {"metrics": {"precision": 0.5, "recall": 0.5, "f1": 0.5}, **fields}
    (directory / f"{stem}_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_discovers_the_only_run_on_disk(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(compare_treatments, "RESULTS_DIR", tmp_path)
    _write_summary(
        tmp_path,
        "single_raw_all_combined_gpt-5.6-luna",
        arm="single",
        split="all",
        source="combined",
        model="openai/gpt-5.6-luna",
    )

    assert compare_treatments._discover_run("single", "all") == ("combined", "openai/gpt-5.6-luna")


def test_ignores_summaries_written_before_source_and_model_were_recorded(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Older summaries carry a model but no source (and vice versa). Reading
    one of those keys unconditionally crashed discovery with a KeyError on a
    real results/ directory -- they must simply be skipped."""
    monkeypatch.setattr(compare_treatments, "RESULTS_DIR", tmp_path)
    _write_summary(
        tmp_path,
        "single_masked_all_2026-08-27_gpt-5.6-luna",
        arm="single",
        split="all",
        model="openai/gpt-5.6-luna",  # no "source"
    )
    _write_summary(
        tmp_path,
        "single_raw_all_combined_gpt-5.6-luna",
        arm="single",
        split="all",
        source="combined",
        model="openai/gpt-5.6-luna",
    )

    assert compare_treatments._discover_run("single", "all") == ("combined", "openai/gpt-5.6-luna")


def test_refuses_to_guess_when_several_runs_exist(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Picking one silently would report a different run's numbers."""
    monkeypatch.setattr(compare_treatments, "RESULTS_DIR", tmp_path)
    for source, model in (("combined", "openai/gpt-5.6-luna"), ("gitleaks", "gemini/flash")):
        _write_summary(
            tmp_path,
            f"single_raw_all_{source}_x",
            arm="single",
            split="all",
            source=source,
            model=model,
        )

    with pytest.raises(SystemExit) as exc:
        compare_treatments._discover_run("single", "all")

    assert "--source combined" in str(exc.value)
    assert "--source gitleaks" in str(exc.value)


def test_exits_clearly_when_nothing_matches(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(compare_treatments, "RESULTS_DIR", tmp_path)

    with pytest.raises(SystemExit) as exc:
        compare_treatments._discover_run("agentic", "dev")

    assert "run_evaluation.py" in str(exc.value)


def test_other_arms_and_splits_do_not_leak_into_discovery(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """The glob is loose enough to match neighbouring runs, so the arm/split
    recorded inside each summary is what actually decides."""
    monkeypatch.setattr(compare_treatments, "RESULTS_DIR", tmp_path)
    _write_summary(
        tmp_path,
        "single_raw_all_combined_gpt-5.6-luna",
        arm="single",
        split="all",
        source="combined",
        model="openai/gpt-5.6-luna",
    )
    _write_summary(
        tmp_path,
        "single_raw_dev_combined_other-model",
        arm="single",
        split="dev",
        source="combined",
        model="openai/other-model",
    )

    assert compare_treatments._discover_run("single", "all") == ("combined", "openai/gpt-5.6-luna")


def test_bytes_of_real_secret_sent_is_zero_for_every_redacted_treatment(tmp_path):
    jsonl = tmp_path / "x.jsonl"
    jsonl.write_text(json.dumps({"real_secret_length": 40}) + "\n", encoding="utf-8")

    for treatment in ("masked", "pseudonymised", "metadata_only"):
        assert compare_treatments._bytes_of_real_secret_sent(jsonl, treatment) == 0
    assert compare_treatments._bytes_of_real_secret_sent(jsonl, "raw") == 40


def test_bytes_of_real_secret_sent_reports_a_file_predating_the_field(tmp_path):
    jsonl = tmp_path / "old.jsonl"
    jsonl.write_text(json.dumps({"candidate_id": "c1"}) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        compare_treatments._bytes_of_real_secret_sent(jsonl, "raw")

    assert "real_secret_length" in str(exc.value)
