from __future__ import annotations

from pathlib import Path

from credhunter_x.config.settings import Mode, load_scan_config
from credhunter_x.models.treatment import Treatment


def test_missing_file_defaults_to_masked_and_agentic(tmp_path: Path):
    config = load_scan_config(tmp_path / "does_not_exist.yml")

    assert config.treatment == Treatment.MASKED
    assert config.mode == Mode.AGENTIC


def test_file_present_without_treatment_key_defaults_to_masked(tmp_path: Path):
    path = tmp_path / ".secretscan.yml"
    path.write_text("mode: agentic\n")

    config = load_scan_config(path)

    assert config.treatment == Treatment.MASKED
    assert config.mode == Mode.AGENTIC


def test_explicit_single_mode_overrides_the_agentic_default(tmp_path: Path):
    path = tmp_path / ".secretscan.yml"
    path.write_text("mode: single\n")

    config = load_scan_config(path)

    assert config.mode == Mode.SINGLE


def test_treatment_key_in_yaml_is_ignored_and_stays_masked(tmp_path: Path):
    """The shipped CLI must never be able to send raw secrets, regardless
    of what a repo's own config file requests — see settings.py's
    load_scan_config docstring and the research scope doc's own
    .secretscan.yml example, which only ever exposes `mode`."""
    path = tmp_path / ".secretscan.yml"
    path.write_text("mode: single\ntreatment: raw\n")

    config = load_scan_config(path)

    assert config.treatment == Treatment.MASKED


def test_treatment_key_is_ignored_even_for_other_unimplemented_treatments(tmp_path: Path):
    path = tmp_path / ".secretscan.yml"
    path.write_text("treatment: pseudonymised\n")

    config = load_scan_config(path)

    assert config.treatment == Treatment.MASKED
