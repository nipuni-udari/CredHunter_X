from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from credhunter_x.gitleaks.runner import run_gitleaks

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_REPO = FIXTURES_DIR / "sample_repo"

pytestmark = pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks binary not on PATH"
)


def test_run_gitleaks_finds_known_fixture_secrets():
    findings = run_gitleaks(SAMPLE_REPO)
    rule_ids = {f["RuleID"] for f in findings}
    assert rule_ids == {"github-pat", "slack-bot-token"}


def test_run_gitleaks_returns_empty_list_for_clean_directory(tmp_path):
    (tmp_path / "clean.py").write_text("x = 1\n")
    assert run_gitleaks(tmp_path) == []
