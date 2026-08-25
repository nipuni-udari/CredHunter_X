from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from credhunter_x.trufflehog.runner import run_trufflehog

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_REPO = FIXTURES_DIR / "sample_repo"

pytestmark = pytest.mark.skipif(
    shutil.which("trufflehog3") is None, reason="trufflehog3 binary not on PATH"
)


def test_run_trufflehog_finds_known_fixture_secrets():
    findings = run_trufflehog(SAMPLE_REPO)
    rule_ids = {f["rule"]["id"] for f in findings}
    assert rule_ids == {"high-entropy", "generic.password-in-url"}


def test_run_trufflehog_returns_empty_list_for_clean_directory(tmp_path):
    (tmp_path / "clean.py").write_text("x = 1\n")
    assert run_trufflehog(tmp_path) == []


def test_include_extensions_ignores_secrets_in_non_matching_files(tmp_path):
    (tmp_path / "secret.py").write_text('TOKEN = "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"\n')
    (tmp_path / "secret.js").write_text(
        'const TOKEN = "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";\n'
    )

    findings = run_trufflehog(tmp_path, include_extensions=(".py",))

    assert all(f["path"].endswith(".py") for f in findings)
    assert len(findings) == 1


def test_include_extensions_preserves_relative_path_structure(tmp_path):
    nested = tmp_path / "app" / "auth.py"
    nested.parent.mkdir(parents=True)
    nested.write_text('TOKEN = "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"\n')

    findings = run_trufflehog(tmp_path, include_extensions=(".py",))

    assert findings[0]["path"] == "app/auth.py"
