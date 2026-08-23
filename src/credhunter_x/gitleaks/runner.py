from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# gitleaks: 0 = no leaks, 1 = leaks found. Both are fine; anything else is a real failure.
_OK_EXIT_CODES = {0, 1}


class GitleaksError(RuntimeError):
    pass


def run_gitleaks(source: Path, gitleaks_binary: str = "gitleaks") -> list[dict[str, Any]]:
    """Scan `source` with gitleaks and return the raw list of finding dicts
    from its JSON report (empty list if nothing found)."""
    # Resolve first so gitleaks' "File" field is always absolute, never
    # ambiguous relative-to-what.
    source = source.resolve()

    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "report.json"
        result = subprocess.run(
            [
                gitleaks_binary,
                "detect",
                "--source",
                str(source),
                "--no-git",
                "--no-banner",
                "--report-format",
                "json",
                "--report-path",
                str(report_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode not in _OK_EXIT_CODES:
            raise GitleaksError(
                f"gitleaks exited with code {result.returncode}: {result.stderr.strip()}"
            )
        if not report_path.exists():
            return []
        findings: list[dict[str, Any]] = json.loads(report_path.read_text())
        return findings
