from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# gitleaks exits 0 for no leaks and 1 for leaks; anything else is an error.
_OK_EXIT_CODES = {0, 1}


class GitleaksError(RuntimeError):
    pass


def run_gitleaks(source: Path, gitleaks_binary: str = "gitleaks") -> list[dict[str, Any]]:
    """Scans source with gitleaks (Rice, 2026) and returns the finding dicts from
    its JSON report (an empty list if nothing was found)."""
    # Resolve first so gitleaks always reports absolute paths.
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
