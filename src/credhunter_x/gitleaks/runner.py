from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# gitleaks' own convention: 0 = scan ran, no leaks found; 1 = scan ran,
# leaks found. Both are normal outcomes for us, not errors — only some
# other exit code means the gitleaks process itself actually failed.
_OK_EXIT_CODES = {0, 1}


class GitleaksError(RuntimeError):
    pass


def run_gitleaks(source: Path, gitleaks_binary: str = "gitleaks") -> list[dict[str, Any]]:
    """Scan `source` with gitleaks and return the raw list of finding dicts
    from its JSON report (empty list if nothing found)."""
    # Resolve to absolute first: gitleaks reports the "File" field relative
    # to whatever --source was given as, so a relative --source would make
    # "File" ambiguous relative-to-what for the parser. Always resolving
    # here means "File" in the report is always absolute.
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
