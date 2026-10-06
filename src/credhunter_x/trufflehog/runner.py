from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class TruffleHogError(RuntimeError):
    pass


def run_trufflehog(
    source: Path,
    trufflehog_binary: str = "trufflehog3",
    include_extensions: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Scans source with trufflehog3 (Radostev, 2024) and returns the issue dicts
    from its JSON report. --zero makes it exit 0 normally, so any other exit is an error.

    include_extensions (e.g. (".py",)) scans a filtered copy of source holding
    only those files. This also stops very large non-Python files from
    exhausting trufflehog3's memory."""
    source = source.resolve()

    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "report.json"
        scan_target = source
        if include_extensions is not None:
            filtered_root = Path(tmp) / "filtered"
            for path in source.rglob("*"):
                if path.is_file() and path.suffix in include_extensions:
                    dest = filtered_root / path.relative_to(source)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
            scan_target = filtered_root

        result = subprocess.run(
            [
                trufflehog_binary,
                "--no-history",
                "--zero",
                "--format",
                "JSON",
                "--output",
                str(report_path),
                str(scan_target),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise TruffleHogError(
                f"trufflehog3 exited with code {result.returncode}: {result.stderr.strip()}"
            )
        if not report_path.exists():
            return []
        findings: list[dict[str, Any]] = json.loads(report_path.read_text())
        return findings
