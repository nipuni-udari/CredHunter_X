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
    """Scan `source` with trufflehog3 and return the raw list of issue dicts
    from its JSON report (empty list if nothing found). --zero makes it
    always exit 0 on a normal run, so any nonzero exit here is a real error.

    include_extensions, if given (e.g. (".py",)), restricts the scan to
    files with those suffixes by scanning a filtered copy of `source`
    instead of `source` itself. Not just a convenience: a single
    pathological file elsewhere in a large tree can crash trufflehog3's own
    entropy scan with a MemoryError (found against CredData's own corpus --
    one file with a single ~700,000-character line). Filtering out
    non-matching files before trufflehog3 ever reads them avoids that whole
    class of problem, not just the one file that happened to trigger it."""
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
