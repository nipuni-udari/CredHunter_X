from __future__ import annotations

from pathlib import Path


def check_file_exists(source_root: Path, file_path: str) -> str:
    """Checks whether `file_path` (relative to source_root) exists in the
    scanned repo."""
    resolved_root = source_root.resolve()
    target = (source_root / file_path).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError:
        return f"'{file_path}' is outside the repository"
    if target.is_file():
        return f"'{file_path}' exists"
    if target.is_dir():
        return f"'{file_path}' exists, but is a directory, not a file"
    return f"'{file_path}' does not exist"
