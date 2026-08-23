from __future__ import annotations

from pathlib import Path

_MAX_MATCHES = 5
_CONTEXT_LINES = 2
_MAX_RESULT_CHARS = 2000


def search_file(source_root: Path, file_path: str, query: str) -> str:
    """Searches one file for a literal text match, with a little
    surrounding context. Returns raw content -- caller must mask it
    before sending to the LLM."""
    if not query:
        return "search_file error: query must not be empty"

    resolved_root = source_root.resolve()
    target = (source_root / file_path).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError:
        return f"search_file error: '{file_path}' is outside the repository"
    if not target.is_file():
        return f"search_file error: '{file_path}' does not exist"

    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"search_file error: could not read '{file_path}': {exc}"

    matches = [i for i, line in enumerate(lines) if query in line]
    if not matches:
        return f"No matches for {query!r} in {file_path}"

    blocks = []
    for i in matches[:_MAX_MATCHES]:
        start = max(0, i - _CONTEXT_LINES)
        end = min(len(lines), i + _CONTEXT_LINES + 1)
        blocks.append("\n".join(f"{n + 1}: {lines[n]}" for n in range(start, end)))

    result = f"{len(matches)} match(es) for {query!r} in {file_path}:\n\n" + "\n---\n".join(blocks)
    return result[:_MAX_RESULT_CHARS]
