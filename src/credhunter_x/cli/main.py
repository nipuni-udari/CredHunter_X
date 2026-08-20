from __future__ import annotations

import sys
from pathlib import Path

import typer

from credhunter_x.config.settings import Settings, load_scan_config
from credhunter_x.pipeline.orchestrator import scan_repository

# Model-generated explanations can contain any Unicode character (smart
# punctuation, em-dashes, etc.) — force UTF-8 stdout so output never
# crashes on Windows' legacy console codepage (cp1252) regardless of what
# a given provider's model happens to generate.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = typer.Typer()


@app.command()
def scan(path: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:  # noqa: B008
    settings = Settings()
    scan_config = load_scan_config()
    results = scan_repository(path, settings=settings, scan_config=scan_config)

    if not results:
        typer.echo("No candidates found.")
        return

    for result in results:
        c, r = result.candidate, result.classification
        typer.echo(
            f"{c.file_path}:{c.line_start}  [{c.rule_id}]  -> {r.label} "
            f"(confidence={r.confidence:.2f}, severity={r.severity})"
        )
        typer.echo(f"    {r.explanation}")
