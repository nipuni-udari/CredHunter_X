from __future__ import annotations

import sys
from pathlib import Path

import typer

from credhunter_x.config.settings import Settings, load_scan_config
from credhunter_x.models.classification import Label
from credhunter_x.pipeline.orchestrator import scan_repository
from credhunter_x.reporting.html_report import write_html_report
from credhunter_x.reporting.sarif import write_sarif_report

# Model-generated explanations can contain any Unicode character (smart
# punctuation, em-dashes, etc.) — force UTF-8 stdout so output never
# crashes on Windows' legacy console codepage (cp1252) regardless of what
# a given provider's model happens to generate.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = typer.Typer()


@app.command()
def scan(
    path: Path = typer.Argument(..., exists=True, file_okay=False),  # noqa: B008
    sarif: Path | None = typer.Option(  # noqa: B008
        None, help="Write a SARIF report to this path (for GitHub code scanning)."
    ),
    html: Path | None = typer.Option(  # noqa: B008
        None, help="Write a developer-friendly HTML report."
    ),
) -> None:
    """Scans PATH and prints one line per candidate. Exits non-zero if any
    candidate is classified true_secret — the signal a CI workflow gates
    on to fail the check/block the merge."""
    settings = Settings()
    scan_config = load_scan_config()
    results = scan_repository(path, settings=settings, scan_config=scan_config)

    if sarif is not None:
        write_sarif_report(results, sarif)
    if html is not None:
        write_html_report(results, html)

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

    if any(result.classification.label == Label.TRUE_SECRET for result in results):
        raise typer.Exit(code=1)
