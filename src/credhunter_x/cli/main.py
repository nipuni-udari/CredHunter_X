from __future__ import annotations

import sys
from pathlib import Path

import typer

from credhunter_x.config.settings import Settings, load_scan_config
from credhunter_x.models.classification import Label
from credhunter_x.pipeline.orchestrator import scan_repository
from credhunter_x.reporting.html_report import write_html_report
from credhunter_x.reporting.markdown_summary import write_markdown_summary
from credhunter_x.reporting.sarif import write_sarif_report

# Force UTF-8 stdout so model output doesn't crash on Windows' cp1252 console.
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
    markdown: Path | None = typer.Option(  # noqa: B008
        None,
        help="Append a markdown summary (e.g. $GITHUB_STEP_SUMMARY). "
        "Findings and reasoning only -- remediation stays in the HTML report.",
    ),
) -> None:
    """Scans PATH and prints one line per candidate. Exits non-zero if any
    candidate is true_secret, or if any candidate couldn't be classified
    at all."""
    settings = Settings()  # type: ignore[call-arg]  # pydantic-settings fills required fields from .env at runtime
    scan_config = load_scan_config()
    outcome = scan_repository(path, settings=settings, scan_config=scan_config)
    results = outcome.results

    if sarif is not None:
        write_sarif_report(results, sarif)
    if html is not None:
        write_html_report(results, html)
    if markdown is not None:
        write_markdown_summary(results, markdown, skipped_count=outcome.skipped_count)

    if not results and outcome.skipped_count == 0:
        typer.echo("No candidates found.")

    for result in results:
        c, r = result.candidate, result.classification
        typer.echo(
            f"{c.file_path}:{c.line_start}  [{c.rule_id}]  -> {r.label} "
            f"(confidence={r.confidence:.2f}, severity={r.severity})"
        )
        typer.echo(f"    {r.explanation}")

    if outcome.skipped_count:
        # Skipped != clean -- don't let an unresolved candidate look safe.
        typer.echo(
            f"\n{outcome.skipped_count} candidate(s) could not be classified "
            "(guard-blocked or unparseable model output) and are NOT reflected "
            "above -- this scan is incomplete, review manually."
        )

    if outcome.skipped_count or any(r.classification.label == Label.TRUE_SECRET for r in results):
        raise typer.Exit(code=1)
