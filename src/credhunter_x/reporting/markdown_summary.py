from __future__ import annotations

from pathlib import Path

from credhunter_x.models.classification import Label, Severity
from credhunter_x.pipeline.orchestrator import ScanResult
from credhunter_x.reporting.rule_titles import rule_title

_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}

_LABEL_MARKERS: dict[Label, str] = {
    Label.TRUE_SECRET: "🔴",
    Label.UNCERTAIN: "🟡",
    Label.FALSE_POSITIVE: "⚪",
}


def _escape(text: str) -> str:
    """Pipes and newlines break a markdown table row."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _sorted(rows: list[ScanResult]) -> list[ScanResult]:
    return sorted(rows, key=lambda r: _SEVERITY_ORDER.get(r.classification.severity, 99))


def build_markdown_summary(results: list[ScanResult], *, skipped_count: int = 0) -> str:
    """GitHub-flavoured markdown for $GITHUB_STEP_SUMMARY: what was found and
    why, but not how to fix it -- the remediation stays in the HTML report,
    which the summary links to instead of duplicating."""
    by_label: dict[Label, list[ScanResult]] = {label: [] for label in Label}
    for result in results:
        by_label[result.classification.label].append(result)

    found = by_label[Label.TRUE_SECRET]
    review = by_label[Label.UNCERTAIN]
    dismissed = by_label[Label.FALSE_POSITIVE]

    lines = ["## CredHunter-X", ""]
    headline = "**No secrets found**" if not found else f"**{len(found)} secret(s) found**"
    lines.append(
        f"{headline} · {len(review)} need review · {len(dismissed)} dismissed "
        f"· {len(results)} candidate(s) scanned"
    )
    lines.append("")

    if skipped_count:
        lines += [
            f"> ⚠️ **{skipped_count} candidate(s) could not be classified** and are not "
            "listed below. This scan is incomplete -- review those manually.",
            "",
        ]

    reported = found + review
    if reported:
        lines += [
            "| | file | rule | severity | confidence |",
            "|---|---|---|---|---|",
        ]
        for result in _sorted(reported):
            c, r = result.candidate, result.classification
            rule = rule_title(c.rule_id)
            lines.append(
                f"| {_LABEL_MARKERS[r.label]} | `{_escape(c.file_path)}:{c.line_start}` "
                f"| {_escape(rule)} | {r.severity} | {r.confidence:.2f} |"
            )
        lines.append("")
        lines += _details("Why these were flagged", _sorted(reported))

    if dismissed:
        lines += _details(f"{len(dismissed)} dismissed as false positive(s)", _sorted(dismissed))

    return "\n".join(lines) + "\n"


def _details(title: str, rows: list[ScanResult]) -> list[str]:
    lines = ["<details>", f"<summary>{title}</summary>", ""]
    for result in rows:
        c, r = result.candidate, result.classification
        lines += [
            f"**`{_escape(c.file_path)}:{c.line_start}`** — {r.label} "
            f"(confidence {r.confidence:.2f}, severity {r.severity})",
            "",
            _escape(r.explanation),
            "",
        ]
    lines += ["</details>", ""]
    return lines


def write_markdown_summary(
    results: list[ScanResult], path: Path, *, skipped_count: int = 0
) -> None:
    """Appends rather than overwrites -- $GITHUB_STEP_SUMMARY is shared with
    every other step in the job."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(build_markdown_summary(results, skipped_count=skipped_count))
