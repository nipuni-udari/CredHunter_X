from __future__ import annotations

from html import escape
from pathlib import Path

from credhunter_x.models.classification import Label, Severity
from credhunter_x.pipeline.orchestrator import ScanResult

_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}

_LABEL_SECTION_TITLES: dict[Label, str] = {
    Label.TRUE_SECRET: "Secrets found",
    Label.UNCERTAIN: "Needs manual review",
    Label.FALSE_POSITIVE: "Reviewed and dismissed as false positives",
}

_STYLE = """
body { font-family: system-ui, -apple-system, sans-serif; max-width: 960px;
       margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
h1 { font-size: 1.5rem; }
h2 { font-size: 1.15rem; margin-top: 2.5rem; border-bottom: 1px solid #ddd;
     padding-bottom: 0.3rem; }
.summary { display: flex; gap: 1.5rem; margin: 1rem 0 2rem; }
.summary div { background: #f4f4f4; border-radius: 6px; padding: 0.6rem 1rem; }
.count { font-size: 1.4rem; font-weight: 600; display: block; }
table { width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; }
th, td { text-align: left; padding: 0.5rem 0.6rem; border-bottom: 1px solid #eee;
         vertical-align: top; font-size: 0.92rem; }
th { color: #666; font-weight: 600; }
code { background: #f4f4f4; padding: 0.1rem 0.3rem; border-radius: 3px;
       font-size: 0.88em; }
.sev-critical, .sev-high { color: #b3261e; font-weight: 600; }
.sev-medium { color: #9a6700; font-weight: 600; }
.sev-low { color: #555; }
.dismissed { opacity: 0.65; }
.empty { color: #777; font-style: italic; }
"""


def build_html_report(results: list[ScanResult]) -> str:
    """Builds a self-contained developer-facing HTML report. All findings
    are shown, not just true_secret -- unlike the SARIF report (which
    deliberately excludes false_positive to avoid reproducing GitLeaks'
    own alert-fatigue problem in GitHub's code scanning UI), a human
    reading this page benefits from seeing what was reviewed and why it
    was dismissed, which is part of what makes an LLM explanation useful
    over a bare pattern match. Every value interpolated from candidate/
    classification data is HTML-escaped -- explanations and remediation
    text are model-generated free text and file paths come from a
    scanned repo, neither of which is trusted input."""
    counts = {label: 0 for label in Label}
    for result in results:
        counts[result.classification.label] += 1

    sections = "".join(
        _build_section(label, [r for r in results if r.classification.label == label])
        for label in (Label.TRUE_SECRET, Label.UNCERTAIN, Label.FALSE_POSITIVE)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CredHunter-X scan report</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>CredHunter-X scan report</h1>
<div class="summary">
  <div><span class="count">{counts[Label.TRUE_SECRET]}</span>secrets found</div>
  <div><span class="count">{counts[Label.UNCERTAIN]}</span>need review</div>
  <div><span class="count">{counts[Label.FALSE_POSITIVE]}</span>dismissed</div>
</div>
{sections}
</body>
</html>
"""


def _build_section(label: Label, rows: list[ScanResult]) -> str:
    title = escape(_LABEL_SECTION_TITLES[label])
    if not rows:
        return f'<h2>{title}</h2>\n<p class="empty">None.</p>\n'

    rows = sorted(rows, key=lambda r: _SEVERITY_ORDER.get(r.classification.severity, 99))
    body_class = ' class="dismissed"' if label == Label.FALSE_POSITIVE else ""
    table_rows = "\n".join(_build_row(r) for r in rows)
    return f"""<h2>{title}</h2>
<table{body_class}>
<thead><tr>
  <th>Location</th><th>Rule</th><th>Severity</th><th>Confidence</th>
  <th>Explanation</th><th>Remediation</th>
</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
"""


def _build_row(result: ScanResult) -> str:
    c, r = result.candidate, result.classification
    location = escape(f"{c.file_path}:{c.line_start}")
    rule_id = escape(c.rule_id)
    severity = escape(r.severity)
    sev_class = f"sev-{r.severity}"
    confidence = f"{r.confidence:.2f}"
    explanation = escape(r.explanation)
    remediation = escape(r.remediation)
    return (
        f"<tr><td><code>{location}</code></td><td>{rule_id}</td>"
        f'<td class="{sev_class}">{severity}</td><td>{confidence}</td>'
        f"<td>{explanation}</td><td>{remediation}</td></tr>"
    )


def write_html_report(results: list[ScanResult], path: Path) -> None:
    path.write_text(build_html_report(results), encoding="utf-8")
