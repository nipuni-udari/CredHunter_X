from __future__ import annotations

import base64
from html import escape
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

_LABEL_SECTION_TITLES: dict[Label, str] = {
    Label.TRUE_SECRET: "Secrets found",
    Label.UNCERTAIN: "Needs manual review",
    Label.FALSE_POSITIVE: "Reviewed and dismissed as false positives",
}

_LABEL_ANCHORS: dict[Label, str] = {
    Label.TRUE_SECRET: "found",
    Label.UNCERTAIN: "review",
    Label.FALSE_POSITIVE: "dismissed",
}

_ICON_PATH = Path(__file__).parent / "assets" / "icon.png"


def _logo() -> str:
    """Inlined as a data URI so the report stays one file that works offline.
    A missing icon degrades to no logo rather than a broken image."""
    if not _ICON_PATH.is_file():
        return ""
    data = base64.b64encode(_ICON_PATH.read_bytes()).decode("ascii")
    return f'<img class="logo" src="data:image/png;base64,{data}" alt="">'


_STYLE = """
:root {
  --bg:#fbfbfd; --card:#fff; --ink:#16161d; --muted:#5f6470; --line:#e4e4ec;
  --brand:#6d28d9; --crit:#b3261e; --warn:#9a6700; --ok:#1a7f4b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#0e0e13; --card:#17171f; --ink:#ececf2; --muted:#9a9aa8; --line:#2a2a36;
    --brand:#a78bfa; --crit:#ff8a80; --warn:#e3b341; --ok:#56d68a;
  }
}
* { box-sizing:border-box; }
body { font-family:system-ui,-apple-system,"Segoe UI",sans-serif; background:var(--bg);
       color:var(--ink); max-width:1000px; margin:0 auto; padding:2.5rem 1.25rem 4rem;
       line-height:1.55; }
header { display:flex; align-items:center; gap:.9rem; margin-bottom:1.75rem; }
.logo { width:56px; height:auto; flex:none; }
h1 { font-size:1.4rem; margin:0; letter-spacing:-.01em; }
h1 small { display:block; font-size:.8rem; font-weight:400; color:var(--muted);
           letter-spacing:0; margin-top:.15rem; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
         gap:.85rem; margin-bottom:2.5rem; }
.tile { display:block; text-decoration:none; color:inherit; background:var(--card);
        border:1px solid var(--line); border-left:4px solid var(--muted);
        border-radius:10px; padding:.85rem 1rem; transition:transform .12s, box-shadow .12s; }
.tile:hover { transform:translateY(-2px); box-shadow:0 6px 18px rgba(0,0,0,.09);
              border-color:var(--brand); }
.tile .count { display:block; font-size:1.9rem; font-weight:650; line-height:1.1; }
.tile .what { font-size:.85rem; color:var(--muted); }
.tile.found { border-left-color:var(--crit); } .tile.found .count { color:var(--crit); }
.tile.review { border-left-color:var(--warn); } .tile.review .count { color:var(--warn); }
.tile.dismissed { border-left-color:var(--ok); } .tile.dismissed .count { color:var(--ok); }
h2 { font-size:1.05rem; margin:2.5rem 0 1rem; padding-bottom:.4rem;
     border-bottom:1px solid var(--line); scroll-margin-top:1rem; }
.finding { background:var(--card); border:1px solid var(--line); border-radius:10px;
           padding:1rem 1.15rem; margin-bottom:.85rem; }
.finding.is-dismissed { opacity:.72; }
.head { display:flex; flex-wrap:wrap; align-items:center; gap:.55rem; margin-bottom:.7rem; }
.loc { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:.86rem;
       background:rgba(125,125,150,.13); padding:.2rem .45rem; border-radius:5px; }
.rule { font-size:.86rem; color:var(--muted); }
.badge { font-size:.72rem; font-weight:650; text-transform:uppercase; letter-spacing:.04em;
         padding:.16rem .45rem; border-radius:20px; border:1px solid currentColor; }
.sev-critical,.sev-high { color:var(--crit); }
.sev-medium { color:var(--warn); }
.sev-low { color:var(--muted); }
.conf { margin-left:auto; font-size:.78rem; color:var(--muted); }
.field { margin-top:.6rem; }
.field b { display:block; font-size:.72rem; text-transform:uppercase; letter-spacing:.05em;
           color:var(--muted); margin-bottom:.15rem; font-weight:650; }
.field p { margin:0; font-size:.92rem; }
.fix { border-left:3px solid var(--brand); padding-left:.75rem; }
.empty { color:var(--muted); font-style:italic; }
"""


def build_html_report(results: list[ScanResult]) -> str:
    """Builds a self-contained developer-facing HTML report. Shows every
    finding, including dismissed false positives, unlike the SARIF report.
    Everything interpolated is HTML-escaped -- explanations and file paths
    both come from untrusted, LLM/repo-generated content."""
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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CredHunter-X scan report</title>
<style>{_STYLE}</style>
</head>
<body>
<header>{_logo()}<h1>CredHunter-X<small>scan report</small></h1></header>
<div class="tiles">
  {_build_tile(Label.TRUE_SECRET, counts, "secrets found")}
  {_build_tile(Label.UNCERTAIN, counts, "need review")}
  {_build_tile(Label.FALSE_POSITIVE, counts, "dismissed")}
</div>
{sections}
</body>
</html>
"""


def _build_tile(label: Label, counts: dict[Label, int], what: str) -> str:
    anchor = _LABEL_ANCHORS[label]
    return (
        f'<a class="tile {anchor}" href="#{anchor}">'
        f'<span class="count">{counts[label]}</span>'
        f'<span class="what">{escape(what)}</span></a>'
    )


def _build_section(label: Label, rows: list[ScanResult]) -> str:
    title = escape(_LABEL_SECTION_TITLES[label])
    anchor = _LABEL_ANCHORS[label]
    if not rows:
        return f'<h2 id="{anchor}">{title}</h2>\n<p class="empty">None.</p>\n'

    rows = sorted(rows, key=lambda r: _SEVERITY_ORDER.get(r.classification.severity, 99))
    findings = "\n".join(_build_finding(r, dismissed=label == Label.FALSE_POSITIVE) for r in rows)
    return f'<h2 id="{anchor}">{title}</h2>\n{findings}\n'


def _build_finding(result: ScanResult, *, dismissed: bool) -> str:
    c, r = result.candidate, result.classification
    return f"""<div class="finding{" is-dismissed" if dismissed else ""}">
  <div class="head">
    <span class="loc">{escape(f"{c.file_path}:{c.line_start}")}</span>
    <span class="rule">{escape(rule_title(c.rule_id))}</span>
    <span class="badge sev-{r.severity}">{escape(r.severity)}</span>
    <span class="conf">confidence {r.confidence:.2f}</span>
  </div>
  <div class="field"><b>Why</b><p>{escape(r.explanation)}</p></div>
  <div class="field fix"><b>How to fix it</b><p>{escape(r.remediation)}</p></div>
</div>"""


def write_html_report(results: list[ScanResult], path: Path) -> None:
    path.write_text(build_html_report(results), encoding="utf-8")
