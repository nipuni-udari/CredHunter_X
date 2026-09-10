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
  --brand:#6d28d9; --brand-soft:#f1ebfe;
  --crit:#c62828; --high:#e05a1c; --warn:#9a6700; --ok:#1a7f4b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#0e0e13; --card:#17171f; --ink:#f2f2f7; --muted:#a8a8b8; --line:#2f2f3d;
    --brand:#b39dfc; --brand-soft:rgba(124,58,237,.16);
    --crit:#e5484d; --high:#e07a3c; --warn:#d4a017; --ok:#3fb27f;
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
.head { display:flex; flex-wrap:wrap; align-items:center; gap:.5rem; margin-bottom:.8rem; }
.loc { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:.86rem;
       font-weight:600; background:rgba(125,125,150,.16); padding:.22rem .5rem;
       border-radius:5px; }
.rule { font-size:.78rem; font-weight:650; letter-spacing:.01em; color:var(--brand);
        background:var(--brand-soft); padding:.22rem .55rem; border-radius:5px; }
.badge { font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
         padding:.2rem .55rem; border-radius:20px; color:#fff; }
.sev-critical { background:var(--crit); } .sev-high { background:var(--high); }
.sev-medium { background:var(--warn); } .sev-low { background:var(--muted); }
.conf { margin-left:auto; font-size:.74rem; font-weight:650; letter-spacing:.03em;
        text-transform:uppercase; color:var(--ink); background:rgba(125,125,150,.16);
        border-radius:20px; padding:.2rem .6rem; }
.conf b { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-weight:700; }
.field { margin-top:.75rem; }
.field b { display:block; font-size:.7rem; text-transform:uppercase; letter-spacing:.08em;
           margin-bottom:.25rem; font-weight:700; color:var(--brand); }
.field p { margin:0; font-size:.93rem; }
.fix { border-left:3px solid var(--brand); background:var(--brand-soft);
       border-radius:0 8px 8px 0; padding:.6rem .85rem; }
.fix b { color:var(--brand); }
.empty { color:var(--muted); font-style:italic; }

.filters { position:sticky; top:0; z-index:5; display:flex; flex-wrap:wrap; gap:.5rem;
           align-items:center; background:var(--bg); border-bottom:1px solid var(--line);
           padding:.7rem 0; margin-bottom:.5rem; }
.filters label { font-size:.72rem; text-transform:uppercase; letter-spacing:.06em;
                 font-weight:700; color:var(--muted); }
.chip { font:inherit; font-size:.78rem; font-weight:600; cursor:pointer; color:var(--muted);
        background:var(--card); border:1px solid var(--line); border-radius:20px;
        padding:.25rem .7rem; }
.chip:hover { border-color:var(--brand); color:var(--ink); }
.chip[aria-pressed="true"] { background:var(--brand); border-color:var(--brand); color:#fff; }
.filters select, .filters input { font:inherit; font-size:.82rem; color:var(--ink);
        background:var(--card); border:1px solid var(--line); border-radius:7px;
        padding:.3rem .5rem; }
.filters input { flex:1; min-width:140px; }
.shown { margin-left:auto; font-size:.76rem; color:var(--muted); }
.is-hidden { display:none; }
"""

# Client-side only: the report is one file opened from disk, so filtering has
# to happen in the page. Tiles stay real anchors and still work without JS.
_SCRIPT = """
const findings = [...document.querySelectorAll('.finding')];
const state = {label:'', sev:'', rule:'', q:''};
const shown = document.querySelector('.shown');

function apply() {
  let n = 0;
  for (const el of findings) {
    const d = el.dataset;
    const hit = (!state.label || d.label === state.label)
      && (!state.sev || d.severity === state.sev)
      && (!state.rule || d.rule === state.rule)
      && (!state.q || el.textContent.toLowerCase().includes(state.q));
    el.classList.toggle('is-hidden', !hit);
    if (hit) n++;
  }
  for (const h of document.querySelectorAll('h2[id]')) {
    const group = [];
    for (let s = h.nextElementSibling; s && s.tagName !== 'H2'; s = s.nextElementSibling) {
      group.push(s);
    }
    const live = group.some(
      s => s.classList.contains('finding') && !s.classList.contains('is-hidden')
    );
    const empty = group.filter(s => s.classList.contains('empty'));
    const filtering = !!(state.sev || state.rule || state.q);
    h.classList.toggle('is-hidden', !live && !empty.length);
    for (const s of empty) s.classList.toggle('is-hidden', filtering);
  }
  shown.textContent = `showing ${n} of ${findings.length}`;
}

for (const b of document.querySelectorAll('.chip[data-sev]')) {
  b.onclick = () => {
    state.sev = b.dataset.sev === state.sev ? '' : b.dataset.sev;
    for (const o of document.querySelectorAll('.chip[data-sev]')) {
      o.setAttribute('aria-pressed', String(o.dataset.sev === state.sev));
    }
    apply();
  };
}
document.querySelector('#rule-filter').onchange = e => { state.rule = e.target.value; apply(); };
document.querySelector('#text-filter').oninput = e => {
  state.q = e.target.value.trim().toLowerCase(); apply();
};
for (const t of document.querySelectorAll('.tile')) {
  t.onclick = () => { state.label = t.dataset.label; apply(); };
}
apply();
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
    filters = _build_filters(results)

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
{filters}
{sections}
<script>{_SCRIPT}</script>
</body>
</html>
"""


def _build_filters(results: list[ScanResult]) -> str:
    """Severity chips plus a rule dropdown built from the rules actually
    present -- offering a filter that matches nothing is just noise."""
    severities = {r.classification.severity for r in results}
    chips = "".join(
        f'<button class="chip" data-sev="{s}" aria-pressed="false">{escape(s)}</button>'
        for s in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW)
        if s in severities
    )
    rules = sorted({rule_title(r.candidate.rule_id) for r in results})
    options = "".join(f'<option value="{escape(r)}">{escape(r)}</option>' for r in rules)
    return f"""<div class="filters">
  <label>Severity</label>{chips}
  <label>Rule</label>
  <select id="rule-filter"><option value="">All</option>{options}</select>
  <input id="text-filter" type="search" placeholder="Search file, reason, fix...">
  <span class="shown"></span>
</div>"""


def _build_tile(label: Label, counts: dict[Label, int], what: str) -> str:
    anchor = _LABEL_ANCHORS[label]
    return (
        f'<a class="tile {anchor}" href="#{anchor}" data-label="{label}">'
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
    rule = rule_title(c.rule_id)
    return f"""<div class="finding{" is-dismissed" if dismissed else ""}"
     data-label="{r.label}" data-severity="{r.severity}" data-rule="{escape(rule)}">
  <div class="head">
    <span class="loc">{escape(f"{c.file_path}:{c.line_start}")}</span>
    <span class="rule">{escape(rule)}</span>
    <span class="badge sev-{r.severity}">{escape(r.severity)}</span>
    <span class="conf">confidence <b>{r.confidence:.2f}</b></span>
  </div>
  <div class="field"><b>Why</b><p>{escape(r.explanation)}</p></div>
  <div class="field fix"><b>How to fix it</b><p>{escape(r.remediation)}</p></div>
</div>"""


def write_html_report(results: list[ScanResult], path: Path) -> None:
    path.write_text(build_html_report(results), encoding="utf-8")
