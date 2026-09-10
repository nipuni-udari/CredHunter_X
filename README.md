# CredHunter-X

**Finds hardcoded secrets in Python code, then works out which ones are real.**

Existing secret scanners are good at spotting things that *look* like
credentials and bad at knowing whether they matter. CredHunter-X adds a second
opinion: it reads the code around each finding and decides whether it is a
genuine leaked credential, a test fixture, or a placeholder — and explains its
reasoning, so you can disagree with it.

It does that **without sending your real secret values to the LLM**.

---

## The problem it solves

A scanner reports this:

```
tests/fixtures.py:12    AKIAIOSFODNN7EXAMPLE
app/config.py:8         AKIAZ3XK7QW2NBVCXR4T
docs/setup.md:31        YOUR_API_KEY_HERE
```

Three alerts. **One of them is an emergency.** The other two are a test fixture
and a documentation placeholder. A pattern-matching scanner cannot tell them
apart, because all three match the same regex.

When most alerts are noise, people stop reading the alerts. That is the actual
failure — not the missed pattern, but the ignored dashboard.

CredHunter-X reads the surrounding code and reports:

```
app/config.py:8    true_secret     (confidence 0.94, severity critical)
    Live AWS access key assigned in application config and used by the
    boto3 client below. Rotate this key in IAM, then load it from the
    environment instead of committing it.

tests/fixtures.py:12   false_positive   -- dismissed, not reported
docs/setup.md:31       false_positive   -- dismissed, not reported
```

One alert instead of three, with the reason and the fix attached.

---

## How it works

```
   your repository
         │
         ├──  GitLeaks    ──┐
         │                  ├──  candidate findings
         └──  TruffleHog  ──┘          │
                                       ▼
                              ┌──────────────────┐
                              │  masking layer   │  real secret value is
                              │                  │  replaced before this point
                              └────────┬─────────┘
                                       ▼
                              ┌──────────────────┐
                              │   LLM classifier │  reads the surrounding code
                              └────────┬─────────┘
                                       ▼
                    true_secret  /  false_positive  /  uncertain
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                   report.sarif                report.html
              (GitHub Security tab)          (for a human)
```

**Two detectors, not one.** GitLeaks matches known credential formats;
TruffleHog additionally catches high-entropy strings that match no known
format. Findings are merged, so a secret spotted by either one gets reviewed.

**Three verdicts, not two.** `uncertain` exists deliberately. A tool that must
choose between "definitely a secret" and "definitely fine" will guess, and a
wrong guess on a real credential is the worst outcome available. `uncertain`
findings are reported for a human to decide.

**Failing to classify is not the same as clean.** If a candidate cannot be
classified at all — an unreadable model response, or a blocked call — the scan
exits non-zero and says so. It is never quietly counted as safe, because the
one it failed on might have been the real one.

---

## The privacy part

Sending source code to a third-party LLM is one thing. Sending the actual
working credential is another — it puts a live secret in someone else's logs.

CredHunter-X never does. Before anything reaches the model, the secret value is
replaced. The model sees the code around it, which is what it needs to judge
context, plus a description of the value (how long, what character set, what
format) — not the value itself.

Four levels are implemented:

| Level | What the model receives | Used by |
|---|---|---|
| `raw` | the real value | research only — never the shipped CLI |
| `masked` | `••••••••` plus a description | **the default** |
| `pseudonymised` | a fake value of the same shape | research |
| `metadata_only` | `[REDACTED]`, no length signal | research |

A **fail-closed leak guard** sits in front of every model call and aborts it if
a registered secret value is about to be transmitted. It is not a filter that
cleans the request — it stops the request.

The shipped CLI always uses `masked`. `raw` cannot be selected through
configuration at all; only a Python caller writing code against the library can
request it, which is how the research runs were done.

---

## Quick start — GitHub Actions

Scan every push and see results in your repository's **Security** tab.

Create `.github/workflows/secret-scan.yml`:

```yaml
name: Secret scan

on: [push, pull_request]

permissions:
  contents: read
  security-events: write   # required to publish into the Security tab

jobs:
  credhunter:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: nipuni-udari/CredHunter_X@v1
        with:
          llm-model: <provider>/<model>
          llm-api-key: ${{ secrets.LLM_API_KEY }}
```

Then add your API key: **Settings → Secrets and variables → Actions → New
repository secret**, named `LLM_API_KEY`.

You supply your own model and your own key — the action never assumes a
provider. Any litellm `provider/model` string works.

### Options

| Input | Default | What it does |
|---|---|---|
| `llm-model` | **required** | e.g. `openai/gpt-5.6-luna`, `anthropic/claude-sonnet-5` |
| `llm-api-key` | **required** | your provider key, from a repository secret |
| `mode` | `agentic` | `agentic` reads nearby files before deciding; `single` is one call per finding and cheaper |
| `llm-reasoning-effort` | `medium` | for reasoning-capable models |
| `path` | `.` | directory to scan |
| `fail-on-secret` | `true` | `false` reports without blocking the merge |
| `gitleaks-version` | `8.18.4` | pinned, so results do not drift |
| `trufflehog-version` | `3.0.10` | pinned, so results do not drift |

Both detector versions are pinned on purpose. An unpinned detector silently
changes what reaches the classifier, and you would see results move with no
change on your side.

**The repository must be public** for SARIF to reach the Security tab — code
scanning is free on public repositories, and needs GitHub Advanced Security on
private ones.

---

## Quick start — command line

Requires [`uv`](https://github.com/astral-sh/uv), plus `gitleaks` and
`trufflehog3` on `PATH`.

```powershell
uv sync
uv run credhunter-x path/to/repo
```

Copy `.env.example` to `.env` and set `LLM_MODEL` and `LLM_API_KEY` first.

Write reports as well as printing to the terminal:

```powershell
uv run credhunter-x path/to/repo --sarif report.sarif --html report.html
```

`report.sarif` is the machine-readable format GitHub ingests — only
`true_secret` and `uncertain` findings go in it, so it genuinely reduces the
list rather than relabelling every hit. `report.html` is a self-contained page
for a person, and it *does* list the dismissed false positives, so you can
check what was thrown away and why.

**Exit code 0** means clean. **Exit code 1** means either a real secret was
found, or something could not be classified.

**Choosing the arm:** create `.secretscan.yml` in the scanned repository with
`mode: single` or `mode: agentic` (default `agentic`).

**Viewing a SARIF file locally:** VS Code's official "SARIF Viewer" extension
opens it, or treat it as plain JSON:

```bash
jq '.runs[0].results[] | {rule: .ruleId, level: .level, message: .message.text}' report.sarif
```

---

## Status

Working and tested: two-detector candidate collection (GitLeaks + TruffleHog),
two classifier arms (single-prompt and agentic/tool-using), four sanitisation
treatments, the fail-closed leak guard, the evaluation harness against CredData
ground truth, the remediation-quality scoring pipeline, SARIF/HTML reporting,
and the composite GitHub Action. **299 tests passing**, no API calls needed to
run them.

Proven live end to end against
[nipuni-udari/credhunter-x-demo](https://github.com/nipuni-udari/credhunter-x-demo):
a real push was correctly waved through with an explained false-positive
dismissal, and a second push carrying a hardcoded credential with no "this is
fake" tell correctly failed the check — SARIF alert in the Security tab, HTML
report attached as a workflow artifact. That run used a hand-written workflow
installing from a pinned tag; `action.yml` packages the same steps behind a
single `uses:` line.

Not published to PyPI. The action installs from this repository, so a tag is
all you need.

---

## Research background

This is an MSc Applied AI dissertation project (CO7047): *LLM-Assisted Secret
Detection and Explanation in Python Source Code Repositories*.

Four questions are evaluated against
[Samsung/CredData](https://github.com/Samsung/CredData) ground truth:

- **RQ1** — does LLM triage improve precision over the detectors alone?
- **RQ2** — is the agentic arm worth its extra cost over a single prompt?
- **RQ3** — are the explanations and remediations technically correct?
- **RQ4** — what accuracy does it cost to never send the real secret? *This is
  the contribution — the other three establish that there is something worth
  protecting.*

Every result comes from one frozen 517-candidate set, so the arms and
treatments are compared on identical inputs rather than separate scans.

### Reproducing the evaluation

These cost real LLM quota and are not part of CI. Each writes
`results/{arm}_{treatment}_{split}.jsonl` plus a `_summary.json`.

```powershell
uv run python scripts/run_evaluation.py --split all --arm gitleaks_only

uv run python scripts/run_evaluation.py --split all --arm single  --treatment raw
uv run python scripts/run_evaluation.py --split all --arm single  --treatment masked
uv run python scripts/run_evaluation.py --split all --arm single  --treatment pseudonymised
uv run python scripts/run_evaluation.py --split all --arm single  --treatment metadata_only

uv run python scripts/run_evaluation.py --split all --arm agentic --treatment raw
uv run python scripts/run_evaluation.py --split all --arm agentic --treatment masked
uv run python scripts/run_evaluation.py --split all --arm agentic --treatment pseudonymised
uv run python scripts/run_evaluation.py --split all --arm agentic --treatment metadata_only
```

Aggregate one arm's four treatments into the RQ4 comparison table — pure
aggregation, no LLM calls:

```powershell
uv run python scripts/compare_treatments.py --arm single  --split all
uv run python scripts/compare_treatments.py --arm agentic --split all
```

### Remediation quality (RQ3)

Scores each remediation against `remediation_reference.yaml`, a standard
written **before** any results were looked at.

```powershell
# pass rate + Wilson 95% CI per rule, against the reference standard
uv run python scripts/score_remediation_quality.py --arm single --treatment raw --split all

# is the automated checker trustworthy? compare it to blind human labels
uv run python scripts/export_for_hand_labeling.py --arm both --treatment raw --split all
# ... fill in the human_present column by hand ...
uv run python scripts/compare_hand_labels_to_checker.py --labels results/hand_labeling_nipuni.csv

# do the two arms give different advice quality? paired, no LLM calls
uv run python scripts/rq3_arm_comparison.py

# the model's own repeat-classification stability (Fleiss' kappa)
uv run python scripts/check_llm_consistency.py --arm single --split all --n 10 --repeats 5
```

### Dataset setup (one-time, already done here)

CredData's downloader reconstructs repositories with filenames that are invalid
on NTFS, so fetching has to happen on a native Linux filesystem:

```bash
# inside WSL, cloned into WSL's own ext4 home -- not /mnt/c/...
uv run python scripts/fetch_creddata.py
```

Then, back on Windows, filter the annotations down to Python files:

```powershell
uv run python scripts/filter_python_creddata.py
```

---

## Development

```powershell
uv sync
uv run pre-commit install

uv run pytest -m "not live"      # 299 tests, no API calls -- what CI runs
uv run pytest -m live            # real API calls against the configured provider
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

---

## License

TBD — a license must be chosen before this repository is made public.
