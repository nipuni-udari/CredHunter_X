# CredHunter-X

LLM-assisted secret detection and explanation for Python source code, with a
privacy-preserving masking layer that measures the cost of never sending real
secret values to a third-party LLM.

Dissertation project (MSc Applied AI, CO7047): *LLM-Assisted Secret Detection
and Explanation in Python Source Code Repositories*.

## Status

Implemented: GitLeaks-based candidate detection, two classifier arms
(single-prompt and agentic/tool-using), four sanitisation treatments (`raw`,
`masked`, `pseudonymised`, `metadata_only`), a fail-closed leak guard, the
evaluation harness against CredData ground truth, and the remediation-quality
scoring pipeline (RQ3). 169 tests passing.

Not done, and not this codebase's job to do: the content of
`remediation_reference.yaml` (a pre-registered methodological step — must be
written blind to any results) and hand-labeling remediations for the Cohen's
kappa validation. See that file's own header comment.

## Setup

Requires [`uv`](https://github.com/astral-sh/uv) and
[`gitleaks`](https://github.com/gitleaks/gitleaks) on `PATH`.

```powershell
uv sync
uv run pre-commit install
```

Copy `.env.example` to `.env` and fill in an LLM provider/model and API key
(litellm-style `provider/model` string, e.g. `openrouter/google/gemma-4-26b-a4b-it`
or `gemini/gemini-flash-latest`). Switching providers is a config change only.

### Dataset (one-time, already done in this repo's `data/` — skip unless refreshing)

[Samsung/CredData](https://github.com/Samsung/CredData)'s own downloader
reconstructs repos with filenames that are invalid on NTFS, so fetching must
happen on a native Linux filesystem:

```bash
# inside WSL, cloned into WSL's own ext4 home (not /mnt/c/...)
uv run python scripts/fetch_creddata.py
```

Then, back on Windows, filter the annotations down to Python files (writes
`data/processed/creddata_python_labels.csv`, the file every evaluation run
reads from):

```powershell
uv run python scripts/filter_python_creddata.py
```

## Running the shipped CLI

Scans one repo, using `.secretscan.yml` for `mode` (`single`/`agentic` —
defaults to `agentic` if the file or key is absent). `treatment` is not
user-configurable here and always resolves to `masked`, the privacy-safe
default; `raw` is research-only and can only be requested by a Python
caller constructing `ScanConfig` directly, never via `.secretscan.yml`.

```powershell
uv run credhunter-x path/to/repo
```

Exits non-zero if any candidate is classified `true_secret` — this is the
signal a CI check gates on to fail the build/block the merge. Optionally
write machine- and human-readable reports:

```powershell
uv run credhunter-x path/to/repo --sarif report.sarif --html report.html
```

`report.sarif` is what GitHub's code scanning UI ingests (only
`true_secret`/`uncertain` findings are included, to actually reduce noise
rather than just relabel every GitLeaks hit); `report.html` is a
self-contained page for a human, and also lists dismissed false positives
for transparency.

## Running the research evaluation

`scripts/run_evaluation.py` scores GitLeaks alone, or GitLeaks + an LLM
classifier, against CredData's ground truth. Costs real LLM quota for
`single`/`agentic` arms — not part of CI, run manually. Each run writes
`results/{arm}_{treatment}_{split}.jsonl` plus a `_summary.json`.

```powershell
uv run python scripts/run_evaluation.py --split all --arm gitleaks_only

uv run python scripts/run_evaluation.py --split all --arm single --treatment raw
uv run python scripts/run_evaluation.py --split all --arm single --treatment masked
uv run python scripts/run_evaluation.py --split all --arm single --treatment pseudonymised
uv run python scripts/run_evaluation.py --split all --arm single --treatment metadata_only

uv run python scripts/run_evaluation.py --split all --arm agentic --treatment raw
uv run python scripts/run_evaluation.py --split all --arm agentic --treatment masked
uv run python scripts/run_evaluation.py --split all --arm agentic --treatment pseudonymised
uv run python scripts/run_evaluation.py --split all --arm agentic --treatment metadata_only
```

`--split` also accepts `dev`/`test` for a smaller/held-out subset. Then
aggregate the four treatments for one arm into the RQ4 comparison table
(precision/recall/F1, Δ vs raw, bytes of real secret sent) — pure
aggregation over the files above, no LLM calls:

```powershell
uv run python scripts/compare_treatments.py --arm single --split all
uv run python scripts/compare_treatments.py --arm agentic --split all
```

### Remediation quality (RQ3)

Requires `remediation_reference.yaml`'s `required_elements` to be filled in
first (pre-registered, blind to results — see the file's header comment).

```powershell
# pass rate + Wilson 95% CI per rule, against the reference standard
uv run python scripts/score_remediation_quality.py --arm single --treatment raw --split all

# model's own repeat-classification stability (Fleiss' kappa)
uv run python scripts/check_llm_consistency.py --arm single --split all --n 10 --repeats 5

# human-validation of the automated element-checker
uv run python scripts/export_for_hand_labeling.py --arm single --treatment raw --split all
# ... fill in the human_present column by hand in results/hand_labeling_sample.csv ...
uv run python scripts/compare_hand_labels_to_checker.py --labels results/hand_labeling_sample.csv
```

## Tests

```powershell
uv run pytest -m "not live"      # unit + integration, no API calls, what CI runs
uv run pytest -m live            # real API calls against the configured provider
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

## License

TBD — choose a license (e.g. MIT, Apache-2.0) before making this repository
public.
