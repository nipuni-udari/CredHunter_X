"""How a run's result files are named.

Lives in the package rather than in a script because two scripts need it:
run_evaluation.py writes these files and compare_treatments.py reads them.
They each used to carry a private copy of the rule, and when --source and
the model were added to the name only the writer was updated -- so the RQ4
comparison table went on looking for the old names, silently reading
pre-rename files or finding nothing at all. One definition, imported twice.
"""

from __future__ import annotations

import re


def filename_slug(model: str) -> str:
    """Model ids are provider-qualified paths ("openai/gpt-5.6-luna",
    "openrouter/google/gemma-4-26b-a4b-it"); the trailing segment is what
    identifies the model and is what this project's archived result files
    already use by hand. "/" would create directories, so anything outside
    a safe filename set is flattened."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", model.rsplit("/", 1)[-1])


def result_stem(*, arm: str, treatment: str, split: str, source: str, model: str) -> str:
    """A run is identified by five things, so all five belong in the name.

    Leaving --source and the model out meant a gitleaks-only run and a
    combined run -- 368 candidates versus 517 -- both wrote to
    "single_masked_all.jsonl", the second silently destroying the first,
    with nothing inside the file recording which scanner produced it.
    Older results in this directory can no longer be attributed at all."""
    return f"{arm}_{treatment}_{split}_{source}_{filename_slug(model)}"
