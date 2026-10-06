"""How result files are named. Shared by run_evaluation.py, which writes
them, and compare_treatments.py, which reads them.
"""

from __future__ import annotations

import re


def filename_slug(model: str) -> str:
    """Model ids include the provider ("openai/gpt-5.6-luna"); only the last
    part is kept, and anything not safe in a file name is replaced."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", model.rsplit("/", 1)[-1])


def result_stem(*, arm: str, treatment: str, split: str, source: str, model: str) -> str:
    """Builds the file stem from the five things that identify a run: arm,
    treatment, split, source and model."""
    return f"{arm}_{treatment}_{split}_{source}_{filename_slug(model)}"
