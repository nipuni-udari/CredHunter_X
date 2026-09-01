from __future__ import annotations

import pytest

from credhunter_x.evaluation.result_files import filename_slug, result_stem


def test_stem_matches_the_names_already_written_to_results():
    """The naming rule moved out of run_evaluation.py so compare_treatments.py
    could share it. Existing result files must keep resolving, so this pins
    the exact names on disk from the completed --split all runs."""
    assert (
        result_stem(
            arm="single",
            treatment="raw",
            split="all",
            source="combined",
            model="openai/gpt-5.6-luna",
        )
        == "single_raw_all_combined_gpt-5.6-luna"
    )
    assert (
        result_stem(
            arm="single",
            treatment="masked",
            split="all",
            source="combined",
            model="openai/gpt-5.6-luna",
        )
        == "single_masked_all_combined_gpt-5.6-luna"
    )


def test_stem_carries_all_five_identifying_fields():
    stem = result_stem(
        arm="agentic",
        treatment="metadata_only",
        split="dev",
        source="trufflehog",
        model="openai/gpt-5.6-luna",
    )

    assert stem == "agentic_metadata_only_dev_trufflehog_gpt-5.6-luna"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/gpt-5.6-luna", "gpt-5.6-luna"),
        # nested provider path -- only the trailing segment identifies the model
        ("openrouter/google/gemma-4-26b-a4b-it", "gemma-4-26b-a4b-it"),
        ("gemini/gemini-flash-latest", "gemini-flash-latest"),
        # bare model id, no provider prefix
        ("some-model", "some-model"),
    ],
)
def test_slug_keeps_the_trailing_segment(model: str, expected: str):
    assert filename_slug(model) == expected


def test_slug_flattens_characters_that_would_break_a_path():
    """A "/" would create a directory; anything outside the safe set is
    replaced so the stem is always a single filename."""
    assert filename_slug("weird:model*name?v1") == "weird-model-name-v1"
    assert "/" not in filename_slug("a/b/c:d")
