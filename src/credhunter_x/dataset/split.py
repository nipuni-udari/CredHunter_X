from __future__ import annotations

import random

from credhunter_x.dataset.creddata import GroundTruthRow


def split_by_repo(
    rows: list[GroundTruthRow], *, dev_fraction: float = 0.5, seed: int = 42
) -> tuple[list[GroundTruthRow], list[GroundTruthRow]]:
    """Splits by RepoID, not by row — putting the same repository's rows
    across both dev and test would leak information (near-duplicate code,
    consistent per-repo secret styles) between the two sets. Deterministic
    given the same seed, so dev/test membership doesn't shift between
    runs."""
    repo_ids = sorted({row.repo_id for row in rows})
    rng = random.Random(seed)
    rng.shuffle(repo_ids)

    split_point = round(len(repo_ids) * dev_fraction)
    dev_repo_ids = set(repo_ids[:split_point])

    dev_rows = [row for row in rows if row.repo_id in dev_repo_ids]
    test_rows = [row for row in rows if row.repo_id not in dev_repo_ids]
    return dev_rows, test_rows
