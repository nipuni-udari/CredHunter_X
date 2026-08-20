from __future__ import annotations

from credhunter_x.dataset.creddata import GroundTruthRow
from credhunter_x.dataset.split import split_by_repo


def make_row(repo_id: str, row_id: str) -> GroundTruthRow:
    return GroundTruthRow(
        id=row_id,
        file_id=f"file-{row_id}",
        repo_id=repo_id,
        file_path=f"data/{repo_id}/{row_id}.py",
        line_start=1,
        line_end=1,
        ground_truth=False,
        value_start=-1,
        value_end=-1,
        category="Key",
    )


def make_rows(repo_ids: list[str], rows_per_repo: int = 3) -> list[GroundTruthRow]:
    return [
        make_row(repo_id, f"{repo_id}-{i}") for repo_id in repo_ids for i in range(rows_per_repo)
    ]


def test_split_never_puts_the_same_repo_in_both_sets():
    rows = make_rows(["r1", "r2", "r3", "r4", "r5", "r6"])

    dev, test = split_by_repo(rows, seed=1)

    dev_repos = {r.repo_id for r in dev}
    test_repos = {r.repo_id for r in test}
    assert dev_repos.isdisjoint(test_repos)


def test_split_covers_every_row_exactly_once():
    rows = make_rows(["r1", "r2", "r3", "r4"])

    dev, test = split_by_repo(rows, seed=1)

    assert sorted(r.id for r in dev + test) == sorted(r.id for r in rows)
    assert len(dev) + len(test) == len(rows)


def test_split_is_deterministic_given_the_same_seed():
    rows = make_rows(["r1", "r2", "r3", "r4", "r5"])

    dev1, test1 = split_by_repo(rows, seed=7)
    dev2, test2 = split_by_repo(rows, seed=7)

    assert [r.id for r in dev1] == [r.id for r in dev2]
    assert [r.id for r in test1] == [r.id for r in test2]


def test_split_respects_dev_fraction_at_the_repo_level():
    rows = make_rows(["r1", "r2", "r3", "r4"])

    dev, test = split_by_repo(rows, dev_fraction=0.5, seed=1)

    dev_repos = {r.repo_id for r in dev}
    test_repos = {r.repo_id for r in test}
    assert len(dev_repos) == 2
    assert len(test_repos) == 2
