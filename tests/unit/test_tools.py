from __future__ import annotations

from pathlib import Path

from credhunter_x.classifiers.tools.check_file_exists import check_file_exists
from credhunter_x.classifiers.tools.get_gitleaks_rule import get_gitleaks_rule
from credhunter_x.classifiers.tools.search_file import search_file


def test_check_file_exists_true_for_a_real_file(tmp_path: Path):
    (tmp_path / "app.py").write_text("x = 1\n")
    assert "exists" in check_file_exists(tmp_path, "app.py")
    assert "does not exist" not in check_file_exists(tmp_path, "app.py")


def test_check_file_exists_false_for_a_missing_file(tmp_path: Path):
    assert "does not exist" in check_file_exists(tmp_path, "missing.py")


def test_check_file_exists_reports_a_directory_distinctly(tmp_path: Path):
    (tmp_path / "subdir").mkdir()
    result = check_file_exists(tmp_path, "subdir")
    assert "directory" in result


def test_check_file_exists_refuses_path_traversal_outside_the_repo(tmp_path: Path):
    result = check_file_exists(tmp_path, "../../../../etc/passwd")
    assert "outside the repository" in result


def test_search_file_finds_a_match_with_context(tmp_path: Path):
    (tmp_path / "app.py").write_text("a = 1\nb = 2\nTOKEN = 'x'\nc = 3\nd = 4\n")
    result = search_file(tmp_path, "app.py", "TOKEN")
    assert "1 match" in result
    assert "TOKEN = 'x'" in result
    # surrounding context lines included
    assert "b = 2" in result
    assert "c = 3" in result


def test_search_file_reports_no_matches(tmp_path: Path):
    (tmp_path / "app.py").write_text("a = 1\n")
    result = search_file(tmp_path, "app.py", "nonexistent")
    assert "No matches" in result


def test_search_file_errors_for_a_missing_file(tmp_path: Path):
    result = search_file(tmp_path, "missing.py", "x")
    assert "does not exist" in result


def test_search_file_errors_for_an_empty_query(tmp_path: Path):
    (tmp_path / "app.py").write_text("a = 1\n")
    result = search_file(tmp_path, "app.py", "")
    assert "error" in result


def test_search_file_refuses_path_traversal_outside_the_repo(tmp_path: Path):
    result = search_file(tmp_path, "../../../../etc/passwd", "root")
    assert "outside the repository" in result


def test_search_file_caps_the_number_of_returned_matches(tmp_path: Path):
    (tmp_path / "app.py").write_text("\n".join(f"TOKEN {i}" for i in range(20)) + "\n")
    result = search_file(tmp_path, "app.py", "TOKEN")
    assert "20 match(es)" in result  # total count is still reported...
    assert result.count("---") == 4  # ...but only 5 (_MAX_MATCHES) blocks are shown


def test_get_gitleaks_rule_returns_a_known_description():
    result = get_gitleaks_rule("aws-access-token")
    assert "AWS" in result


def test_get_gitleaks_rule_falls_back_for_an_unknown_rule():
    result = get_gitleaks_rule("some-made-up-rule")
    assert "some-made-up-rule" in result
    assert "No specific description" in result


def test_get_gitleaks_rule_errors_for_an_empty_rule_id():
    result = get_gitleaks_rule("")
    assert "error" in result
