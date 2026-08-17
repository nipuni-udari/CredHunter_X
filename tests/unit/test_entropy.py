from __future__ import annotations

import math

from credhunter_x.masking.entropy import classify_charset, shannon_entropy


def test_shannon_entropy_of_empty_string_is_zero():
    assert shannon_entropy("") == 0.0


def test_shannon_entropy_of_single_repeated_char_is_zero():
    assert shannon_entropy("aaaa") == 0.0


def test_shannon_entropy_of_four_distinct_chars_is_two_bits():
    assert math.isclose(shannon_entropy("abcd"), 2.0)


def test_classify_charset_detects_each_category():
    assert classify_charset("ABC") == "A-Z"
    assert classify_charset("abc") == "a-z"
    assert classify_charset("123") == "0-9"
    assert classify_charset("!@#") == "symbols"


def test_classify_charset_combines_categories_in_order():
    assert classify_charset("Ab1!") == "A-Za-z0-9symbols"


def test_classify_charset_of_empty_string_is_empty():
    assert classify_charset("") == ""
