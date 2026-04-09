# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comprehensive tests for tool parsing utility functions."""



from partial_json_parser.core.options import Allow

from easydel.inference.tools.utils import (
    consume_space,
    extract_intermediate_diff,
    find_all_indices,
    find_common_prefix,
    find_common_suffix,
    is_complete_json,
    partial_json_loads,
)


class TestFindCommonPrefix:
    def test_identical_strings(self):
        assert find_common_prefix("hello", "hello") == "hello"

    def test_shared_prefix(self):
        assert find_common_prefix("testing", "test") == "test"

    def test_no_common(self):
        assert find_common_prefix("hello", "world") == ""

    def test_empty_first(self):
        assert find_common_prefix("", "hello") == ""

    def test_empty_second(self):
        assert find_common_prefix("hello", "") == ""

    def test_both_empty(self):
        assert find_common_prefix("", "") == ""

    def test_json_prefix(self):
        assert find_common_prefix('{"fruit": "ap"}', '{"fruit": "apple"}') == '{"fruit": "ap'

    def test_one_char_match(self):
        assert find_common_prefix("abc", "axy") == "a"

    def test_symmetric(self):
        assert find_common_prefix("test", "testing") == find_common_prefix("testing", "test")


class TestFindCommonSuffix:
    def test_json_suffix(self):
        assert find_common_suffix('{"fruit": "ap"}', '{"fruit": "apple"}') == '"}'

    def test_no_alphanumeric(self):
        assert find_common_suffix("test123", "abc123") == ""

    def test_punctuation_suffix(self):
        assert find_common_suffix("hello!", "world!") == "!"

    def test_empty_strings(self):
        assert find_common_suffix("", "") == ""

    def test_no_common(self):
        assert find_common_suffix("abc", "xyz") == ""

    def test_only_special_chars(self):
        assert find_common_suffix("!@#", "!@#") == "!@#"


class TestExtractIntermediateDiff:
    def test_json_extension(self):
        result = extract_intermediate_diff('{"fruit": "apple"}', '{"fruit": "ap"}')
        assert result == "ple"

    def test_number_extension(self):
        result = extract_intermediate_diff('{"a": 123}', '{"a": 1}')
        assert result == "23"

    def test_simple_extension(self):
        result = extract_intermediate_diff("hello world", "hello ")
        assert result == "world"

    def test_identical(self):
        result = extract_intermediate_diff("hello", "hello")
        assert result == ""

    def test_empty_old(self):
        result = extract_intermediate_diff("hello", "")
        assert result == "hello"


class TestFindAllIndices:
    def test_multiple_occurrences(self):
        assert find_all_indices("hello hello hello", "hello") == [0, 6, 12]

    def test_overlapping(self):
        assert find_all_indices("aaa", "aa") == [0, 1]

    def test_no_match(self):
        assert find_all_indices("test", "xyz") == []

    def test_single_match(self):
        assert find_all_indices("hello world", "world") == [6]

    def test_empty_string(self):
        assert find_all_indices("", "test") == []

    def test_consecutive(self):
        assert find_all_indices("abcabc", "bc") == [1, 4]


class TestPartialJsonLoads:
    def test_complete_json(self):
        obj, _consumed = partial_json_loads('{"name": "test"}', Allow.ALL)
        assert obj == {"name": "test"}

    def test_incomplete_json(self):
        obj, _consumed = partial_json_loads('{"name": "test"', Allow.ALL)
        assert isinstance(obj, dict)
        assert obj["name"] == "test"

    def test_extra_data(self):
        obj, consumed = partial_json_loads('{"a": 1}{"b": 2}', Allow.ALL)
        assert obj == {"a": 1}
        assert consumed == 8

    def test_simple_number(self):
        obj, _consumed = partial_json_loads("42", Allow.ALL)
        assert obj == 42

    def test_simple_string(self):
        obj, _consumed = partial_json_loads('"hello"', Allow.ALL)
        assert obj == "hello"

    def test_array(self):
        obj, _consumed = partial_json_loads("[1, 2, 3]", Allow.ALL)
        assert obj == [1, 2, 3]


class TestIsCompleteJson:
    def test_complete_object(self):
        assert is_complete_json('{"name": "test"}') is True

    def test_incomplete_object(self):
        assert is_complete_json('{"name": "test"') is False

    def test_complete_array(self):
        assert is_complete_json("[1, 2, 3]") is True

    def test_incomplete_array(self):
        assert is_complete_json("[1, 2, ") is False

    def test_not_json(self):
        assert is_complete_json("not json at all") is False

    def test_null(self):
        assert is_complete_json("null") is True

    def test_number(self):
        assert is_complete_json("42") is True

    def test_string(self):
        assert is_complete_json('"hello"') is True

    def test_empty_string(self):
        assert is_complete_json("") is False

    def test_nested_object(self):
        assert is_complete_json('{"a": {"b": [1, 2]}}') is True


class TestConsumeSpace:
    def test_leading_spaces(self):
        assert consume_space(0, "   hello") == 3

    def test_no_spaces(self):
        assert consume_space(0, "hello") == 0

    def test_all_spaces(self):
        assert consume_space(0, "   ") == 3

    def test_middle_spaces(self):
        assert consume_space(5, "hello   world") == 8

    def test_tabs_and_newlines(self):
        assert consume_space(0, "\t\n\r hello") == 4

    def test_at_end(self):
        assert consume_space(5, "hello") == 5

    def test_empty_string(self):
        assert consume_space(0, "") == 0
