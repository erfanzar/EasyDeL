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

"""Preserve the documented public eSurge helper capabilities (host-only)."""

import pytest
from easydel.inference.esurge.utils import (
    chunk_list,
    is_list_of,
    next_power_of_2,
    prev_power_of_2,
    round_down,
    round_up,
)

pytestmark = pytest.mark.cpu_ok


def test_documented_list_helpers() -> None:
    assert is_list_of([1, 2, 3], int)
    assert not is_list_of([1, "a", 3], int, check="all")
    assert is_list_of([], str)
    assert not is_list_of((1, 2), int)
    assert list(chunk_list([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(chunk_list([1, 2, 3], 5)) == [[1, 2, 3]]


@pytest.mark.parametrize(
    ("value", "typ", "check", "expected"),
    [
        ([1, "later"], int, "first", True),
        (["first", 1], int, "first", False),
        ([1, "later"], int, "all", False),
        ([1, 2], int, "all", True),
        ([1, "two"], (int, str), "all", True),
        ([], int, "first", True),
        ([], int, "all", True),
        ([1], int, "invalid", False),
        ([], int, "invalid", False),
        ((1, 2), int, "invalid", False),
    ],
)
def test_list_type_guard_preserves_serving_compatibility(value, typ, check, expected) -> None:
    assert is_list_of(value, typ, check=check) is expected


def test_list_type_guard_preserves_invalid_type_errors() -> None:
    with pytest.raises(TypeError):
        is_list_of([1], "not-a-type", check="all")


def test_documented_integer_helpers() -> None:
    assert [next_power_of_2(n) for n in (5, 8, 0)] == [8, 8, 1]
    assert [prev_power_of_2(n) for n in (5, 8, 0)] == [4, 8, 0]
    assert [round_up(n, 4) for n in (7, 8, 9)] == [8, 8, 12]
    assert [round_down(n, 4) for n in (7, 8, 9)] == [4, 8, 8]
