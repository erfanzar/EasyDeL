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

"""Tests for ``easydel.inference.esurge.utils``.

Pure helpers used by eSurge scheduling and sampling. Coverage:

* ``ConstantList`` -- read-only Sequence wrapper, mutation methods raise
* ``is_list_of`` -- list type-guard with first/all checking modes
* ``chunk_list`` -- generator yields fixed-size sublists
* ``cdiv`` / ``next_power_of_2`` / ``prev_power_of_2`` / ``round_up`` /
  ``round_down`` -- integer math helpers
* ``get_dtype_size`` -- dtype byte width
* ``truncate_tokens`` -- left/right/middle truncation strategies
* ``_rope_scaling_uses_mrope`` -- mRoPE detection across config shapes
* ``model_uses_mrope`` -- model-level RoPE inference
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from easydel.inference.esurge.utils import (
    ConstantList,
    _rope_scaling_uses_mrope,
    cdiv,
    chunk_list,
    get_dtype_size,
    is_list_of,
    model_uses_mrope,
    next_power_of_2,
    prev_power_of_2,
    round_down,
    round_up,
    truncate_tokens,
)


def test_constant_list_indexing_and_length():
    cl = ConstantList([1, 2, 3])
    assert cl[0] == 1
    assert cl[2] == 3
    assert len(cl) == 3


def test_constant_list_slice_returns_list():
    cl = ConstantList([1, 2, 3, 4])
    sliced = cl[1:3]
    assert sliced == [2, 3]


def test_constant_list_iter():
    cl = ConstantList([10, 20, 30])
    assert list(cl) == [10, 20, 30]


def test_constant_list_contains():
    cl = ConstantList(["a", "b"])
    assert "a" in cl
    assert "z" not in cl


def test_constant_list_index_finds_first_match():
    cl = ConstantList([10, 20, 30, 20])
    assert cl.index(20) == 1
    assert cl.index(20, start=2) == 3


def test_constant_list_index_raises_when_missing():
    cl = ConstantList([1, 2])
    with pytest.raises(ValueError):
        cl.index(99)


def test_constant_list_repr_contains_inner_list():
    cl = ConstantList([1, 2])
    assert repr(cl) == "ConstantList([1, 2])"


@pytest.mark.parametrize(
    "method,args",
    [
        ("append", (4,)),
        ("extend", ([4, 5],)),
        ("insert", (0, 4)),
        ("pop", ()),
        ("remove", (1,)),
        ("clear", ()),
    ],
)
def test_constant_list_mutation_methods_raise(method: str, args: tuple):
    cl = ConstantList([1, 2, 3])
    with pytest.raises(TypeError):
        getattr(cl, method)(*args)


def test_constant_list_setitem_raises():
    cl = ConstantList([1, 2, 3])
    with pytest.raises(TypeError):
        cl[0] = 99


def test_constant_list_delitem_raises():
    cl = ConstantList([1, 2, 3])
    with pytest.raises(TypeError):
        del cl[0]


def test_is_list_of_non_list_returns_false():
    assert is_list_of((1, 2, 3), int) is False
    assert is_list_of("abc", str) is False
    assert is_list_of(None, int) is False


def test_is_list_of_empty_list_first_mode_returns_true():
    """Empty list has no first element to mismatch -> True in 'first' mode."""
    assert is_list_of([], int) is True


def test_is_list_of_first_mode_only_checks_first_element():
    """In 'first' mode, mismatched later elements don't fail the check."""
    assert is_list_of([1, "a", "b"], int, check="first") is True
    assert is_list_of(["a", 1, 2], int, check="first") is False


def test_is_list_of_all_mode_checks_every_element():
    assert is_list_of([1, 2, 3], int, check="all") is True
    assert is_list_of([1, "a", 3], int, check="all") is False


def test_is_list_of_supports_tuple_of_types():
    assert is_list_of([1, "a"], (int, str), check="all") is True
    assert is_list_of([1, "a", 3.0], (int, str), check="all") is False


def test_chunk_list_evenly_divisible():
    assert list(chunk_list([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]


def test_chunk_list_last_chunk_may_be_smaller():
    assert list(chunk_list([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_chunk_list_chunk_size_larger_than_list():
    assert list(chunk_list([1, 2, 3], 10)) == [[1, 2, 3]]


def test_chunk_list_empty_input_yields_nothing():
    assert list(chunk_list([], 3)) == []


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (7, 3, 3),
        (6, 3, 2),
        (5, 3, 2),
        (1, 1, 1),
        (0, 5, 0),
        (10, 1, 10),
    ],
)
def test_cdiv(a: int, b: int, expected: int):
    assert cdiv(a, b) == expected


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 4),
        (5, 8),
        (8, 8),
        (9, 16),
        (1024, 1024),
    ],
)
def test_next_power_of_2(n: int, expected: int):
    assert next_power_of_2(n) == expected


def test_next_power_of_2_negative_returns_one():
    assert next_power_of_2(-5) == 1


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 2),
        (5, 4),
        (8, 8),
        (15, 8),
        (16, 16),
    ],
)
def test_prev_power_of_2(n: int, expected: int):
    assert prev_power_of_2(n) == expected


def test_prev_power_of_2_zero_or_negative():
    assert prev_power_of_2(0) == 0
    assert prev_power_of_2(-3) == 0


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (7, 4, 8),
        (8, 4, 8),
        (9, 4, 12),
        (0, 4, 0),
        (15, 16, 16),
    ],
)
def test_round_up(x: int, y: int, expected: int):
    assert round_up(x, y) == expected


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (7, 4, 4),
        (8, 4, 8),
        (9, 4, 8),
        (0, 4, 0),
    ],
)
def test_round_down(x: int, y: int, expected: int):
    assert round_down(x, y) == expected


@pytest.mark.parametrize(
    "dtype,expected_bytes",
    [
        (jnp.float32, 4),
        (jnp.float16, 2),
        (jnp.bfloat16, 2),
        (jnp.float64, 8),
        (jnp.int32, 4),
        (jnp.int64, 8),
        (jnp.int8, 1),
        (jnp.uint8, 1),
    ],
)
def test_get_dtype_size(dtype, expected_bytes: int):
    assert get_dtype_size(dtype) == expected_bytes


def test_truncate_tokens_below_target_no_op():
    truncated, dropped = truncate_tokens([1, 2, 3], target_len=10)
    assert truncated == [1, 2, 3]
    assert dropped == 0


def test_truncate_tokens_left_keeps_recent():
    truncated, dropped = truncate_tokens([1, 2, 3, 4, 5], target_len=3, mode="left")
    assert truncated == [3, 4, 5]
    assert dropped == 2


def test_truncate_tokens_right_keeps_initial():
    truncated, dropped = truncate_tokens([1, 2, 3, 4, 5], target_len=3, mode="right")
    assert truncated == [1, 2, 3]
    assert dropped == 2


def test_truncate_tokens_middle_keeps_both_ends():
    truncated, dropped = truncate_tokens([1, 2, 3, 4, 5], target_len=3, mode="middle")

    assert truncated == [1, 2, 5]
    assert dropped == 2


def test_truncate_tokens_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown truncate_mode"):
        truncate_tokens([1, 2, 3], target_len=2, mode="reverse")


def test_truncate_tokens_target_zero():
    """target_len=0 drops all tokens."""
    truncated, dropped = truncate_tokens([1, 2, 3], target_len=0, mode="left")
    assert truncated == []
    assert dropped == 3


def test_rope_scaling_uses_mrope_none():
    assert _rope_scaling_uses_mrope(None) is False


def test_rope_scaling_uses_mrope_empty_dict():
    assert _rope_scaling_uses_mrope({}) is False


def test_rope_scaling_uses_mrope_via_mrope_section():
    """Presence of ``mrope_section`` is the canonical mRoPE marker."""
    assert _rope_scaling_uses_mrope({"mrope_section": [16, 24, 24]}) is True


def test_rope_scaling_uses_mrope_via_rope_type_string():
    assert _rope_scaling_uses_mrope({"rope_type": "mrope"}) is True
    assert _rope_scaling_uses_mrope({"rope_type": "MRope"}) is True


def test_rope_scaling_uses_mrope_via_legacy_type_field():
    """Older configs use ``type`` instead of ``rope_type``."""
    assert _rope_scaling_uses_mrope({"type": "mrope"}) is True


def test_rope_scaling_uses_mrope_via_mrope_interleaved_flag():
    assert _rope_scaling_uses_mrope({"mrope_interleaved": True}) is True

    assert _rope_scaling_uses_mrope({"mrope_interleaved": False}) is True


def test_rope_scaling_uses_mrope_object_with_to_dict():
    """An object with ``to_dict()`` is recognized via the dict view."""

    class FakeRopeConfig:
        def to_dict(self):
            return {"rope_type": "mrope"}

    assert _rope_scaling_uses_mrope(FakeRopeConfig()) is True


def test_rope_scaling_uses_mrope_to_dict_failure_falls_back():
    """If ``to_dict()`` raises, the helper still returns False (defensive)."""

    class BadRope:
        def to_dict(self):
            raise RuntimeError("oops")

    assert _rope_scaling_uses_mrope(BadRope()) is False


def test_rope_scaling_uses_mrope_non_dict_returns_false():
    """A non-Mapping value (e.g. int, string) returns False."""
    assert _rope_scaling_uses_mrope(42) is False
    assert _rope_scaling_uses_mrope("mrope") is False


def test_rope_scaling_uses_mrope_unrelated_type_returns_false():
    assert _rope_scaling_uses_mrope({"rope_type": "linear"}) is False
    assert _rope_scaling_uses_mrope({"rope_type": "yarn"}) is False


def test_model_uses_mrope_none_returns_false():
    assert model_uses_mrope(None) is False


def test_model_uses_mrope_no_config_attr_returns_false():
    """A model without a ``.config`` attribute is rejected gracefully."""
    assert model_uses_mrope(SimpleNamespace()) is False


def test_model_uses_mrope_via_text_config_rope_scaling():
    """Standard path: ``model.config.text_config.rope_scaling`` carries mRoPE markers."""
    text_cfg = SimpleNamespace(rope_scaling={"mrope_section": [16, 24]})
    config = SimpleNamespace(text_config=text_cfg)
    model = SimpleNamespace(config=config)
    assert model_uses_mrope(model) is True


def test_model_uses_mrope_returns_false_for_non_mrope_config():
    text_cfg = SimpleNamespace(rope_scaling={"rope_type": "linear"})
    config = SimpleNamespace(text_config=text_cfg)
    model = SimpleNamespace(config=config)
    assert model_uses_mrope(model) is False
