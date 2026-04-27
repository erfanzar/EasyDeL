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

"""Comprehensive tests for eSurge utility functions."""

import typing

import pytest
from jax import numpy as jnp

from easydel.inference.esurge.utils import (
    ConstantList,
    _get_text_config,
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


class TestConstantList:
    def test_read_access(self):
        cl = ConstantList([1, 2, 3])
        assert cl[0] == 1
        assert cl[-1] == 3
        assert len(cl) == 3

    def test_slice(self):
        cl = ConstantList([1, 2, 3, 4])
        assert cl[1:3] == [2, 3]

    def test_iteration(self):
        cl = ConstantList([1, 2, 3])
        assert list(cl) == [1, 2, 3]

    def test_contains(self):
        cl = ConstantList([1, 2, 3])
        assert 2 in cl
        assert 5 not in cl

    def test_index(self):
        cl = ConstantList([10, 20, 30])
        assert cl.index(20) == 1

    def test_index_not_found(self):
        cl = ConstantList([1, 2, 3])
        with pytest.raises(ValueError):
            cl.index(99)

    def test_append_raises(self):
        cl = ConstantList([1, 2])
        with pytest.raises(TypeError, match="Cannot append"):
            cl.append(3)

    def test_extend_raises(self):
        cl = ConstantList([1])
        with pytest.raises(TypeError, match="Cannot extend"):
            cl.extend([2, 3])

    def test_insert_raises(self):
        cl = ConstantList([1])
        with pytest.raises(TypeError, match="Cannot insert"):
            cl.insert(0, 0)

    def test_pop_raises(self):
        cl = ConstantList([1])
        with pytest.raises(TypeError, match="Cannot pop"):
            cl.pop()

    def test_remove_raises(self):
        cl = ConstantList([1])
        with pytest.raises(TypeError, match="Cannot remove"):
            cl.remove(1)

    def test_clear_raises(self):
        cl = ConstantList([1])
        with pytest.raises(TypeError, match="Cannot clear"):
            cl.clear()

    def test_setitem_raises(self):
        cl = ConstantList([1, 2])
        with pytest.raises(TypeError, match="Cannot set"):
            cl[0] = 99

    def test_delitem_raises(self):
        cl = ConstantList([1, 2])
        with pytest.raises(TypeError, match="Cannot delete"):
            del cl[0]

    def test_repr(self):
        cl = ConstantList([1, 2])
        assert repr(cl) == "ConstantList([1, 2])"

    def test_empty_list(self):
        cl = ConstantList([])
        assert len(cl) == 0
        assert list(cl) == []


class TestIsListOf:
    def test_list_of_ints(self):
        assert is_list_of([1, 2, 3], int) is True

    def test_list_of_mixed_first(self):
        assert is_list_of([1, "a", 3], int, check="first") is True

    def test_list_of_mixed_all(self):
        assert is_list_of([1, "a", 3], int, check="all") is False

    def test_empty_list(self):
        assert is_list_of([], int) is True

    def test_not_a_list(self):
        assert is_list_of("hello", str) is False
        assert is_list_of((1, 2), int) is False

    def test_tuple_of_types(self):
        assert is_list_of([1, 2.0], (int, float)) is True

    def test_wrong_type(self):
        assert is_list_of(["a", "b"], int, check="first") is False


class TestChunkList:
    def test_even_chunks(self):
        result = list(chunk_list([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_uneven_chunks(self):
        result = list(chunk_list([1, 2, 3, 4, 5], 2))
        assert result == [[1, 2], [3, 4], [5]]

    def test_single_chunk(self):
        result = list(chunk_list([1, 2, 3], 10))
        assert result == [[1, 2, 3]]

    def test_empty_list(self):
        result = list(chunk_list([], 5))
        assert result == []

    def test_chunk_size_one(self):
        result = list(chunk_list([1, 2, 3], 1))
        assert result == [[1], [2], [3]]


class TestCdiv:
    def test_exact_division(self):
        assert cdiv(6, 3) == 2

    def test_ceiling(self):
        assert cdiv(7, 3) == 3

    def test_one(self):
        assert cdiv(1, 1) == 1

    def test_larger_divisor(self):
        assert cdiv(3, 5) == 1

    def test_zero_dividend(self):
        assert cdiv(0, 5) == 0


class TestNextPowerOf2:
    def test_exact_power(self):
        assert next_power_of_2(8) == 8

    def test_non_power(self):
        assert next_power_of_2(5) == 8

    def test_one(self):
        assert next_power_of_2(1) == 1

    def test_zero(self):
        assert next_power_of_2(0) == 1

    def test_negative(self):
        assert next_power_of_2(-5) == 1

    def test_large_value(self):
        assert next_power_of_2(1000) == 1024


class TestPrevPowerOf2:
    def test_exact_power(self):
        assert prev_power_of_2(8) == 8

    def test_non_power(self):
        assert prev_power_of_2(5) == 4

    def test_one(self):
        assert prev_power_of_2(1) == 1

    def test_zero(self):
        assert prev_power_of_2(0) == 0

    def test_negative(self):
        assert prev_power_of_2(-5) == 0

    def test_large_value(self):
        assert prev_power_of_2(1000) == 512


class TestRoundUp:
    def test_exact(self):
        assert round_up(8, 4) == 8

    def test_round(self):
        assert round_up(7, 4) == 8

    def test_one_above(self):
        assert round_up(9, 4) == 12

    def test_zero(self):
        assert round_up(0, 4) == 0


class TestRoundDown:
    def test_exact(self):
        assert round_down(8, 4) == 8

    def test_round(self):
        assert round_down(7, 4) == 4

    def test_zero(self):
        assert round_down(0, 4) == 0


class TestGetDtypeSize:
    def test_float32(self):
        assert get_dtype_size(jnp.float32) == 4

    def test_float16(self):
        assert get_dtype_size(jnp.float16) == 2

    def test_bfloat16(self):
        assert get_dtype_size(jnp.bfloat16) == 2

    def test_int32(self):
        assert get_dtype_size(jnp.int32) == 4

    def test_int8(self):
        assert get_dtype_size(jnp.int8) == 1


class TestTruncateTokens:
    def test_no_truncation_needed(self):
        tokens, dropped = truncate_tokens([1, 2, 3], 5)
        assert tokens == [1, 2, 3]
        assert dropped == 0

    def test_exact_length(self):
        tokens, dropped = truncate_tokens([1, 2, 3], 3)
        assert tokens == [1, 2, 3]
        assert dropped == 0

    def test_left_truncation(self):
        tokens, dropped = truncate_tokens([1, 2, 3, 4, 5], 3, "left")
        assert tokens == [3, 4, 5]
        assert dropped == 2

    def test_right_truncation(self):
        tokens, dropped = truncate_tokens([1, 2, 3, 4, 5], 3, "right")
        assert tokens == [1, 2, 3]
        assert dropped == 2

    def test_middle_truncation(self):
        tokens, dropped = truncate_tokens([1, 2, 3, 4, 5], 3, "middle")
        assert tokens == [1, 2, 5]
        assert dropped == 2

    def test_middle_truncation_even(self):
        tokens, dropped = truncate_tokens([1, 2, 3, 4, 5, 6], 4, "middle")
        assert len(tokens) == 4
        assert dropped == 2
        assert tokens == [1, 2, 5, 6]

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown truncate_mode"):
            truncate_tokens([1, 2, 3], 2, "invalid")

    def test_truncate_to_one(self):
        tokens, dropped = truncate_tokens([1, 2, 3, 4, 5], 1, "left")
        assert tokens == [5]
        assert dropped == 4

    def test_empty_tokens(self):
        tokens, dropped = truncate_tokens([], 5)
        assert tokens == []
        assert dropped == 0


class TestRopeScalingUsesMrope:
    def test_none(self):
        assert _rope_scaling_uses_mrope(None) is False

    def test_with_mrope_section(self):
        assert _rope_scaling_uses_mrope({"mrope_section": [1, 2, 3]}) is True

    def test_with_mrope_type(self):
        assert _rope_scaling_uses_mrope({"rope_type": "mrope"}) is True

    def test_with_mrope_type_uppercase(self):
        assert _rope_scaling_uses_mrope({"rope_type": "MROPE"}) is True

    def test_with_mrope_interleaved(self):
        assert _rope_scaling_uses_mrope({"mrope_interleaved": True}) is True

    def test_with_type_key(self):
        assert _rope_scaling_uses_mrope({"type": "mrope"}) is True

    def test_empty_dict(self):
        assert _rope_scaling_uses_mrope({}) is False

    def test_non_mapping(self):
        assert _rope_scaling_uses_mrope("string") is False

    def test_with_to_dict_method(self):
        class FakeConfig:
            def to_dict(self):
                return {"mrope_section": [1, 2]}

        assert _rope_scaling_uses_mrope(FakeConfig()) is True


class TestGetTextConfig:
    def test_none_input(self):
        assert _get_text_config(None) is None

    def test_with_text_config_attr(self):
        class Config:
            text_config: typing.ClassVar = {"hidden_size": 768}

        assert _get_text_config(Config()) == {"hidden_size": 768}

    def test_with_get_text_config_method(self):
        class Config:
            def get_text_config(self):
                return {"hidden_size": 768}

        assert _get_text_config(Config()) == {"hidden_size": 768}

    def test_with_get_text_config_decoder(self):
        class Config:
            def get_text_config(self, decoder=False):
                return {"hidden_size": 768, "decoder": decoder}

        result = _get_text_config(Config())
        assert result["hidden_size"] == 768

    def test_no_text_config(self):
        class Config:
            pass

        result = _get_text_config(Config())
        assert isinstance(result, Config)


class TestModelUsesMrope:
    def test_model_without_config(self):
        class Model:
            pass

        assert model_uses_mrope(Model()) is False

    def test_model_with_mrope_scaling(self):
        class TextConfig:
            rope_scaling: typing.ClassVar = {"mrope_section": [1, 2]}

        class Config:
            text_config = TextConfig()

            def get_text_config(self):
                return self.text_config

        class Model:
            config = Config()

        assert model_uses_mrope(Model()) is True

    def test_model_with_legacy_flag(self):
        class Model:
            config = None
            _uses_mrope = True

        assert model_uses_mrope(Model()) is True

    def test_model_without_mrope(self):
        class TextConfig:
            rope_scaling: typing.ClassVar = {"type": "linear"}

        class Config:
            text_config = TextConfig()

            def get_text_config(self):
                return self.text_config

        class Model:
            config = Config()

        assert model_uses_mrope(Model()) is False
