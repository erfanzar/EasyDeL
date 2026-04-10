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

"""Tests for eSurge server API helper functions and edge cases."""

from easydel.inference.esurge.mixins.utils import EngineUtilsMixin
from easydel.inference.sampling_params import SamplingParams


class TestMergeSystemContent:
    def test_two_strings(self):
        result = EngineUtilsMixin._merge_system_content("hello", "world")
        assert result == "hello\n\nworld"

    def test_empty_existing(self):
        result = EngineUtilsMixin._merge_system_content("", "world")
        assert result == "world"

    def test_empty_new(self):
        result = EngineUtilsMixin._merge_system_content("hello", "")
        assert result == "hello"

    def test_both_empty(self):
        result = EngineUtilsMixin._merge_system_content("", "")
        assert result == ""

    def test_none_existing(self):
        result = EngineUtilsMixin._merge_system_content(None, "world")
        assert result == "world"

    def test_none_new(self):
        result = EngineUtilsMixin._merge_system_content("hello", None)
        assert result == "hello"

    def test_both_none(self):
        result = EngineUtilsMixin._merge_system_content(None, None)
        assert result == ""

    def test_list_existing(self):
        result = EngineUtilsMixin._merge_system_content([{"type": "text", "text": "hi"}], "world")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_list_new(self):
        result = EngineUtilsMixin._merge_system_content("hello", [{"type": "text", "text": "world"}])
        assert isinstance(result, list)

    def test_both_lists(self):
        result = EngineUtilsMixin._merge_system_content(
            [{"type": "text", "text": "a"}],
            [{"type": "text", "text": "b"}],
        )
        assert isinstance(result, list)
        assert len(result) == 2


class TestPreparePromptSegments:
    def setup_method(self):
        self.engine = EngineUtilsMixin()

    def test_string_input(self):
        result = self.engine._prepare_prompt_segments("hello")
        assert result == ["hello"]

    def test_list_input(self):
        result = self.engine._prepare_prompt_segments(["hello", "world"])
        assert result == ["hello", "world"]

    def test_non_string_items_coerced(self):
        result = self.engine._prepare_prompt_segments([42, None, True])
        assert result == ["42", "None", "True"]

    def test_non_string_scalar(self):
        result = self.engine._prepare_prompt_segments(42)
        assert result == ["42"]


class TestFilterEosTokens:
    def setup_method(self):
        self.engine = EngineUtilsMixin()

    def test_no_eos_set(self):
        result = self.engine._filter_eos_tokens([1, 2, 3])
        assert result == [1, 2, 3]

    def test_with_eos_set(self):
        self.engine._eos_set = {2}
        result = self.engine._filter_eos_tokens([1, 2, 3, 2])
        assert result == [1, 3]

    def test_empty_eos_set(self):
        self.engine._eos_set = set()
        result = self.engine._filter_eos_tokens([1, 2, 3])
        assert result == [1, 2, 3]

    def test_all_eos(self):
        self.engine._eos_set = {1, 2, 3}
        result = self.engine._filter_eos_tokens([1, 2, 3])
        assert result == []

    def test_backward_compat_prefix(self):
        self.engine._eSurge__eos_set = {5}
        result = self.engine._filter_eos_tokens([1, 5, 3])
        assert result == [1, 3]


class TestToPythonScalar:
    def test_int_passthrough(self):
        assert EngineUtilsMixin._to_python_scalar(42) == 42

    def test_float_passthrough(self):
        assert EngineUtilsMixin._to_python_scalar(3.14) == 3.14

    def test_string_passthrough(self):
        assert EngineUtilsMixin._to_python_scalar("hello") == "hello"

    def test_none_passthrough(self):
        assert EngineUtilsMixin._to_python_scalar(None) is None

    def test_object_with_item_method(self):
        class FakeArray:
            def item(self):
                return 42

        assert EngineUtilsMixin._to_python_scalar(FakeArray()) == 42

    def test_item_method_fails_gracefully(self):
        class BadArray:
            def item(self):
                raise RuntimeError("no scalar")

        assert isinstance(EngineUtilsMixin._to_python_scalar(BadArray()), BadArray)


class TestSanitizeMetricsPayload:
    def test_basic_dict(self):
        engine = EngineUtilsMixin()
        result = engine._sanitize_metrics_payload({"a": 1, "b": 2.0})
        assert result == {"a": 1, "b": 2.0}

    def test_with_array_like(self):
        engine = EngineUtilsMixin()

        class FakeArray:
            def item(self):
                return 42

        result = engine._sanitize_metrics_payload({"val": FakeArray()})
        assert result == {"val": 42}

    def test_empty_dict(self):
        engine = EngineUtilsMixin()
        assert engine._sanitize_metrics_payload({}) == {}


class TestCloneSamplingParams:
    def test_clone_is_independent(self):
        engine = EngineUtilsMixin()
        original = SamplingParams(stop=["end"], temperature=0.5)
        cloned = engine._clone_sampling_params(original)
        cloned.stop.append("new")
        assert "new" not in original.stop

    def test_clone_preserves_values(self):
        engine = EngineUtilsMixin()
        original = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
        cloned = engine._clone_sampling_params(original)
        assert cloned.temperature == 0.7
        assert cloned.top_p == 0.9
        assert cloned.max_tokens == 100


class TestApplyGenerationConfig:
    def test_no_generation_config(self):
        engine = EngineUtilsMixin()
        params = SamplingParams()
        result = engine._apply_generation_config_to_sampling_params(params)
        assert result is params

    def test_with_generation_config(self):
        engine = EngineUtilsMixin()
        engine._generation_config_dict = {"eos_token_id": [50256]}
        engine._primary_eos_token_id = 50256
        params = SamplingParams()
        result = engine._apply_generation_config_to_sampling_params(params)
        assert 50256 in result.all_stop_token_ids

    def test_exception_handled_gracefully(self):
        engine = EngineUtilsMixin()
        engine._generation_config_dict = {"eos_token_id": "not_an_int"}
        engine._primary_eos_token_id = None
        params = SamplingParams()
        result = engine._apply_generation_config_to_sampling_params(params)
        assert result is params
