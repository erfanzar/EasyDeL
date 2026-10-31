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

"""Tests for eSurge engine-component helper functions and edge cases."""

from easydel.inference.esurge.engine.admission import RequestAdmission, clone_sampling_params
from easydel.inference.esurge.engine.chat_templating import merge_system_content
from easydel.inference.esurge.engine.output_pipeline import OutputPipeline
from easydel.inference.esurge.engine.registry import RequestRegistry
from easydel.inference.esurge.esurge_engine import eSurge
from easydel.inference.sampling_params import SamplingParams


def _make_admission(generation_config=None, primary_eos_token_id=None):
    """Build a RequestAdmission with inert fakes for helper tests."""
    return RequestAdmission(
        registry=RequestRegistry(),
        scheduler_submit=lambda requests: None,
        tokenizer_client=None,
        tokenizer=None,
        context_config=None,
        reserve_tokens=0,
        max_model_len=2048,
        tool_parser_class=None,
        reasoning_parser_class=None,
        sampling_params_callback=lambda: None,
        generation_config_dict=generation_config or {},
        primary_eos_token_id=primary_eos_token_id,
        eos_token_ids=[],
        extra_stops=[],
        ignore_stop_strings_in_reasoning=False,
        on_activity=lambda: None,
        info=lambda *args, **kwargs: None,
        callback_engine=None,
    )


def _make_pipeline(eos_token_ids=()):
    """Build an OutputPipeline with inert fakes for token-filter tests."""
    return OutputPipeline(
        registry=RequestRegistry(),
        detokenizer_client=None,
        eos_token_ids=list(eos_token_ids),
        decode_interval_tokens=1,
        decode_interval_secs=0.0,
        on_stop_strings=lambda stops: None,
        on_activity=lambda: None,
        on_fatal=lambda exc, tb: None,
    )


class _EngineStub(eSurge):
    """eSurge subclass exposing engine helpers without engine construction."""

    def __init__(self):
        pass


class TestMergeSystemContent:
    def test_two_strings(self):
        result = merge_system_content("hello", "world")
        assert result == "hello\n\nworld"

    def test_empty_existing(self):
        result = merge_system_content("", "world")
        assert result == "world"

    def test_empty_new(self):
        result = merge_system_content("hello", "")
        assert result == "hello"

    def test_both_empty(self):
        result = merge_system_content("", "")
        assert result == ""

    def test_none_existing(self):
        result = merge_system_content(None, "world")
        assert result == "world"

    def test_none_new(self):
        result = merge_system_content("hello", None)
        assert result == "hello"

    def test_both_none(self):
        result = merge_system_content(None, None)
        assert result == ""

    def test_list_existing(self):
        result = merge_system_content([{"type": "text", "text": "hi"}], "world")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_list_new(self):
        result = merge_system_content("hello", [{"type": "text", "text": "world"}])
        assert isinstance(result, list)

    def test_both_lists(self):
        result = merge_system_content(
            [{"type": "text", "text": "a"}],
            [{"type": "text", "text": "b"}],
        )
        assert isinstance(result, list)
        assert len(result) == 2


class TestPreparePromptSegments:
    def setup_method(self):
        self.admission = _make_admission()

    def test_string_input(self):
        result = self.admission._prepare_prompt_segments("hello")
        assert result == ["hello"]

    def test_list_input(self):
        result = self.admission._prepare_prompt_segments(["hello", "world"])
        assert result == ["hello", "world"]

    def test_non_string_items_coerced(self):
        result = self.admission._prepare_prompt_segments([42, None, True])
        assert result == ["42", "None", "True"]

    def test_non_string_scalar(self):
        result = self.admission._prepare_prompt_segments(42)
        assert result == ["42"]


class TestFilterEosTokens:
    def test_no_eos_set(self):
        pipeline = _make_pipeline()
        result = pipeline._filter_eos_tokens([1, 2, 3])
        assert result == [1, 2, 3]

    def test_with_eos_set(self):
        pipeline = _make_pipeline(eos_token_ids=[2])
        result = pipeline._filter_eos_tokens([1, 2, 3, 2])
        assert result == [1, 3]

    def test_empty_eos_set(self):
        pipeline = _make_pipeline(eos_token_ids=[])
        result = pipeline._filter_eos_tokens([1, 2, 3])
        assert result == [1, 2, 3]

    def test_all_eos(self):
        pipeline = _make_pipeline(eos_token_ids=[1, 2, 3])
        result = pipeline._filter_eos_tokens([1, 2, 3])
        assert result == []


class TestToPythonScalar:
    def test_int_passthrough(self):
        assert eSurge._to_python_scalar(42) == 42

    def test_float_passthrough(self):
        assert eSurge._to_python_scalar(3.14) == 3.14

    def test_string_passthrough(self):
        assert eSurge._to_python_scalar("hello") == "hello"

    def test_none_passthrough(self):
        assert eSurge._to_python_scalar(None) is None

    def test_object_with_item_method(self):
        class FakeArray:
            def item(self):
                return 42

        assert eSurge._to_python_scalar(FakeArray()) == 42

    def test_item_method_fails_gracefully(self):
        class BadArray:
            def item(self):
                raise RuntimeError("no scalar")

        assert isinstance(eSurge._to_python_scalar(BadArray()), BadArray)


class TestSanitizeMetricsPayload:
    def test_basic_dict(self):
        engine = _EngineStub()
        result = engine._sanitize_metrics_payload({"a": 1, "b": 2.0})
        assert result == {"a": 1, "b": 2.0}

    def test_with_array_like(self):
        engine = _EngineStub()

        class FakeArray:
            def item(self):
                return 42

        result = engine._sanitize_metrics_payload({"val": FakeArray()})
        assert result == {"val": 42}

    def test_empty_dict(self):
        engine = _EngineStub()
        assert engine._sanitize_metrics_payload({}) == {}


class TestCloneSamplingParams:
    def test_clone_is_independent(self):
        original = SamplingParams(stop=["end"], temperature=0.5)
        cloned = clone_sampling_params(original)
        cloned.stop.append("new")
        assert "new" not in original.stop

    def test_clone_preserves_values(self):
        original = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
        cloned = clone_sampling_params(original)
        assert cloned.temperature == 0.7
        assert cloned.top_p == 0.9
        assert cloned.max_tokens == 100


class TestApplyGenerationConfig:
    def test_no_generation_config(self):
        admission = _make_admission()
        params = SamplingParams()
        result = admission._apply_generation_config_to_sampling_params(params)
        assert result is params

    def test_with_generation_config(self):
        admission = _make_admission(
            generation_config={"eos_token_id": [50256]},
            primary_eos_token_id=50256,
        )
        params = SamplingParams()
        result = admission._apply_generation_config_to_sampling_params(params)
        assert 50256 in result.all_stop_token_ids

    def test_exception_handled_gracefully(self):
        admission = _make_admission(
            generation_config={"eos_token_id": "not_an_int"},
            primary_eos_token_id=None,
        )
        params = SamplingParams()
        result = admission._apply_generation_config_to_sampling_params(params)
        assert result is params
