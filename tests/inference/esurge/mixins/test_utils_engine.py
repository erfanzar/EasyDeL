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

"""Tests for EngineUtilsMixin static helpers and instance methods."""

from easydel.inference.esurge.mixins.utils import EngineUtilsMixin
from easydel.inference.sampling_params import SamplingParams


class TestCoerceMappingLike:
    def test_plain_string_unchanged(self):
        assert EngineUtilsMixin._coerce_mapping_like("hello") == "hello"

    def test_json_dict_parsed(self):
        result = EngineUtilsMixin._coerce_mapping_like('{"a": 1}')
        assert result == {"a": 1}

    def test_json_list_parsed(self):
        result = EngineUtilsMixin._coerce_mapping_like("[1, 2]")
        assert result == [1, 2]

    def test_invalid_json_returns_original(self):
        assert EngineUtilsMixin._coerce_mapping_like("{invalid") == "{invalid"

    def test_non_string_passthrough(self):
        assert EngineUtilsMixin._coerce_mapping_like(42) == 42
        assert EngineUtilsMixin._coerce_mapping_like(None) is None
        assert EngineUtilsMixin._coerce_mapping_like({"a": 1}) == {"a": 1}


class TestNormalizeChatTemplateMessages:
    def test_none_content_replaced_with_empty(self):
        msgs = [{"role": "user", "content": None}]
        result = EngineUtilsMixin._normalize_chat_template_messages(msgs)
        assert result[0]["content"] == ""

    def test_tool_calls_normalized(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {"name": "foo", "arguments": '{"x": 1}'},
                    }
                ],
            }
        ]
        result = EngineUtilsMixin._normalize_chat_template_messages(msgs)
        args = result[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict)
        assert args == {"x": 1}

    def test_tool_calls_none_arguments_default_to_dict(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "foo", "arguments": None}}],
            }
        ]
        result = EngineUtilsMixin._normalize_chat_template_messages(msgs)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {}

    def test_non_dict_arguments_wrapped(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "foo", "arguments": 42}}],
            }
        ]
        result = EngineUtilsMixin._normalize_chat_template_messages(msgs)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {"value": "42"}

    def test_non_dict_tool_calls_skipped(self):
        msgs = [{"role": "assistant", "content": "", "tool_calls": ["not_a_dict", 42]}]
        result = EngineUtilsMixin._normalize_chat_template_messages(msgs)
        assert result[0]["tool_calls"] == []

    def test_function_call_arguments_coerced(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "function_call": {"name": "bar", "arguments": '{"y": 2}'},
            }
        ]
        result = EngineUtilsMixin._normalize_chat_template_messages(msgs)
        assert result[0]["function_call"]["arguments"] == {"y": 2}

    def test_original_messages_not_mutated(self):
        original = {"role": "user", "content": None}
        msgs = [original]
        EngineUtilsMixin._normalize_chat_template_messages(msgs)
        assert original["content"] is None  # original unchanged


class TestCollapseSystemMessages:
    def test_single_system_unchanged(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        result = EngineUtilsMixin._collapse_system_messages(msgs)
        assert len(result) == 2
        assert result[0]["content"] == "sys"

    def test_multiple_systems_merged(self):
        msgs = [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "second"},
        ]
        result = EngineUtilsMixin._collapse_system_messages(msgs)
        assert result[0]["role"] == "system"
        assert "first" in result[0]["content"]
        assert "second" in result[0]["content"]
        assert len([m for m in result if m["role"] == "system"]) == 1

    def test_empty_messages(self):
        assert EngineUtilsMixin._collapse_system_messages([]) == []

    def test_no_system_messages(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = EngineUtilsMixin._collapse_system_messages(msgs)
        assert result == msgs


class TestNormalizeChatTemplateTools:
    def test_none_tools_returns_none(self):
        assert EngineUtilsMixin._normalize_chat_template_tools(None) is None

    def test_empty_list_returns_none(self):
        assert EngineUtilsMixin._normalize_chat_template_tools([]) is None

    def test_non_dict_entries_skipped(self):
        tools = ["not_a_dict", 42, None]
        assert EngineUtilsMixin._normalize_chat_template_tools(tools) is None

    def test_valid_tool_normalized(self):
        tools = [{"name": "foo", "parameters": {"type": "object", "properties": {}}}]
        result = EngineUtilsMixin._normalize_chat_template_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "foo"

    def test_function_wrapper_unwrapped(self):
        tools = [{"function": {"name": "bar", "parameters": {"type": "object"}}}]
        result = EngineUtilsMixin._normalize_chat_template_tools(tools)
        assert result[0]["name"] == "bar"

    def test_string_parameters_parsed(self):
        tools = [{"name": "foo", "parameters": '{"type": "object"}'}]
        result = EngineUtilsMixin._normalize_chat_template_tools(tools)
        assert isinstance(result[0]["parameters"], dict)

    def test_invalid_string_parameters_default_to_empty(self):
        tools = [{"name": "foo", "parameters": "not json"}]
        result = EngineUtilsMixin._normalize_chat_template_tools(tools)
        assert result[0]["parameters"] == {}

    def test_string_properties_parsed(self):
        tools = [{"name": "foo", "parameters": {"properties": '{"a": {"type": "string"}}'}}]
        result = EngineUtilsMixin._normalize_chat_template_tools(tools)
        assert isinstance(result[0]["parameters"]["properties"], dict)

    def test_string_required_wrapped_in_list(self):
        tools = [{"name": "foo", "parameters": {"required": "param1"}}]
        result = EngineUtilsMixin._normalize_chat_template_tools(tools)
        assert result[0]["parameters"]["required"] == ["param1"]

    def test_non_list_required_defaults_to_empty(self):
        tools = [{"name": "foo", "parameters": {"required": 42}}]
        result = EngineUtilsMixin._normalize_chat_template_tools(tools)
        assert result[0]["parameters"]["required"] == []

    def test_empty_name_skipped(self):
        tools = [{"name": ""}, {"name": "valid"}]
        result = EngineUtilsMixin._normalize_chat_template_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "valid"

    def test_whitespace_only_name_skipped(self):
        tools = [{"name": "   "}]
        assert EngineUtilsMixin._normalize_chat_template_tools(tools) is None

    def test_non_string_description_coerced(self):
        tools = [{"name": "foo", "description": 42}]
        result = EngineUtilsMixin._normalize_chat_template_tools(tools)
        assert result[0]["description"] == "42"

    def test_missing_parameters_default_to_empty(self):
        tools = [{"name": "foo"}]
        result = EngineUtilsMixin._normalize_chat_template_tools(tools)
        assert result[0]["parameters"] == {}


class TestNormalizeStopSequences:
    def test_none_returns_empty(self):
        assert EngineUtilsMixin._normalize_stop_sequences(None) == []

    def test_single_string(self):
        assert EngineUtilsMixin._normalize_stop_sequences("stop") == ["stop"]

    def test_list_of_strings(self):
        result = EngineUtilsMixin._normalize_stop_sequences(["a", "b"])
        assert result == ["a", "b"]

    def test_deduplication(self):
        result = EngineUtilsMixin._normalize_stop_sequences(["a", "b", "a"])
        assert result == ["a", "b"]

    def test_empty_strings_filtered(self):
        result = EngineUtilsMixin._normalize_stop_sequences(["", "a", ""])
        assert result == ["a"]

    def test_none_entries_filtered(self):
        result = EngineUtilsMixin._normalize_stop_sequences(["a", None, "b"])
        assert result == ["a", "b"]

    def test_non_string_coerced(self):
        result = EngineUtilsMixin._normalize_stop_sequences([42])
        assert result == ["42"]

    def test_set_input(self):
        result = EngineUtilsMixin._normalize_stop_sequences({"a", "b"})
        assert set(result) == {"a", "b"}

    def test_non_iterable_wrapped(self):
        result = EngineUtilsMixin._normalize_stop_sequences(42)
        assert result == ["42"]


class TestToStructuredTextMessages:
    def test_string_content_becomes_text_part(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = EngineUtilsMixin._to_structured_text_messages(msgs)
        assert result[0]["content"] == [{"type": "text", "text": "hello"}]

    def test_none_content_becomes_empty_list(self):
        msgs = [{"role": "user", "content": None}]
        result = EngineUtilsMixin._to_structured_text_messages(msgs)
        assert result[0]["content"] == []

    def test_dict_content_wrapped_in_list(self):
        msgs = [{"role": "user", "content": {"type": "text", "text": "hi"}}]
        result = EngineUtilsMixin._to_structured_text_messages(msgs)
        assert result[0]["content"] == [{"type": "text", "text": "hi"}]

    def test_list_content_strings_converted(self):
        msgs = [{"role": "user", "content": ["hello", "world"]}]
        result = EngineUtilsMixin._to_structured_text_messages(msgs)
        assert result[0]["content"] == [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]

    def test_input_text_type_converted(self):
        msgs = [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
        result = EngineUtilsMixin._to_structured_text_messages(msgs)
        assert result[0]["content"] == [{"type": "text", "text": "hi"}]

    def test_non_text_type_preserved(self):
        part = {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}}
        msgs = [{"role": "user", "content": [part]}]
        result = EngineUtilsMixin._to_structured_text_messages(msgs)
        assert result[0]["content"] == [part]


class TestComputeSnapshotDeltaText:
    def test_prefix_match(self):
        assert EngineUtilsMixin._compute_snapshot_delta_text("Hello World", "Hello ", "World") == "World"

    def test_empty_previous(self):
        assert EngineUtilsMixin._compute_snapshot_delta_text("Hello", "", "Hello") == "Hello"

    def test_no_change(self):
        assert EngineUtilsMixin._compute_snapshot_delta_text("Hello", "Hello", "") == ""

    def test_rollback_returns_empty(self):
        assert EngineUtilsMixin._compute_snapshot_delta_text("", "Hello", "") == ""

    def test_suffix_overlap_recovery(self):
        result = EngineUtilsMixin._compute_snapshot_delta_text("abcd", "xxab", "")
        assert result == "cd"

    def test_shorter_current_with_fallback(self):
        result = EngineUtilsMixin._compute_snapshot_delta_text("Hi", "Hello World", "Hi")
        assert result == "Hi"

    def test_none_handling(self):
        assert EngineUtilsMixin._compute_snapshot_delta_text(None, None, None) == ""


class _DummyTokenizer:
    def get_vocab(self):
        return {
            "<tool_call>": 100,
            "<tool_response>": 101,
            "hello": 102,
        }


class _NoVocabTokenizer:
    pass


class _DummyEngine(EngineUtilsMixin):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer


class TestPrepareChatSamplingParams:
    def test_no_tools_suppresses_tool_tokens(self):
        engine = _DummyEngine(_DummyTokenizer())
        params = engine._prepare_chat_sampling_params(None, tools=None)
        assert params.skip_special_tokens is True
        assert params.logit_bias is not None
        assert 100 in params.logit_bias
        assert 101 in params.logit_bias
        assert 102 not in params.logit_bias  # "hello" doesn't match tool patterns

    def test_with_tools_preserves_tokens(self):
        engine = _DummyEngine(_DummyTokenizer())
        tools = [{"name": "test"}]
        params = engine._prepare_chat_sampling_params(SamplingParams(), tools=tools)
        assert params.skip_special_tokens is False
        assert params.logit_bias is None

    def test_none_sampling_params_creates_default(self):
        engine = _DummyEngine(_DummyTokenizer())
        params = engine._prepare_chat_sampling_params(None)
        assert isinstance(params, SamplingParams)

    def test_no_vocab_method_still_works(self):
        engine = _DummyEngine(_NoVocabTokenizer())
        params = engine._prepare_chat_sampling_params(None)
        assert params.skip_special_tokens is True
        assert params.logit_bias is None or params.logit_bias == {}


class TestApplyExtraStops:
    def test_no_extra_stops(self):
        engine = _DummyEngine()
        params = SamplingParams(stop=["end"])
        result = engine._apply_extra_stops_to_sampling_params(params)
        assert result.stop == ["end"]

    def test_extra_stops_merged(self):
        engine = _DummyEngine()
        engine.extra_stops = ["<|stop|>", "<|end|>"]
        params = SamplingParams(stop=["end"])
        result = engine._apply_extra_stops_to_sampling_params(params)
        assert "<|stop|>" in result.stop
        assert "<|end|>" in result.stop
        assert "end" in result.stop

    def test_deduplication(self):
        engine = _DummyEngine()
        engine.extra_stops = ["end", "extra"]
        params = SamplingParams(stop=["end"])
        result = engine._apply_extra_stops_to_sampling_params(params)
        assert result.stop.count("end") == 1


class TestContentToTextParts:
    def test_none_returns_empty(self):
        assert EngineUtilsMixin._content_to_text_parts(None) == []

    def test_string_returns_text_part(self):
        result = EngineUtilsMixin._content_to_text_parts("hello")
        assert result == [{"type": "text", "text": "hello"}]

    def test_list_of_dicts(self):
        parts = [{"type": "text", "text": "hi"}]
        result = EngineUtilsMixin._content_to_text_parts(parts)
        assert result == [{"type": "text", "text": "hi"}]

    def test_list_with_non_dict_items(self):
        parts = ["hello", None, 42]
        result = EngineUtilsMixin._content_to_text_parts(parts)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "hello"}
        assert result[1] == {"type": "text", "text": "42"}

    def test_deep_copy_of_dicts(self):
        original = {"type": "text", "text": "hi"}
        result = EngineUtilsMixin._content_to_text_parts([original])
        result[0]["text"] = "modified"
        assert original["text"] == "hi"
