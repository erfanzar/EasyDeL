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

"""Tests for ToolCallingMixin methods."""

import typing

from easydel.inference.tools.tool_calling_mixin import ToolCallingMixin


class _FakeToolCall:
    def __init__(self, name, arguments):
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class _FakeExtracted:
    def __init__(self, tools_called=False, tool_calls=None, content="text"):
        self.tools_called = tools_called
        self.tool_calls = tool_calls
        self.content = content


class _FakeToolParser:
    def __init__(self):
        self.extract_calls = []
        self.streaming_calls = []

    def extract_tool_calls(self, text, request):
        self.extract_calls.append((text, request))
        if "<tool_call>" in text:
            return _FakeExtracted(
                tools_called=True,
                tool_calls=[_FakeToolCall("test_fn", {"x": 1})],
                content="",
            )
        return _FakeExtracted(content=text)

    def extract_tool_calls_streaming(self, **kwargs):
        self.streaming_calls.append(kwargs)
        return None


class _FakeRequest:
    tools: typing.ClassVar = [{"name": "test_fn"}]
    tool_choice = "auto"


class _TestMixin(ToolCallingMixin):
    def __init__(self):
        self.tool_parsers = {}


class TestExtractToolCallsBatch:
    def test_no_parser_returns_stop(self):
        mixin = _TestMixin()
        msg, reason = mixin.extract_tool_calls_batch("hello", _FakeRequest(), "model_a")
        assert reason == "stop"
        assert msg.content == "hello"

    def test_with_parser_no_tool_calls(self):
        mixin = _TestMixin()
        mixin.tool_parsers["model_a"] = _FakeToolParser()
        msg, reason = mixin.extract_tool_calls_batch("hello", _FakeRequest(), "model_a")
        assert reason == "stop"
        assert msg.content == "hello"

    def test_with_parser_tool_calls_detected(self):
        mixin = _TestMixin()
        mixin.tool_parsers["model_a"] = _FakeToolParser()
        msg, reason = mixin.extract_tool_calls_batch('<tool_call>{"name": "test"}', _FakeRequest(), "model_a")
        assert reason == "tool_calls"
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_tools_called_true_but_empty_list(self):
        """Edge case: tools_called=True but tool_calls=[] (empty list is falsy)."""
        mixin = _TestMixin()

        class EmptyToolParser:
            def extract_tool_calls(self, text, request):
                return _FakeExtracted(tools_called=True, tool_calls=[], content="text")

        mixin.tool_parsers["model_a"] = EmptyToolParser()
        _msg, reason = mixin.extract_tool_calls_batch("text", _FakeRequest(), "model_a")
        assert reason == "stop"


class TestExtractToolCallsStreaming:
    def test_no_parser_returns_none(self):
        mixin = _TestMixin()
        result = mixin.extract_tool_calls_streaming("model_a", "", "hi", "hi")
        assert result is None

    def test_with_parser_none_delta(self):
        mixin = _TestMixin()
        mixin.tool_parsers["model_a"] = _FakeToolParser()
        result = mixin.extract_tool_calls_streaming("model_a", "", "hi", "hi")
        assert result is None

    def test_parser_exception_falls_back(self):
        mixin = _TestMixin()

        class FailingParser:
            def extract_tool_calls_streaming(self, **kwargs):
                raise RuntimeError("parse error")

        mixin.tool_parsers["model_a"] = FailingParser()
        result = mixin.extract_tool_calls_streaming("model_a", "", "hi", "hi")
        assert result is not None
        assert result.content == "hi"

    def test_string_delta_wrapped(self):
        mixin = _TestMixin()

        class StringParser:
            def extract_tool_calls_streaming(self, **kwargs):
                return "some content"

        mixin.tool_parsers["model_a"] = StringParser()
        result = mixin.extract_tool_calls_streaming("model_a", "", "hi", "hi")
        assert result.content == "some content"

    def test_dict_delta_converted(self):
        mixin = _TestMixin()

        class DictParser:
            def extract_tool_calls_streaming(self, **kwargs):
                return {"content": "dict_content"}

        mixin.tool_parsers["model_a"] = DictParser()
        result = mixin.extract_tool_calls_streaming("model_a", "", "hi", "hi")
        assert result.content == "dict_content"

    def test_none_token_ids_default_to_empty(self):
        mixin = _TestMixin()
        parser = _FakeToolParser()
        mixin.tool_parsers["model_a"] = parser
        mixin.extract_tool_calls_streaming(
            "model_a",
            "",
            "hi",
            "hi",
            previous_token_ids=None,
            current_token_ids=None,
            delta_token_ids=None,
        )
        call = parser.streaming_calls[0]
        assert call["previous_token_ids"] == []
        assert call["current_token_ids"] == []
        assert call["delta_token_ids"] == []


class TestGetToolParserForModel:
    def test_no_parsers_attr(self):
        mixin = ToolCallingMixin.__new__(ToolCallingMixin)
        result = mixin.get_tool_parser_for_model("model_a")
        assert result is None

    def test_model_not_found(self):
        mixin = _TestMixin()
        assert mixin.get_tool_parser_for_model("missing") is None

    def test_model_found(self):
        mixin = _TestMixin()
        parser = _FakeToolParser()
        mixin.tool_parsers["model_a"] = parser
        assert mixin.get_tool_parser_for_model("model_a") is parser


class TestCreateToolsResponse:
    def test_basic_response_structure(self):
        mixin = _TestMixin()
        result = mixin.create_tools_response(["model_a"])
        assert "models" in result
        assert "default_format" in result
        assert result["default_format"] == "openai"

    def test_model_without_parser(self):
        mixin = _TestMixin()
        result = mixin.create_tools_response(["model_a"])
        assert result["models"]["model_a"]["tool_parser"] is None

    def test_model_with_parser(self):
        mixin = _TestMixin()
        mixin.tool_parsers["model_a"] = _FakeToolParser()
        mixin.tool_parser_name = "hermes"
        result = mixin.create_tools_response(["model_a"])
        assert result["models"]["model_a"]["tool_parser"] == "hermes"

    def test_empty_model_list(self):
        mixin = _TestMixin()
        result = mixin.create_tools_response([])
        assert result["models"] == {}


class TestInitializeToolParsers:
    def test_disabled_returns_empty(self):
        mixin = _TestMixin()
        result = mixin.initialize_tool_parsers({}, "hermes", enable_function_calling=False)
        assert result == {}

    def test_invalid_parser_name_skipped(self):
        mixin = _TestMixin()

        class FakeProcessor:
            pass

        result = mixin.initialize_tool_parsers(
            {"model_a": FakeProcessor()},
            "completely_invalid_parser_xyz",
            enable_function_calling=True,
        )
        assert "model_a" not in result

    def test_none_parser_name_from_dict(self):
        mixin = _TestMixin()
        result = mixin.initialize_tool_parsers(
            {"model_a": object()},
            {"model_b": "hermes"},  # model_a not in dict
            enable_function_calling=True,
        )
        assert "model_a" not in result

    def test_empty_string_parser_name_skipped(self):
        mixin = _TestMixin()
        result = mixin.initialize_tool_parsers(
            {"model_a": object()},
            "",
            enable_function_calling=True,
        )
        assert result == {}
