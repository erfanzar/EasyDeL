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

"""Regression tests for the GLM-4.7, InternLM2 and Granite-20B-FC tool parsers (no array computation)."""

from __future__ import annotations

import json

import pytest
from easydel.inference.tools.parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser
from easydel.inference.tools.parsers.granite_20b_fc_tool_parser import Granite20bFCToolParser
from easydel.inference.tools.parsers.internlm2_tool_parser import Internlm2ToolParser

from ._helpers import DummyTokenizer, function_tool, make_request, stream

pytestmark = pytest.mark.cpu_ok


def _glm47() -> Glm47MoeModelToolParser:
    return Glm47MoeModelToolParser(DummyTokenizer({"<tool_call>": 1, "</tool_call>": 2}))


def test_glm47_zero_argument_call_is_returned():
    result = _glm47().extract_tool_calls("<tool_call>ping</tool_call>", make_request())
    assert result.tools_called is True
    assert [tc.function.name for tc in result.tool_calls] == ["ping"]
    assert json.loads(result.tool_calls[0].function.arguments) == {}


def test_glm47_zero_argument_call_is_streamed():
    deltas = stream(_glm47(), ["<tool_call>", "ping", "</tool_call>"], make_request())
    calls = [tc for delta in deltas for tc in delta.tool_calls or []]
    assert [tc.function.name for tc in calls] == ["ping"]
    assert json.loads(calls[0].function.arguments) == {}


def test_glm47_name_is_stripped_and_string_schema_keeps_string_values():
    request = make_request([function_tool("get_weather", {"zip": {"type": "string"}})])
    output = "<tool_call>get_weather\n<arg_key>zip</arg_key>\n<arg_value>12345</arg_value>\n</tool_call>"
    result = _glm47().extract_tool_calls(output, request)
    assert result.tools_called is True
    call = result.tool_calls[0].function
    assert call.name == "get_weather"
    # A name carrying a trailing newline would never match the schema and "12345" would become an int.
    assert json.loads(call.arguments) == {"zip": "12345"}


def _internlm() -> Internlm2ToolParser:
    return Internlm2ToolParser(DummyTokenizer())


def test_internlm2_second_action_marker_does_not_raise():
    request = make_request([function_tool("a", {}), function_tool("b", {})])
    output = (
        'hi<|action_start|><|plugin|>{"name": "a", "parameters": {"x": 1}}<|action_end|>'
        '<|action_start|><|plugin|>{"name": "b", "parameters": {}}<|action_end|>'
    )
    result = _internlm().extract_tool_calls(output, request)
    assert result.tools_called is True
    assert result.tool_calls[0].function.name == "a"
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": 1}


def test_internlm2_malformed_action_returns_content_instead_of_raising():
    request = make_request([function_tool("a", {})])
    output = "x<|action_start|><|plugin|>{not json"
    result = _internlm().extract_tool_calls(output, request)
    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == output


def test_internlm2_streaming_with_two_markers_in_one_chunk_does_not_raise():
    request = make_request([function_tool("a", {})])
    chunk = (
        '<|action_start|><|plugin|>{"name": "a", "parameters": {}}<|action_end|><|action_start|><|plugin|>{"name": "b"'
    )
    deltas = stream(_internlm(), [chunk], request)
    assert [tc.function.name for d in deltas for tc in d.tool_calls or [] if tc.function] == ["a"]


def test_granite_20b_fc_multi_object_jump_flushes_already_announced_empty_args():
    chunks = [
        '<function_call>{"name":"a",',
        '"arguments":{}}<function_call>{"name":"b","arguments":{}}',
    ]
    deltas = stream(Granite20bFCToolParser(DummyTokenizer()), chunks, make_request())
    names = {}
    arguments = {}
    for delta in deltas:
        for call in delta.tool_calls or []:
            if call.function.name:
                names[call.index] = call.function.name
            arguments[call.index] = arguments.get(call.index, "") + (call.function.arguments or "")
    assert names == {0: "a", 1: "b"}
    assert {index: json.loads(value) for index, value in arguments.items()} == {0: {}, 1: {}}


def test_granite_20b_fc_multi_object_chunk_announces_every_tool():
    parser = Granite20bFCToolParser(DummyTokenizer())
    request = make_request()
    full = (
        '<function_call>{"name": "a", "arguments": {"x": 1}}'
        '<function_call>{"name": "b", "arguments": {"y": 2}}'
        '<function_call>{"name": "c", "arguments": {"z": 3}}'
    )
    # All three complete calls must be delivered without an artificial extra tick.
    deltas = stream(parser, [full], request)
    names: dict[int, str] = {}
    args: dict[int, str] = {}
    for delta in deltas:
        for tool_call in delta.tool_calls or []:
            if tool_call.function is None:
                continue
            if tool_call.function.name:
                names[tool_call.index] = tool_call.function.name
            if tool_call.function.arguments:
                args[tool_call.index] = args.get(tool_call.index, "") + tool_call.function.arguments
    assert names == {0: "a", 1: "b", 2: "c"}
    assert json.loads(args[0]) == {"x": 1}
    assert json.loads(args[1]) == {"y": 2}
    assert json.loads(args[2]) == {"z": 3}
