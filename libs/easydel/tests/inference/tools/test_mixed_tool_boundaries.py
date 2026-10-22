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

"""Mixed tool boundaries and schema/markup literal regressions (no array computation)."""

import json

import pytest
from easydel.inference.tools.parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser
from easydel.inference.tools.parsers.granite_20b_fc_tool_parser import Granite20bFCToolParser
from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

from ._helpers import DummyTokenizer, function_tool, make_request, stream

pytestmark = pytest.mark.cpu_ok


def _calls(deltas):
    names, arguments = {}, {}
    for delta in deltas:
        for call in delta.tool_calls or []:
            if call.function.name:
                assert call.index not in names, "A tool name must be announced once, in full"
                names[call.index] = call.function.name
            arguments[call.index] = arguments.get(call.index, "") + (call.function.arguments or "")
    return names, {index: json.loads(value) for index, value in arguments.items()}


@pytest.mark.parametrize("first_args", [{}, {"x": 1}])
def test_granite_waits_for_complete_name_after_multi_tool_jump(first_args):
    chunks = [
        '<function_call>{"name":"a","arguments":' + json.dumps(first_args) + "}",
        '<function_call>{"name":"be',
        'ta","arguments":{}}',
    ]
    parser = Granite20bFCToolParser(DummyTokenizer())
    deltas = stream(parser, chunks[:2], make_request())
    assert [call.function.name for delta in deltas for call in delta.tool_calls or [] if call.function.name] == ["a"]
    previous = "".join(chunks[:2])
    final = parser.extract_tool_calls_streaming(previous, previous + chunks[2], chunks[2], [], [], [], make_request())
    assert final is not None
    names, arguments = _calls([*deltas, final])
    assert names == {0: "a", 1: "beta"}
    assert arguments == {0: first_args, 1: {}}


@pytest.mark.parametrize("schema_type", [["string", "null"], ["null", "string"]])
@pytest.mark.parametrize("string_flag, expected", [("false", None), ("true", "null")])
def test_dsml_nullable_string_respects_explicit_string_flag(schema_type, string_flag, expected):
    output = (
        '<｜DSML｜function_calls><｜DSML｜invoke name="f">'
        f'<｜DSML｜parameter name="s" string="{string_flag}">null</｜DSML｜parameter>'
        "</｜DSML｜invoke></｜DSML｜function_calls>"
    )
    request = make_request([function_tool("f", {"s": {"type": schema_type}})])
    batch = DeepSeekV32ToolParser(DummyTokenizer()).extract_tool_calls(output, request)
    assert batch.tools_called
    assert json.loads(batch.tool_calls[0].function.arguments) == {"s": expected}
    for chunks in ([output], list(output)):
        names, arguments = _calls(stream(DeepSeekV32ToolParser(DummyTokenizer()), chunks, request))
        assert names == {0: "f"}
        assert arguments == {0: {"s": expected}}


@pytest.mark.parametrize("next_body", ["", '{"name":"b","arguments":{}}'])
def test_hermes_flushes_closing_call_before_opening_next(next_body):
    chunks = [
        "<tool_call>",
        '{"name":"a","arguments":{"q":"Pa',
        'ris"}}</tool_call><tool_call>' + next_body,
    ]
    if not next_body:
        chunks.append('{"name":"b","arguments":{}}')
    chunks.append("</tool_call>")
    names, arguments = _calls(stream(HermesToolParser(DummyTokenizer()), chunks, make_request()))
    assert names == {0: "a", 1: "b"}
    assert arguments == {0: {"q": "Paris"}, 1: {}}


@pytest.mark.parametrize("argument", ["hello", "null", "", "雪", "<scratch_pad>literal</scratch_pad>"])
def test_hermes_string_arguments_use_same_json_serialization_until_close(argument):
    serialized = json.dumps(argument, ensure_ascii=False)
    chunks = ["<tool_call>", '{"name":"echo",', '"arguments":' + serialized[:-1], serialized[-1:] + "}", "</tool_call>"]
    parser = HermesToolParser(DummyTokenizer())
    names, arguments = _calls(stream(parser, chunks, make_request()))
    batch = HermesToolParser(DummyTokenizer()).extract_tool_calls("".join(chunks), make_request())
    assert names == {0: "echo"}
    assert arguments == {0: argument}
    assert json.loads(batch.tool_calls[0].function.arguments) == argument


@pytest.mark.parametrize(
    "literal",
    [
        "<scratch_pad>literal</scratch_pad>",
        "<scratch_pad>unfinished literal",
        '<scratch_pad><tool_call>{"name":"fake"}</tool_call></scratch_pad>',
        '<tool_call>quoted \\" text</tool_call>',
    ],
)
def test_hermes_scratch_pad_tags_in_argument_strings_are_not_reasoning(literal):
    arguments = {"text": literal}
    output = '<tool_call>{"name":"echo","arguments":' + json.dumps(arguments) + "}</tool_call>"
    batch = HermesToolParser(DummyTokenizer()).extract_tool_calls(output, make_request())
    assert batch.tools_called
    assert json.loads(batch.tool_calls[0].function.arguments) == arguments
    for chunks in ([output], list(output)):
        names, streamed = _calls(stream(HermesToolParser(DummyTokenizer()), chunks, make_request()))
        assert names == {0: "echo"}
        assert streamed == {0: arguments}


def test_hermes_scratch_reasoning_and_trailing_content_survive_character_chunks():
    reasoning = '<scratch_pad>try <tool_call>{"name":"fake","arguments":{}}</tool_call></scratch_pad>'
    output = reasoning + '<tool_call>{"name":"real","arguments":{}}</tool_call>done'
    deltas = stream(HermesToolParser(DummyTokenizer()), list(output), make_request())
    names, arguments = _calls(deltas)
    assert names == {0: "real"}
    assert arguments == {0: {}}
    assert "".join(delta.content or "" for delta in deltas) == reasoning + "done"
