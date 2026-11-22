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

"""Regression tests for the Hermes scratch-pad handling, Llama4 pythonic gate and plugin loading."""

from __future__ import annotations

import json

import pytest
import regex
from easydel.inference.tools.abstract_tool import ToolParserManager
from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser
from easydel.inference.tools.parsers.llama4_pythonic_tool_parser import Llama4PythonicToolParser

from ._helpers import DummyTokenizer, make_request, stream

pytestmark = pytest.mark.cpu_ok


def _hermes() -> HermesToolParser:
    return HermesToolParser(DummyTokenizer({"<tool_call>": 1, "</tool_call>": 2}))


def test_hermes_batch_ignores_tool_calls_inside_scratch_pad():
    output = (
        '<scratch_pad>maybe <tool_call>{"name": "a", "arguments": {}}</tool_call></scratch_pad>'
        'real <tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    )
    result = _hermes().extract_tool_calls(output, make_request())
    assert result.tools_called is True
    assert [tc.function.name for tc in result.tool_calls] == ["b"]
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": 1}


def test_hermes_batch_open_scratch_pad_does_not_execute_example_call():
    output = '<scratch_pad>maybe <tool_call>{"name":"a","arguments":{}}</tool_call>'
    result = _hermes().extract_tool_calls(output, make_request())
    assert result.tools_called is False
    assert result.content == output


def test_hermes_batch_without_scratch_pad_is_unchanged():
    output = 'hi <tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    result = _hermes().extract_tool_calls(output, make_request())
    assert result.tools_called is True
    assert result.content == "hi "
    assert result.tool_calls[0].function.name == "b"


def test_hermes_streaming_does_not_emit_tool_calls_from_scratch_pad():
    chunks = [
        "<scratch_pad>think ",
        "<tool_call>",
        '{"name": "a", "arguments": {}}',
        "</tool_call>",
        "</scratch_pad>",
        "done",
    ]
    deltas = stream(_hermes(), chunks, make_request())
    assert not any(d.tool_calls for d in deltas)
    assert "done" in "".join(d.content or "" for d in deltas)


def test_hermes_streams_arguments_before_closing_tag_and_flushes_latest_values():
    parser = _hermes()
    chunks = [
        "<tool_call>",
        '{"name":"search", "arguments":{"q":"Par',
        "is",
        '","n":1',
        'e-2,"ids":[1,2]}}</tool_call>',
    ]
    initial = stream(parser, chunks[:-1], make_request())
    fragments = [tc.function.arguments or "" for d in initial for tc in d.tool_calls or [] if tc.function]
    assert "".join(fragments)  # Actual argument deltas, before any closing tag.
    previous = "".join(chunks[:-1])
    final = parser.extract_tool_calls_streaming(previous, previous + chunks[-1], chunks[-1], [], [], [], make_request())
    assert final is not None
    fragments += [tc.function.arguments or "" for tc in final.tool_calls or [] if tc.function]
    assert json.loads("".join(fragments)) == {"q": "Paris", "n": 0.01, "ids": [1, 2]}


@pytest.mark.parametrize(
    "chunks",
    [
        ["<tool_call>", '{"name":"a","arguments":{}}</tool_call>'],
        ['<tool_call>{"name":"a","arguments":{}}</tool_call>'],
        list('<tool_call>{"name":"a","arguments":{}}</tool_call>'),
    ],
)
def test_hermes_name_and_empty_arguments_are_not_lost_on_closing_chunk(chunks):
    calls = [tc for delta in stream(_hermes(), chunks, make_request()) for tc in delta.tool_calls or []]
    assert [tc.function.name for tc in calls if tc.function and tc.function.name] == ["a"]
    assert json.loads("".join(tc.function.arguments or "" for tc in calls if tc.function)) == {}


@pytest.mark.parametrize("arguments", [{}, {"q": "Paris"}])
def test_hermes_closing_chunk_can_carry_all_arguments(arguments):
    deltas = stream(
        _hermes(),
        ["<tool_call>", '{"name":"search",', '"arguments":' + json.dumps(arguments) + "}</tool_call>"],
        make_request(),
    )
    calls = [tc for d in deltas for tc in d.tool_calls or []]
    assert [tc.function.name for tc in calls if tc.function and tc.function.name] == ["search"]
    assert json.loads("".join(tc.function.arguments or "" for tc in calls if tc.function)) == arguments


def _llama4() -> Llama4PythonicToolParser:
    return Llama4PythonicToolParser(DummyTokenizer())


def test_llama4_pythonic_extracts_calls_with_and_without_delimiters():
    for output in ('[get_weather(city="NYC")]', '<|python_start|>[get_weather(city="NYC")]<|python_end|>'):
        result = _llama4().extract_tool_calls(output, make_request())
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "NYC"}


@pytest.mark.parametrize("output", ["plain answer", "[Note] this is prose", "[]", "[1, 2]"])
def test_llama4_pythonic_treats_non_calls_as_content(output):
    result = _llama4().extract_tool_calls(output, make_request())
    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == output


def test_llama4_pythonic_adversarial_input_returns_promptly():
    output = "[" + "a(b=1," * 500 + "]"
    result = _llama4().extract_tool_calls(output, make_request())
    assert result.tools_called is False
    assert result.content == output


def test_llama4_pythonic_regex_timeout_returns_content():
    class BoundedParser(Llama4PythonicToolParser):
        TOOL_CALL_REGEX = regex.compile(r"\[(a+)+\]")

    output = "[" + "a" * 10000 + "!]"
    parser = BoundedParser(DummyTokenizer(), regex_timeout=0.001)
    result = parser.extract_tool_calls(output, make_request())
    assert result.tools_called is False
    assert result.content == output


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_llama4_pythonic_rejects_invalid_regex_timeout(timeout):
    with pytest.raises(ValueError, match="regex_timeout"):
        Llama4PythonicToolParser(DummyTokenizer(), regex_timeout=timeout)


def test_import_tool_parser_registers_plugin_and_survives_broken_plugin(tmp_path):
    plugin = tmp_path / "hermes_llama4_tool_plugin.py"
    plugin.write_text(
        "from easydel.inference.tools.abstract_tool import ToolParser, ToolParserManager\n"
        "\n"
        "@ToolParserManager.register_module('hermes_llama4_tool_plugin')\n"
        "class _P(ToolParser):\n"
        "    def extract_tool_calls(self, model_output, request):\n"
        "        raise NotImplementedError\n"
    )
    ToolParserManager.import_tool_parser(str(plugin))
    assert ToolParserManager.get_tool_parser("hermes_llama4_tool_plugin").__name__ == "_P"

    broken = tmp_path / "hermes_llama4_broken_plugin.py"
    broken.write_text("raise RuntimeError('boom')\n")
    ToolParserManager.import_tool_parser(str(broken))  # logged, not raised
