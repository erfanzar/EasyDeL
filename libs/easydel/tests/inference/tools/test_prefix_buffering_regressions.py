# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prefix buffering and scalar JSON regressions (no array computation)."""

import json

import pytest
from easydel.inference.parsing.delegating_parser import DelegatingParser
from easydel.inference.tools.parsers.gigachat3_tool_parser import GigaChat3ToolParser
from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser
from easydel.inference.tools.tool_calling_mixin import ToolCallingMixin

from ._helpers import DummyTokenizer, function_tool, make_request, stream

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("prefix", ["Here: ", "intro ", ""])
@pytest.mark.parametrize("separator", ["", "\n", "<|role_sep|>", "<|role_sep|>\n"])
def test_gigachat_prefixed_header_is_hidden_at_every_chunk_boundary(prefix, separator):
    output = prefix + "function call" + separator + '{"name":"f","arguments":{}}'
    chunkings = [[output], list(output)]
    chunkings += [[output[:split], output[split:]] for split in range(1, len(output))]
    chunkings.append([prefix + "function call", "<|role_sep|>", "\n", '{"name":"f","arguments":{}}'])
    for chunks in chunkings:
        parser = GigaChat3ToolParser(DummyTokenizer())
        deltas = stream(parser, chunks, make_request())
        assert "".join(delta.content or "" for delta in deltas) == prefix, chunks
        calls = [call for delta in deltas for call in delta.tool_calls or []]
        assert [call.function.name for call in calls if call.function.name] == ["f"], chunks
        assert json.loads("".join(call.function.arguments or "" for call in calls)) == {}, chunks
        assert parser.flush_streaming_content() == ""


@pytest.mark.parametrize("prefix", ["", "intro ", " \n", " \nintro \t\n"])
@pytest.mark.parametrize("separator", ["", "\n", "<|role_sep|>", "<|role_sep|>\n"])
def test_gigachat_batch_preserves_ordinary_prefix_contract(prefix, separator):
    output = prefix + "function call" + separator + '{"name":"f","arguments":{}}'
    batch = GigaChat3ToolParser(DummyTokenizer()).extract_tool_calls(output, make_request())
    assert batch.tools_called
    assert batch.content == (prefix.rstrip() if prefix.strip() else None)
    assert [call.function.name for call in batch.tool_calls] == ["f"]
    assert json.loads(batch.tool_calls[0].function.arguments) == {}
    for chunks in ([output], list(output)):
        deltas = stream(GigaChat3ToolParser(DummyTokenizer()), chunks, make_request())
        assert "".join(delta.content or "" for delta in deltas) == prefix


@pytest.mark.parametrize("chunking", ["single", "characters", "split_header", "two_chunks"])
@pytest.mark.parametrize("separator", ["", "\n", "<|role_sep|>", "<|role_sep|>\n"])
def test_gigachat_orchestration_does_not_expose_prefixed_protocol(separator, chunking):
    output = "intro function call" + separator + '{"name":"f","arguments":{}}'
    if chunking == "single":
        chunkings = [[output]]
    elif chunking == "characters":
        chunkings = [list(output)]
    elif chunking == "split_header":
        chunkings = [["intro function call", separator, '{"name":"f","arguments":{}}']]
        if separator == "<|role_sep|>\n":
            chunkings.append(["intro function call", "<|role_sep|>", "\n", '{"name":"f","arguments":{}}'])
    else:
        chunkings = [[output[:split], output[split:]] for split in range(1, len(output))]
    for chunks in chunkings:
        _assert_gigachat_orchestration_preserves_prefix(chunks)


def _assert_gigachat_orchestration_preserves_prefix(chunks):
    """Check public streaming/final results without accepting content loss as nonleakage."""
    request = make_request([function_tool("f", {})])
    mixin = ToolCallingMixin()
    mixin.tool_parsers = {"dummy": GigaChat3ToolParser(DummyTokenizer())}
    delegating = DelegatingParser(tool_parser=GigaChat3ToolParser(DummyTokenizer()), tool_request=request)
    previous = ""
    visible = []
    delegated_visible = []
    for index, chunk in enumerate(chunks):
        current = previous + chunk
        delta = mixin.extract_tool_calls_streaming(
            "dummy", previous, current, chunk, request=request, finished=index == len(chunks) - 1
        )
        visible.append(delta.content or "" if delta else "")
        result = delegating.process_delta(current, chunk, [], previous, [])
        delegated_visible.append(result.delta_content or "")
        previous = current
    final = delegating.process_final(previous, [])
    delegated_visible.append(final.delta_content or "")
    assert "".join(visible) == "intro ", chunks
    assert "".join(delegated_visible) == "intro ", chunks
    assert final.accumulated_content == "intro ", chunks
    assert [call.function.name for call in final.tool_calls] == ["f"], chunks
    assert json.loads(final.tool_calls[0].function.arguments) == {}, chunks


@pytest.mark.parametrize("character_chunks", [False, True])
@pytest.mark.parametrize(
    "text", ["intro function call is a phrase", "intro function call<|role_", "intro function call\n"]
)
def test_gigachat_prefixed_false_or_unfinished_header_preserves_content(text, character_chunks):
    parser = GigaChat3ToolParser(DummyTokenizer())
    chunks = list(text) if character_chunks else [text]
    deltas = stream(parser, chunks, make_request())
    assert "".join(delta.content or "" for delta in deltas) + parser.flush_streaming_content() == text
    assert parser.flush_streaming_content() == ""
    assert not any(delta.tool_calls for delta in deltas)


@pytest.mark.parametrize(
    "arguments", ["1e-2", "1e+20", "-2.5e-3", "123", "-12", "0.001", "true", "null", '"text"', "{}", "[]"]
)
def test_hermes_scalar_arguments_match_batch_at_every_chunk_boundary(arguments):
    output = '<tool_call>{"name":"f","arguments":' + arguments + "}</tool_call>"
    batch = HermesToolParser(DummyTokenizer()).extract_tool_calls(output, make_request())
    assert batch.tools_called
    expected = batch.tool_calls[0].function.arguments
    chunkings = [[output], list(output)]
    chunkings += [[output[:split], output[split:]] for split in range(1, len(output))]
    for chunks in chunkings:
        parser = HermesToolParser(DummyTokenizer())
        deltas = stream(parser, chunks, make_request())
        calls = [call for delta in deltas for call in delta.tool_calls or []]
        assert [call.function.name for call in calls if call.function.name] == ["f"], chunks
        assert "".join(call.function.arguments or "" for call in calls) == expected, chunks
        assert not any(delta.content for delta in deltas), chunks
