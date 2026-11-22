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

"""Hermes content-boundary regressions (no array computation)."""

import pytest
from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser
from easydel.inference.tools.tool_calling_mixin import ToolCallingMixin

from ._helpers import DummyTokenizer, function_tool, make_request, stream

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("suffix", ["<", "<t", "<tool_", "<tool_call"])
def test_hermes_terminal_flush_preserves_possible_marker_prefix(suffix):
    parser = HermesToolParser(DummyTokenizer())
    deltas = stream(parser, ["ordinary text " + suffix], make_request())
    assert "".join(delta.content or "" for delta in deltas) == "ordinary text "
    assert parser.flush_streaming_content() == suffix
    assert parser.flush_streaming_content() == ""


def test_hermes_releases_false_marker_prefix_when_next_chunk_disambiguates():
    parser = HermesToolParser(DummyTokenizer())
    deltas = stream(parser, ["ordinary <tool_", "box", "!"], make_request())
    assert [delta.content for delta in deltas] == ["ordinary ", "<tool_box", "!"]
    assert parser.flush_streaming_content() == ""


def test_hermes_does_not_withhold_visible_prefix_before_open_call():
    parser = HermesToolParser(DummyTokenizer())
    deltas = stream(parser, ["ordinary <", '<tool_call>{"name":"f","arguments":{}'], make_request())
    assert "".join(delta.content or "" for delta in deltas) == "ordinary <"
    assert parser.flush_streaming_content() == ""


@pytest.mark.parametrize("suffix", ["<", "<tool_"])
def test_hermes_flushes_content_suffix_after_completed_call(suffix):
    parser = HermesToolParser(DummyTokenizer())
    output = '<tool_call>{"name":"f","arguments":{}}</tool_call>done ' + suffix
    deltas = stream(parser, list(output), make_request())
    assert "".join(delta.content or "" for delta in deltas) + parser.flush_streaming_content() == "done " + suffix
    assert parser.flush_streaming_content() == ""


def test_hermes_flush_does_not_expose_in_progress_tool_markup():
    parser = HermesToolParser(DummyTokenizer())
    deltas = stream(parser, ["ordinary <tool_", 'call>{"name":"f","arguments":{"text":"<tool_'], make_request())
    assert "".join(delta.content or "" for delta in deltas) == "ordinary "
    assert parser.flush_streaming_content() == ""


def test_hermes_split_marker_completion_discards_only_protocol_not_content():
    parser = HermesToolParser(DummyTokenizer())
    deltas = stream(parser, ["ordinary <tool_", 'call>{"name":"f","arguments":{}}</tool_call>'], make_request())
    assert "".join(delta.content or "" for delta in deltas) == "ordinary "
    assert [call.function.name for delta in deltas for call in delta.tool_calls or []] == ["f"]
    assert parser.flush_streaming_content() == ""


@pytest.mark.parametrize("empty_final_tick", [False, True])
def test_hermes_mixin_terminal_call_flushes_content_without_duplication(empty_final_tick):
    mixin = ToolCallingMixin()
    mixin.tool_parsers = {"dummy": HermesToolParser(DummyTokenizer())}
    request = make_request([function_tool("f", {})])
    chunks = ["ordinary ", "<tool_"] + ([""] if empty_final_tick else [])
    previous = ""
    content = []
    for index, chunk in enumerate(chunks):
        current = previous + chunk
        delta = mixin.extract_tool_calls_streaming(
            "dummy",
            previous,
            current,
            chunk,
            request=request,
            finished=index == len(chunks) - 1,
            stream_id="terminal-content",
        )
        if delta is not None:
            content.append(delta.content or "")
        previous = current
    assert "".join(content) == "ordinary <tool_"
