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

"""Public parser integration regressions (no array computation)."""

import json
import sys

import pytest
from easydel.inference.esurge.utils import is_list_of as serving_is_list_of
from easydel.inference.parsing.delegating_parser import DelegatingParser
from easydel.inference.reasoning.abstract_reasoning import _is_list_of as reasoning_is_list_of
from easydel.inference.tools.abstract_tool import import_from_path, is_list_of
from easydel.inference.tools.parsers.gigachat3_tool_parser import GigaChat3ToolParser
from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser
from easydel.inference.tools.parsers.llama4_pythonic_tool_parser import Llama4PythonicToolParser
from easydel.inference.tools.tool_calling_mixin import ToolCallingMixin

from ._helpers import DummyTokenizer, function_tool, make_request, stream

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    "text", ["function call\n", "function", "function call<|role_sep|>", "  function call<|role_", "  "]
)
@pytest.mark.parametrize("empty_final_tick", [False, True])
@pytest.mark.parametrize("character_chunks", [False, True])
def test_gigachat_mixin_finalization_preserves_ambiguous_content(text, empty_final_tick, character_chunks):
    mixin = ToolCallingMixin()
    mixin.tool_parsers = {"dummy": GigaChat3ToolParser(DummyTokenizer())}
    request = make_request([function_tool("f", {})])
    chunks = (list(text) if character_chunks else [text]) + ([""] if empty_final_tick else [])
    previous = ""
    visible = []
    for index, chunk in enumerate(chunks):
        current = previous + chunk
        delta = mixin.extract_tool_calls_streaming(
            "dummy", previous, current, chunk, request=request, finished=index == len(chunks) - 1
        )
        if delta:
            visible.append(delta.content or "")
            assert not delta.tool_calls
        previous = current
    assert "".join(visible) == text


@pytest.mark.parametrize("tail", ["\n", "<|role_sep|>\n"])
def test_gigachat_terminal_flush_is_idempotent_and_does_not_expose_tool_json(tail):
    parser = GigaChat3ToolParser(DummyTokenizer())
    text = "function call" + tail
    assert stream(parser, list(text), make_request()) == []
    assert parser.flush_streaming_content() == text
    assert parser.flush_streaming_content() == ""
    parser = GigaChat3ToolParser(DummyTokenizer())
    active = text + '{"name":"f","arguments":{"secret":"incomplete'
    deltas = stream(parser, list(active), make_request())
    assert not any(delta.content for delta in deltas)
    assert parser.flush_streaming_content() == ""
    assert parser.flush_streaming_content() == ""


@pytest.mark.parametrize(
    "output",
    [
        "[f(1)]",
        "[f(1, x=2)]",
        "[f(*[1])]",
        '[f(**{"x":1})]',
        '[f(x=1, **{"y":2})]',
        "[f(x=1, x=2)]",
        "[obj.f(x=1)]",
        "[f(x=g())]",
        "[f(x=1)]; [g(x=2)]",
    ],
)
def test_llama4_rejects_unsupported_calls_without_dropping_arguments(output):
    parser = Llama4PythonicToolParser(DummyTokenizer())
    result = parser.extract_tool_calls(output, make_request())
    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content == output
    deltas = stream(Llama4PythonicToolParser(DummyTokenizer()), [output], make_request())
    assert not any(delta.tool_calls for delta in deltas)


@pytest.mark.parametrize("delimiters", [False, True])
def test_llama4_keeps_supported_nested_literals_and_multiple_calls(delimiters):
    output = '[f(x={"items":[1, -2, true, null], "text":"雪"}), g()]'
    if delimiters:
        output = "<|python_start|>" + output + "<|python_end|>"
    result = Llama4PythonicToolParser(DummyTokenizer()).extract_tool_calls(output, make_request())
    assert result.tools_called
    assert [call.function.name for call in result.tool_calls] == ["f", "g"]
    assert [json.loads(call.function.arguments) for call in result.tool_calls] == [
        {"x": {"items": [1, -2, True, None], "text": "雪"}},
        {},
    ]


@pytest.mark.parametrize("suffix", ["<box", "<tool_", "<", "<tool_call"])
@pytest.mark.parametrize("character_chunks", [False, True])
def test_hermes_delegating_gate_does_not_repeat_bypassed_content(suffix, character_chunks):
    parser = DelegatingParser(
        tool_parser=HermesToolParser(DummyTokenizer()), tool_request=make_request([function_tool("f", {})])
    )
    previous = ""
    visible = []
    chunks = list(suffix) if character_chunks else ["<", suffix[1:]]
    for chunk in ["ordinary ", *chunks]:
        current = previous + chunk
        result = parser.process_delta(current, chunk, [], previous, [])
        visible.append(result.delta_content or "")
        previous = current
    final = parser.process_final(previous, [])
    visible.append(final.delta_content or "")
    assert "".join(visible) == "ordinary " + suffix
    assert final.accumulated_content == "ordinary " + suffix
    assert not final.tool_calls


def test_hermes_delegating_gate_preserves_real_tool_after_bypassed_content():
    parser = DelegatingParser(
        tool_parser=HermesToolParser(DummyTokenizer()), tool_request=make_request([function_tool("f", {})])
    )
    previous = ""
    results = []
    for chunk in ["ordinary ", "<", 'tool_call>{"name":"f","arguments":{"x":1}}</tool_call>']:
        current = previous + chunk
        results.append(parser.process_delta(current, chunk, [], previous, []))
        previous = current
    final = parser.process_final(previous, [])
    results.append(final)
    assert "".join(result.delta_content or "" for result in results) == "ordinary "
    calls = [call for result in results for call in result.delta_tool_calls or []]
    assert [call.function.name for call in calls if call.function.name] == ["f"]
    assert json.loads("".join(call.function.arguments or "" for call in calls)) == {"x": 1}
    assert final.accumulated_content == "ordinary "


@pytest.mark.parametrize("checker", [is_list_of, reasoning_is_list_of, serving_is_list_of])
def test_shared_list_checker_preserves_first_all_and_non_list_semantics(checker):
    assert checker([1, "x"], int)
    assert not checker([1, "x"], int, check="all")
    assert checker([], int)
    assert checker([], int, check="all")
    assert checker([1, "x"], (int, str), check="all")
    assert not checker((1,), int)
    assert not checker(None, int, check="invalid")


def test_shared_list_checker_preserves_distinct_invalid_check_policies():
    for checker in (is_list_of, reasoning_is_list_of):
        with pytest.raises(AssertionError):
            checker([], int, check="invalid")
    assert not serving_is_list_of([], int, check="invalid")
    assert not serving_is_list_of([1], int, check="invalid")


def test_compatibility_plugin_loader_registers_before_execution_and_propagates_errors(tmp_path):
    path = tmp_path / "parser_compatibility.py"
    name = "review_inference8_compatibility_plugin"
    path.write_text("import sys\nregistered = __name__ in sys.modules\nvalue = 42\n")
    module = import_from_path(name, path)
    assert module is sys.modules[name]
    assert module.registered
    assert module.value == 42
    path.write_text("raise RuntimeError('plugin failure')\n")
    with pytest.raises(RuntimeError, match="plugin failure"):
        import_from_path(name, path)
    assert sys.modules[name] is not module
    with pytest.raises(FileNotFoundError):
        import_from_path(name + "_missing", tmp_path / "missing.py")
