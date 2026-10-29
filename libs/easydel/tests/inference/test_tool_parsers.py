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

"""Tool-parser regression tests (pure Python, no array computation)."""

from __future__ import annotations

import json
import re

import easydel.inference.tools.parsers  # noqa: F401
import pytest
from easydel.inference.openai_api_modules import ChatCompletionRequest
from easydel.inference.tools import ToolParserManager

pytestmark = pytest.mark.cpu_ok


class _Tokenizer:
    """Minimal tokenizer stub exposing a vocabulary with the protocol markers as single tokens."""

    _TOKENS = (
        "<|python_tag|>",
        "<minimax:tool_call>",
        "</minimax:tool_call>",
        "<seed:tool_call>",
        "</seed:tool_call>",
        "<seed:think>",
        "</seed:think>",
    )

    def __init__(self):
        self._vocab = {tok: i + 1 for i, tok in enumerate(self._TOKENS)}
        self.chat_template = ""

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [self._vocab[text]] if text in self._vocab else [ord(c) for c in text]


def _request(tools=None) -> ChatCompletionRequest:
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    if tools is not None:
        payload["tools"] = [{"type": "function", "function": {"name": n, "parameters": p}} for n, p in tools]
        payload["tool_choice"] = "auto"
    return ChatCompletionRequest.model_validate(payload)


def _parser(name: str):
    return ToolParserManager.get_tool_parser(name)(_Tokenizer())


def test_phi4mini_arguments_containing_brackets_are_parsed():
    out = 'functools[{"name": "f", "arguments": {"ids": [1, 2]}}]'
    result = _parser("phi4_mini_json").extract_tool_calls(out, _request())
    assert result.tools_called
    assert result.tool_calls[0].function.name == "f"
    assert json.loads(result.tool_calls[0].function.arguments) == {"ids": [1, 2]}


def test_phi4mini_invalid_json_is_returned_as_content():
    out = "functools[{not json}]"
    result = _parser("phi4_mini_json").extract_tool_calls(out, _request())
    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content == out


_MINIMAX_TOOLS = [
    (
        "f",
        {
            "type": "object",
            "properties": {
                "city": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "n": {"type": ["integer", "null"]},
            },
        },
    )
]


def _minimax_output(city: str, n: str) -> str:
    return (
        "<minimax:tool_call>\n"
        '<invoke name="f">\n'
        f'<parameter name="city">{city}</parameter>\n'
        f'<parameter name="n">{n}</parameter>\n'
        "</invoke>\n"
        "</minimax:tool_call>"
    )


def test_minimax_m2_nullable_schema_keeps_supplied_values():
    result = _parser("minimax_m2").extract_tool_calls(_minimax_output("Paris", "5"), _request(_MINIMAX_TOOLS))
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Paris", "n": 5}


def test_minimax_m2_literal_null_still_becomes_none():
    result = _parser("minimax_m2").extract_tool_calls(_minimax_output("null", "null"), _request(_MINIMAX_TOOLS))
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": None, "n": None}


def test_minimax_m2_string_none_kept_for_string_typed_param():
    result = _parser("minimax_m2").extract_tool_calls(_minimax_output("none", "7"), _request(_MINIMAX_TOOLS))
    assert json.loads(result.tool_calls[0].function.arguments)["city"] == "none"


def test_minimax_m2_streaming_populates_streamed_args_for_tool():
    parser = _parser("minimax_m2")
    request = _request(_MINIMAX_TOOLS)
    output = _minimax_output("Paris", "5")
    previous = ""
    for ch in re.split(r"(<minimax:tool_call>|</minimax:tool_call>)", output):
        pieces = [ch] if ch.startswith("<minimax:") or ch.startswith("</minimax:") else list(ch)
        for piece in pieces:
            parser.extract_tool_calls_streaming(previous, previous + piece, piece, [], [], [], request)
            previous += piece
    assert len(parser.streamed_args_for_tool) == 1
    assert json.loads(parser.streamed_args_for_tool[0]) == {"city": "Paris", "n": 5}


_SEED_TOOLS = [
    (
        "add",
        {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}}},
    )
]


def _seed_output() -> str:
    call = "<seed:tool_call>\n<function=add>\n<parameter=a>{a}</parameter>\n<parameter=b>{b}</parameter>\n</function>\n</seed:tool_call>"
    return "<seed:think>x</seed:think>" + call.format(a=5, b="x") + call.format(a=7, b="y")


def _seed_stream():
    parser = _parser("seed_oss")
    request = _request(_SEED_TOOLS)
    names: dict[int, str] = {}
    args: dict[int, str] = {}
    previous = ""
    for part in re.split(r"(</?seed:(?:think|tool_call)>)", _seed_output()):
        pieces = [part] if part.startswith(("<seed:", "</seed:")) else list(part)
        for piece in pieces:
            delta = parser.extract_tool_calls_streaming(previous, previous + piece, piece, [], [], [], request)
            previous += piece
            for tc in (delta.tool_calls if delta else None) or []:
                if tc.function is None:
                    continue
                if tc.function.name:
                    names[tc.index] = tc.function.name
                if tc.function.arguments:
                    args[tc.index] = args.get(tc.index, "") + tc.function.arguments
    return parser, request, names, args


def test_seed_oss_streaming_uses_schema_types_matching_batch_extraction():
    parser, request, names, args = _seed_stream()
    assert names == {0: "add", 1: "add"}
    assert json.loads(args[0]) == {"a": 5, "b": "x"}
    assert json.loads(args[1]) == {"a": 7, "b": "y"}
    batch = parser.extract_tool_calls(_seed_output(), request)
    assert [json.loads(tc.function.arguments) for tc in batch.tool_calls] == [json.loads(args[0]), json.loads(args[1])]


@pytest.mark.parametrize(
    "schema",
    [
        {"anyOf": [{"type": "string"}, {"type": "null"}]},
        {"type": ["string", "null"]},
        {"oneOf": [{"type": "null"}, {"type": "string"}]},
    ],
)
@pytest.mark.parametrize(
    "value, expected",
    [
        ("123", "123"),
        ("true", "true"),
        ("false", "false"),
        ("[1,2]", "[1,2]"),
        ('{"key":123}', '{"key":123}'),
        ('"quoted"', '"quoted"'),
        ("null", None),
    ],
)
def test_seed_oss_nullable_string_union_batch_and_streaming(schema, value, expected):
    _assert_seed_schema_extraction(schema, value, expected)


@pytest.mark.parametrize(
    "schema, value, expected",
    [
        ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, "123", 123),
        ({"type": ["number", "null"]}, "1.5", 1.5),
        ({"anyOf": [{"type": "boolean"}, {"type": "null"}]}, "true", True),
        ({"type": ["boolean", "null"]}, "false", False),
        ({"anyOf": [{"type": "object"}, {"type": "null"}]}, '{"x":1}', {"x": 1}),
        ({"type": ["array", "null"]}, "[1,2]", [1, 2]),
        ({"anyOf": [{"type": "object"}, {"type": "null"}]}, "123", "123"),
        ({"anyOf": [{"type": "object"}, {"type": "null"}]}, "true", "true"),
        ({"type": ["array", "null"]}, '{"x":1}', '{"x":1}'),
        ({"type": ["integer", "null"]}, "true", "true"),
        ({"type": ["boolean", "null"]}, "invalid", "invalid"),
        ({"type": ["string", "integer"]}, "123", 123),
        ({"type": ["string", "integer"]}, "null", "null"),
        ({"anyOf": [{"type": "string"}, {"anyOf": [{"type": "array"}, {"type": "null"}]}]}, "[1]", [1]),
    ],
)
def test_seed_oss_union_coercion_only_converts_to_permitted_types(schema, value, expected):
    _assert_seed_schema_extraction(schema, value, expected)


def _assert_seed_schema_extraction(schema, value, expected):
    """Assert public batch and incremental extraction against an independent expected value."""
    request = _request([("f", {"type": "object", "properties": {"x": schema}})])
    output = (
        "<seed:think>x</seed:think><seed:tool_call>\n<function=f>\n"
        f"<parameter=x>{value}</parameter>\n</function>\n</seed:tool_call>"
    )
    batch = _parser("seed_oss").extract_tool_calls(output, request)
    assert batch.tools_called
    assert batch.tool_calls[0].function.name == "f"
    batch_arguments = json.loads(batch.tool_calls[0].function.arguments)
    assert batch_arguments == {"x": expected}
    assert type(batch_arguments["x"]) is type(expected)

    parser = _parser("seed_oss")
    names = []
    arguments = ""
    previous = ""
    for part in re.split(r"(</?seed:(?:think|tool_call)>)", output):
        pieces = [part] if part.startswith(("<seed:", "</seed:")) else list(part)
        for piece in pieces:
            delta = parser.extract_tool_calls_streaming(previous, previous + piece, piece, [], [], [], request)
            previous += piece
            for tool in (delta.tool_calls if delta else None) or []:
                assert tool.index == 0
                if tool.function:
                    if tool.function.name:
                        names.append(tool.function.name)
                    arguments += tool.function.arguments or ""
    assert names == ["f"]
    streamed_arguments = json.loads(arguments)
    assert streamed_arguments == {"x": expected}
    assert type(streamed_arguments["x"]) is type(expected)


def test_seed_oss_same_name_parallel_calls_keep_separate_entries():
    parser, *_ = _seed_stream()
    assert len(parser.prev_tool_call_arr) == 2
    assert json.loads(parser.prev_tool_call_arr[0]["arguments"]) == {"a": 5, "b": "x"}
    assert json.loads(parser.prev_tool_call_arr[1]["arguments"]) == {"a": 7, "b": "y"}


def test_llama_semicolon_inside_string_argument_does_not_split_call():
    out = '{"name": "run", "arguments": {"code": "a = 1; b = 2"}}; {"name": "other", "parameters": {"k": "v"}}'
    result = _parser("llama3_json").extract_tool_calls(out, _request())
    assert result.tools_called
    assert [tc.function.name for tc in result.tool_calls] == ["run", "other"]
    assert json.loads(result.tool_calls[0].function.arguments) == {"code": "a = 1; b = 2"}
    assert json.loads(result.tool_calls[1].function.arguments) == {"k": "v"}


def test_pythonic_negative_numbers_are_accepted():
    result = _parser("pythonic").extract_tool_calls("[f(x=-1, y=-2.5, z=[-3])]", _request())
    assert result.tools_called
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": -1, "y": -2.5, "z": [-3]}


def test_olmo3_negative_numbers_and_json_names_are_accepted():
    out = "<function_calls>\nf(x=-1, flag=true, none=null)\n</function_calls>"
    result = _parser("olmo3").extract_tool_calls(out, _request())
    assert result.tools_called
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": -1, "flag": True, "none": None}


def test_pythonic_non_literal_argument_is_not_a_tool_call():
    out = "[f(x=g())]"
    result = _parser("pythonic").extract_tool_calls(out, _request())
    assert not result.tools_called
    assert result.content == out


@pytest.mark.parametrize("value", ["none", "nil"])
def test_minimax_m2_only_literal_null_becomes_none(value):
    result = _parser("minimax_m2").extract_tool_calls(_minimax_output("Paris", value), _request(_MINIMAX_TOOLS))
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Paris", "n": value}


def test_pythonic_pattern_timeout_preserves_original_content():
    parser = _parser("pythonic")
    parser.REGEX_TIMEOUT_S = 0.0
    output = '[f(x="value")]'
    result = parser.extract_tool_calls(output, _request())
    assert not result.tools_called
    assert result.content == output


@pytest.mark.parametrize("preseed", [False, True])
def test_xlam_streaming_populates_compatibility_state_without_duplicate_headers(preseed):
    parser = _parser("xlam")
    if preseed:
        parser.current_tools_sent = [False]
    output = '[{"name": "f", "arguments": {"city": "Paris"}}]'
    emitted_names = []
    emitted_arguments = ""
    previous = ""
    for piece in [*output, " ", " ", " "]:
        delta = parser.extract_tool_calls_streaming(previous, previous + piece, piece, [], [], [], _request())
        previous += piece
        for tool in (delta.tool_calls if delta else None) or []:
            if tool.function and tool.function.name:
                emitted_names.append(tool.function.name)
                assert tool.id
            if tool.function and tool.function.arguments:
                emitted_arguments += tool.function.arguments
    assert emitted_names == ["f"]
    assert json.loads(emitted_arguments) == {"city": "Paris"}
    assert parser.current_tools_sent == [True]
    assert parser.current_tool_name_sent
    assert parser.prev_tool_calls == [{"name": "f", "arguments": {"city": "Paris"}}]
