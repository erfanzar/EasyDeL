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

"""Streaming regression tests for tool parsers fixed in the round-2 review.

Each test drives a parser's ``extract_tool_calls_streaming`` the way a
consumer does — one delta per token or per character — and asserts that the
concatenated argument deltas equal the final batch-extracted arguments JSON,
that every tool's name is emitted exactly once, and that no protocol markup
leaks as visible content.
"""

from __future__ import annotations

import json
from typing import ClassVar

import easydel.inference.tools.parsers  # noqa: F401
import pytest
from easydel.inference.openai_api_modules import ChatCompletionRequest, DeltaToolCall
from easydel.inference.tools import ToolParserManager


class _GreedyTokenizer:
    """Tokenizer stub that keeps tool-call boundary markers as single tokens."""

    _SPECIAL_TOKENS: ClassVar[list[str]] = [
        "<tool_call>",
        "</tool_call>",
        "<|tool_call|>",
        "[TOOL_CALLS]",
        "<tool_calls>",
        "</tool_calls>",
        "<|python_start|>",
        "<|python_end|>",
        "<function_calls>",
        "</function_calls>",
    ]

    def __init__(self):
        self.chat_template = ""
        self._vocab: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._next_token_id = 1
        self._special_tokens = sorted(set(self._SPECIAL_TOKENS), key=len, reverse=True)
        for token in self._special_tokens:
            self._register(token)

    def _register(self, token: str) -> None:
        if token not in self._vocab:
            self._vocab[token] = self._next_token_id
            self._id_to_token[self._next_token_id] = token
            self._next_token_id += 1

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        token_ids: list[int] = []
        position = 0
        while position < len(text):
            matched = None
            for token in self._special_tokens:
                if text.startswith(token, position):
                    matched = token
                    break
            if matched is None:
                matched = text[position]
            self._register(matched)
            token_ids.append(self._vocab[matched])
            position += len(matched)
        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(self._id_to_token[token_id] for token_id in token_ids)


def _make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {"type": "string"},
                            },
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )


def _chunks(output: str, granularity: str) -> list[str]:
    tokenizer = _GreedyTokenizer()
    if granularity == "token":
        return [tokenizer.decode([token_id]) for token_id in tokenizer.encode(output)]
    if granularity == "char":
        return list(output)
    raise ValueError(granularity)


def _harvest(delta, names: dict[int, str], args: dict[int, str], content: list[str]) -> None:
    if delta is None:
        return
    if delta.content:
        content.append(delta.content)
    for tool_call in delta.tool_calls or []:
        raw = tool_call if isinstance(tool_call, DeltaToolCall) else DeltaToolCall.model_validate(tool_call)
        if raw.function is None:
            continue
        index = raw.index if raw.index is not None else 0
        if raw.function.name:
            names[index] = names.get(index, "") + raw.function.name
        if raw.function.arguments:
            args[index] = args.get(index, "") + raw.function.arguments


def _stream(parser_name: str, output: str, granularity: str):
    parser = ToolParserManager.get_tool_parser(parser_name)(_GreedyTokenizer())
    request = _make_request()
    names: dict[int, str] = {}
    args: dict[int, str] = {}
    content: list[str] = []
    previous = ""
    for chunk in _chunks(output, granularity):
        delta = parser.extract_tool_calls_streaming(previous, previous + chunk, chunk, [], [0], [0], request)
        _harvest(delta, names, args, content)
        previous += chunk
    return parser, request, names, args, content


def _assert_stream_matches_batch(parser_name: str, output: str, granularity: str) -> None:
    parser, request, names, args, _content = _stream(parser_name, output, granularity)
    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called and extracted.tool_calls
    for index, tool_call in enumerate(extracted.tool_calls):
        assert names.get(index) == tool_call.function.name, (
            f"{parser_name}/{granularity}: name mismatch for tool {index}: {names} vs {tool_call.function.name}"
        )
        concatenated = args.get(index, "")
        assert concatenated == tool_call.function.arguments, (
            f"{parser_name}/{granularity}: concatenated args {concatenated!r} != final {tool_call.function.arguments!r}"
        )
        assert json.loads(concatenated) == json.loads(tool_call.function.arguments)
    assert not args or extracted.tool_calls, "streamed args without a final tool call"


GRANITE_ONE = '<|tool_call|>[{"name":"get_weather","arguments":{"location":"Paris"}}]'
GRANITE_TWO = (
    '<|tool_call|>[{"name":"get_weather","arguments":{"location":"Paris"}},'
    '{"name":"get_weather","arguments":{"location":"NYC"}}]'
)
MISTRAL_ONE = '[TOOL_CALLS][{"name":"get_weather","arguments":{"location":"Paris"}}]'
JAMBA_ONE = '<tool_calls>[{"name":"get_weather","arguments":{"location":"Paris"}}]</tool_calls>'
GLM45_ONE = "<tool_call>get_weather\n<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>"


@pytest.mark.parametrize("granularity", ["token", "char"])
def test_granite_streaming_single_call_concat_matches_final(granularity):
    """R2PA-01: no swallowed IndexError on the leading '[' chunk; args stream intact."""
    _assert_stream_matches_batch("granite", GRANITE_ONE, granularity)


@pytest.mark.parametrize("granularity", ["token", "char"])
def test_granite_streaming_multi_call_concat_matches_final(granularity):
    """R2PA-02: both tools in the array stream with complete args."""
    _assert_stream_matches_batch("granite", GRANITE_TWO, granularity)


def test_granite_multi_call_single_big_delta_emits_intermediate_tool():
    """R2PA-02: a delta spanning several array elements emits every closed tool.

    Before the fix the parser jumped ``current_tool_id`` to the last element,
    never emitting the intermediate tool and desyncing ``streamed_args_for_tool``.
    """
    parser = ToolParserManager.get_tool_parser("granite")(_GreedyTokenizer())
    request = _make_request()
    names: dict[int, str] = {}
    args: dict[int, str] = {}
    content: list[str] = []
    delta = parser.extract_tool_calls_streaming("", GRANITE_TWO, GRANITE_TWO, [], [0], [0], request)
    _harvest(delta, names, args, content)
    assert names.get(0) == "get_weather", f"first tool name missing: {names}"
    assert args.get(0) == '{"location": "Paris"}', f"first tool args missing: {args}"
    # The newest element may still be waiting for its name/args streaming calls.
    delta = parser.extract_tool_calls_streaming(GRANITE_TWO, GRANITE_TWO, "", [], [0], [0], request)
    _harvest(delta, names, args, content)
    assert names.get(1) == "get_weather", f"second tool name missing: {names}"
    # And a final empty-delta pass completes its arguments.
    delta = parser.extract_tool_calls_streaming(GRANITE_TWO, GRANITE_TWO, "", [], [0], [0], request)
    _harvest(delta, names, args, content)
    assert args.get(1) == '{"location": "NYC"}', f"second tool args missing: {args}"


def test_llama4_pythonic_start_marker_only_chunk_does_not_raise():
    """R2PA-03: '<|python_start|>' alone must not raise an empty-module IndexError."""
    parser = ToolParserManager.get_tool_parser("llama4_pythonic")(_GreedyTokenizer())
    request = _make_request()
    delta = parser.extract_tool_calls_streaming(
        "<|python_start|>",
        "<|python_start|>",
        "<|python_start|>",
        [],
        [0],
        [0],
        request,
    )
    assert delta is None or delta.content is not None


@pytest.mark.parametrize("granularity", ["token", "char"])
def test_llama4_pythonic_streaming_concat_matches_final(granularity):
    _assert_stream_matches_batch(
        "llama4_pythonic", '<|python_start|>[get_weather(location="Paris")]<|python_end|>', granularity
    )


@pytest.mark.parametrize("granularity", ["token", "char"])
def test_mistral_streaming_concat_matches_final(granularity):
    """R2PA-04: prefix-based diffing; concatenated deltas end at the final JSON."""
    _assert_stream_matches_batch("mistral", MISTRAL_ONE, granularity)


@pytest.mark.parametrize("granularity", ["token", "char"])
def test_mistral_streaming_multi_call_concat_matches_final(granularity):
    _assert_stream_matches_batch(
        "mistral",
        '[TOOL_CALLS][{"name":"get_weather","arguments":{"location":"Paris"}},'
        '{"name":"get_weather","arguments":{"location":"NYC"}}]',
        granularity,
    )


@pytest.mark.parametrize("granularity", ["token", "char"])
def test_jamba_streaming_concat_matches_final(granularity):
    """R2PA-05: transition uses prefix remainder + completion flush of the tail."""
    _assert_stream_matches_batch("jamba", JAMBA_ONE, granularity)


@pytest.mark.parametrize("granularity", ["token", "char"])
def test_jamba_streaming_multi_call_concat_matches_final(granularity):
    _assert_stream_matches_batch(
        "jamba",
        '<tool_calls>[{"name":"get_weather","arguments":{"location":"Paris"}},'
        '{"name":"get_weather","arguments":{"location":"NYC"}}]</tool_calls>',
        granularity,
    )


@pytest.mark.parametrize("granularity", ["char", "split"])
def test_glm45_split_start_marker_still_yields_tool_call(granularity):
    """R2PA-06: '<tool_call>' split across deltas must not be dropped from the buffer."""
    if granularity == "char":
        chunks = _chunks(GLM45_ONE, "char")
    else:
        marker_split = 4  # "<tool_call>"[:4] == "<too"
        chunks = [GLM45_ONE[:marker_split], GLM45_ONE[marker_split:]]

    parser = ToolParserManager.get_tool_parser("glm45")(_GreedyTokenizer())
    request = _make_request()
    names: dict[int, str] = {}
    args: dict[int, str] = {}
    content: list[str] = []
    previous = ""
    for chunk in chunks:
        delta = parser.extract_tool_calls_streaming(previous, previous + chunk, chunk, [], [0], [0], request)
        _harvest(delta, names, args, content)
        previous += chunk

    extracted = parser.extract_tool_calls(GLM45_ONE, request)
    assert extracted.tools_called and extracted.tool_calls
    assert names.get(0) == extracted.tool_calls[0].function.name, f"tool name never emitted: {names}"
    assert args.get(0) == extracted.tool_calls[0].function.arguments, (
        f"args {args.get(0)!r} != final {extracted.tool_calls[0].function.arguments!r}"
    )
    assert "".join(content) == "", f"tool-call markup leaked as content: {content}"


def test_tool_calling_mixin_finished_catch_up_completes_withheld_tail():
    """R2PA-07: finished=True delivers the argument tail direct mixin users miss."""
    from easydel.inference.tools.tool_calling_mixin import ToolCallingMixin

    class _Server(ToolCallingMixin):
        def __init__(self, parser_name: str):
            self.tool_parsers = {"m": ToolParserManager.get_tool_parser(parser_name)(_GreedyTokenizer())}

    for parser_name, output in (
        ("granite", GRANITE_ONE),
        ("mistral", MISTRAL_ONE),
        ("jamba", JAMBA_ONE),
        ("glm45", GLM45_ONE),
    ):
        server = _Server(parser_name)
        parser = server.tool_parsers["m"]
        request = _make_request()
        names: dict[int, str] = {}
        args: dict[int, str] = {}
        content: list[str] = []
        # Whole generation delivered as one chunk (worst-case delta granularity).
        delta = server.extract_tool_calls_streaming("m", "", output, output, [], [0], [0], request)
        _harvest(delta, names, args, content)
        delta = server.extract_tool_calls_streaming("m", output, output, "", [], [0], [], request, finished=True)
        _harvest(delta, names, args, content)

        extracted = parser.extract_tool_calls(output, request)
        assert extracted.tools_called and extracted.tool_calls
        assert names.get(0) == extracted.tool_calls[0].function.name, parser_name
        assert args.get(0) == extracted.tool_calls[0].function.arguments, (
            f"{parser_name}: args {args.get(0)!r} != final {extracted.tool_calls[0].function.arguments!r}"
        )


def test_internlm2_array_argument_stream_concat_matches_final():
    """An array-valued argument must keep streaming as an extension of what was already sent.

    Withholding only ``"``/``}`` committed a trailing ``]`` early, so later fills were never
    prefix extensions: streaming stalled and the catch-up re-sent the full arguments,
    producing invalid concatenated JSON.
    """
    from easydel.inference.tools.tool_calling_mixin import ToolCallingMixin

    class _Server(ToolCallingMixin):
        def __init__(self):
            self.tool_parsers = {"m": ToolParserManager.get_tool_parser("internlm")(_GreedyTokenizer())}

    server = _Server()
    parser = server.tool_parsers["m"]
    request = _make_request()
    chunks = [
        "<|action_start|><|plugin|>",
        '{"name": "get_weather", ',
        '"parameters": {"location": ["Paris"',
        ', "NYC"',
        "]}}",
        "<|action_end|>",
    ]
    output = "".join(chunks)
    names: dict[int, str] = {}
    args: dict[int, str] = {}
    content: list[str] = []
    previous = ""
    for chunk in chunks:
        delta = server.extract_tool_calls_streaming("m", previous, previous + chunk, chunk, [], [0], [0], request)
        _harvest(delta, names, args, content)
        previous += chunk
    delta = server.extract_tool_calls_streaming("m", output, output, "", [], [0], [], request, finished=True)
    _harvest(delta, names, args, content)

    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called and extracted.tool_calls
    assert names.get(0) == "get_weather"
    assert args.get(0) == extracted.tool_calls[0].function.arguments
    assert json.loads(args[0]) == {"location": ["Paris", "NYC"]}


def test_stable_json_stream_prefix_withholds_unfinished_numbers():
    """Trailing closers and a trailing number literal are provisional (r12 S12-1)."""
    from easydel.inference.tools.utils import stable_json_stream_prefix

    assert stable_json_stream_prefix('{"x": 1}') == '{"x": '
    assert stable_json_stream_prefix('{"x": [1, -2.5e3]}') == '{"x": [1, '
    assert stable_json_stream_prefix('{"a": "Par"}') == '{"a": "Par'
    assert stable_json_stream_prefix('{"a": "Paris", "b": true}') == '{"a": "Paris", "b": true'


MISTRAL_EXP = '[TOOL_CALLS][{"name":"get_weather","arguments":{"location":"Paris","temp":1e-2,"n":12}}]'
JAMBA_EXP = '<tool_calls>[{"name":"get_weather","arguments":{"temp":1e-2,"location":"Paris"}}]</tool_calls>'


@pytest.mark.parametrize("granularity", ["token", "char"])
@pytest.mark.parametrize(("parser_name", "output"), [("mistral", MISTRAL_EXP), ("jamba", JAMBA_EXP)])
def test_scientific_notation_arguments_stream_append_only(parser_name, output, granularity):
    """``1`` must not be committed before ``e-2`` turns it into ``0.01`` (r12 S12-1)."""
    _assert_stream_matches_batch(parser_name, output, granularity)


def test_internlm2_negative_exponent_stream_concat_matches_final():
    """The r12 S12-1 scenario: chunks ``{"x":1`` then ``e-2}}`` must not yield ``{"x": 1{"x": 0.01}``."""
    from easydel.inference.tools.tool_calling_mixin import ToolCallingMixin

    class _Server(ToolCallingMixin):
        def __init__(self):
            self.tool_parsers = {"m": ToolParserManager.get_tool_parser("internlm")(_GreedyTokenizer())}

    server = _Server()
    parser = server.tool_parsers["m"]
    request = _make_request()
    chunks = ['<|action_start|><|plugin|>{"name":"get_weather","parameters":', '{"x":1', "e-2}}<|action_end|>"]
    output = "".join(chunks)
    names: dict[int, str] = {}
    args: dict[int, str] = {}
    content: list[str] = []
    previous = ""
    for chunk in chunks:
        delta = server.extract_tool_calls_streaming("m", previous, previous + chunk, chunk, [], [0], [0], request)
        _harvest(delta, names, args, content)
        previous += chunk
    delta = server.extract_tool_calls_streaming("m", output, output, "", [], [0], [], request, finished=True)
    _harvest(delta, names, args, content)

    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called and extracted.tool_calls
    assert names.get(0) == "get_weather"
    assert args.get(0) == extracted.tool_calls[0].function.arguments
    assert json.loads(args[0]) == {"x": 0.01}


def test_internlm2_split_surrogate_pair_stream_concat_matches_final():
    """``\\ud83d`` then ``\\ude00`` must not commit a lone high surrogate (r13 S12-1)."""
    from easydel.inference.tools.tool_calling_mixin import ToolCallingMixin
    from easydel.inference.tools.utils import stable_json_stream_prefix

    assert stable_json_stream_prefix('{"x": "a\ud83d"}') == '{"x": "a'

    class _Server(ToolCallingMixin):
        def __init__(self):
            self.tool_parsers = {"m": ToolParserManager.get_tool_parser("internlm")(_GreedyTokenizer())}

    server = _Server()
    parser = server.tool_parsers["m"]
    request = _make_request()
    chunks = [
        '<|action_start|><|plugin|>{"name":"get_weather","parameters":',
        '{"x":"\\ud83d',
        '\\ude00"}}<|action_end|>',
    ]
    output = "".join(chunks)
    names: dict[int, str] = {}
    args: dict[int, str] = {}
    content: list[str] = []
    previous = ""
    for chunk in chunks:
        delta = server.extract_tool_calls_streaming("m", previous, previous + chunk, chunk, [], [0], [0], request)
        _harvest(delta, names, args, content)
        previous += chunk
    delta = server.extract_tool_calls_streaming("m", output, output, "", [], [0], [], request, finished=True)
    _harvest(delta, names, args, content)

    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called and extracted.tool_calls
    assert args.get(0) == extracted.tool_calls[0].function.arguments
    assert json.loads(args[0]) == {"x": "\U0001f600"}


def test_tool_stream_cap_never_evicts_a_stream_that_already_emitted(monkeypatch):
    """Past the tracked-stream cap only idle streams are evicted; a live one keeps its state (r14 S12-1)."""
    import easydel.inference.tools.tool_calling_mixin as mixin_module
    from easydel.inference.tools.tool_calling_mixin import ToolCallingMixin

    monkeypatch.setattr(mixin_module, "_MAX_TRACKED_TOOL_STREAMS", 2)

    class _Server(ToolCallingMixin):
        def __init__(self):
            self.tool_parsers = {"m": ToolParserManager.get_tool_parser("internlm")(_GreedyTokenizer())}

    server = _Server()
    parser = server.tool_parsers["m"]
    request = _make_request()
    chunks = ['<|action_start|><|plugin|>{"name":"get_weather","parameters":', '{"x":"Par', 'is"}}<|action_end|>']
    output = "".join(chunks)
    names: dict[int, str] = {}
    args: dict[int, str] = {}
    content: list[str] = []

    def step(prev, chunk):
        delta = server.extract_tool_calls_streaming("m", prev, prev + chunk, chunk, [], [0], [0], request, stream_id="A")
        _harvest(delta, names, args, content)

    step("", chunks[0])
    step(chunks[0], chunks[1])
    assert names.get(0) == "get_weather" and args.get(0), (names, args)  # A has emitted name + prefix

    # Three idle streams (nothing emitted to the client) push the table past the cap of 2.
    for sid in ("B", "C", "D"):
        server.extract_tool_calls_streaming("m", "", "", "", [], [0], [0], request, stream_id=sid)
    tracked = {key[1] for key in server._mixin_tool_streams}
    assert "A" in tracked and len(tracked) == 2, tracked  # idle B, C evicted; live A kept

    step(chunks[0] + chunks[1], chunks[2])
    delta = server.extract_tool_calls_streaming(
        "m", output, output, "", [], [0], [], request, finished=True, stream_id="A"
    )
    _harvest(delta, names, args, content)
    extracted = parser.extract_tool_calls(output, request)
    assert extracted.tools_called and extracted.tool_calls
    assert names.get(0) == "get_weather"  # announced once, not re-announced
    assert args.get(0) == extracted.tool_calls[0].function.arguments
    assert json.loads(args[0]) == {"x": "Paris"}


def _internlm_mixin_server():
    from easydel.inference.tools.tool_calling_mixin import ToolCallingMixin

    class _Server(ToolCallingMixin):
        def __init__(self):
            self.tool_parsers = {"m": ToolParserManager.get_tool_parser("internlm")(_GreedyTokenizer())}

    return _Server()


def test_tool_stream_cap_keeps_streams_that_only_emitted_content(monkeypatch):
    """Delivered content makes a stream non-restartable, so it is not evicted as idle (r15 S12-3)."""
    import easydel.inference.tools.tool_calling_mixin as mixin_module

    monkeypatch.setattr(mixin_module, "_MAX_TRACKED_TOOL_STREAMS", 2)
    server = _internlm_mixin_server()
    request = _make_request()
    delta = server.extract_tool_calls_streaming("m", "", "hello", "hello", [], [0], [0], request, stream_id="A")
    assert delta is not None and delta.content == "hello"
    for sid in ("B", "C"):
        server.extract_tool_calls_streaming("m", "", "", "", [], [0], [0], request, stream_id=sid)
    tracked = {key[1] for key in server._mixin_tool_streams}
    assert "A" in tracked and len(tracked) == 2, tracked  # idle B evicted, content stream A kept


def test_tool_stream_hard_cap_drops_least_recently_used_and_abort_releases(monkeypatch):
    """Abandoned emitted streams cannot grow the table forever (r15 S12-2)."""
    import easydel.inference.tools.tool_calling_mixin as mixin_module

    monkeypatch.setattr(mixin_module, "_MAX_TRACKED_TOOL_STREAMS", 1)
    monkeypatch.setattr(mixin_module, "_HARD_MAX_TRACKED_TOOL_STREAMS_FACTOR", 2)
    server = _internlm_mixin_server()
    request = _make_request()
    for sid in ("A", "B", "C"):
        server.extract_tool_calls_streaming("m", "", "hello", "hello", [], [0], [0], request, stream_id=sid)
    tracked = [key[1] for key in server._mixin_tool_streams]
    assert tracked == ["B", "C"], tracked  # over the hard cap of 2: oldest emitted stream A dropped

    server.abort_tool_stream("m", stream_id="B")
    assert [key[1] for key in server._mixin_tool_streams] == ["C"]


def test_tool_stream_terminal_parser_failure_still_ends_stream_and_catches_up():
    """A parser raising on the finished call must not skip cleanup or the argument catch-up (r15 S12-1)."""
    server = _internlm_mixin_server()
    parser = server.tool_parsers["m"]
    request = _make_request()
    chunks = ['<|action_start|><|plugin|>{"name":"get_weather","parameters":', '{"x":"Par', 'is"}}<|action_end|>']
    output = "".join(chunks)
    names: dict[int, str] = {}
    args: dict[int, str] = {}
    content: list[str] = []
    previous = ""
    for chunk in chunks[:2]:
        delta = server.extract_tool_calls_streaming(
            "m", previous, previous + chunk, chunk, [], [0], [0], request, stream_id="A"
        )
        _harvest(delta, names, args, content)
        previous += chunk
    assert names.get(0) == "get_weather" and args.get(0)

    stream_parser = server._mixin_tool_streams[("m", "A")].parser

    def _raise(*_args, **_kwargs):
        raise RuntimeError("terminal parse failure")

    stream_parser.extract_tool_calls_streaming = _raise
    delta = server.extract_tool_calls_streaming(
        "m", previous, output, chunks[2], [], [0], [], request, finished=True, stream_id="A"
    )
    _harvest(delta, names, args, content)
    assert ("m", "A") not in server._mixin_tool_streams  # stream state released
    extracted = parser.extract_tool_calls(output, request)
    assert args.get(0) == extracted.tool_calls[0].function.arguments  # withheld tail delivered
    assert json.loads(args[0]) == {"x": "Paris"}


def _mixin_server(parser_name: str):
    from easydel.inference.tools.tool_calling_mixin import ToolCallingMixin

    class _Server(ToolCallingMixin):
        def __init__(self):
            self.tool_parsers = {"m": ToolParserManager.get_tool_parser(parser_name)(_GreedyTokenizer())}

    return _Server()


def _drive_mixin_stream(server, request, output: str, sink):
    """Yield one generator step per character, then a finished step."""
    previous = ""
    for char in output:
        yield server.extract_tool_calls_streaming("m", previous, previous + char, char, [], [0], [0], request)
        previous += char
    yield server.extract_tool_calls_streaming("m", output, output, "", [], [0], [], request, finished=True)


def test_tool_calling_mixin_interleaved_streams_on_one_shared_parser_do_not_leak():
    """F3: two concurrent streams on the same registered parser keep independent state.

    Previously the parser instance and the name flags were shared per model, so stream B
    starting cleared A's flags (A's catch-up re-sent the tool name) and A/B corrupted each
    other's ``streamed_args_for_tool``.
    """
    server = _mixin_server("granite")
    shared = server.tool_parsers["m"]
    request_a, request_b = _make_request(), _make_request()
    sinks = {key: ({}, {}, []) for key in ("a", "b")}
    gens = {
        "a": _drive_mixin_stream(server, request_a, GRANITE_ONE, sinks["a"]),
        "b": _drive_mixin_stream(server, request_b, GRANITE_TWO, sinks["b"]),
    }
    live = dict(gens)
    while live:
        for key in list(live):
            try:
                delta = next(live[key])
            except StopIteration:
                del live[key]
                continue
            _harvest(delta, *sinks[key])

    for key, output in (("a", GRANITE_ONE), ("b", GRANITE_TWO)):
        names, args, _content = sinks[key]
        extracted = shared.extract_tool_calls(output, request_a)
        assert extracted.tools_called
        for index, tool_call in enumerate(extracted.tool_calls):
            assert names.get(index) == tool_call.function.name, f"stream {key}: {names}"
            assert args.get(index) == tool_call.function.arguments, f"stream {key}: {args}"
    # The registered instance is never driven by a stream.
    assert shared.current_tool_id == -1
    assert shared.streamed_args_for_tool == []


def test_tool_calling_mixin_sequential_streams_on_one_shared_parser_do_not_leak():
    """F3: a second stream must not be diffed against the first stream's leftovers."""
    server = _mixin_server("granite")
    shared = server.tool_parsers["m"]
    for output in (GRANITE_TWO, GRANITE_ONE):
        request = _make_request()
        sink = ({}, {}, [])
        for delta in _drive_mixin_stream(server, request, output, sink):
            _harvest(delta, *sink)
        names, args, _content = sink
        extracted = shared.extract_tool_calls(output, request)
        for index, tool_call in enumerate(extracted.tool_calls):
            assert names.get(index) == tool_call.function.name
            assert args.get(index) == tool_call.function.arguments


def test_tool_calling_mixin_catch_up_first_announcement_carries_id_and_type():
    """F6: when the finished catch-up is the first time a call is announced it has id/type."""
    server = _mixin_server("granite")
    request = _make_request()
    delta = server.extract_tool_calls_streaming("m", "", GRANITE_ONE, "", [], [0], [], request, finished=True)
    assert delta is not None and delta.tool_calls
    call = delta.tool_calls[0]
    assert call.function.name == "get_weather"
    assert call.id, "catch-up announcement must carry a call id"
    assert call.type == "function"


def test_tool_calling_mixin_catch_up_does_not_duplicate_args_for_minimax():
    """F2: parsers that keep their own state (minimax) must not have args re-sent by the catch-up."""
    output = (
        '<tool_calls>{"name": "get_weather", "arguments": {"location": "Paris"}}\n'
        '{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_calls>'
    )
    server = _mixin_server("minimax")
    request = _make_request()
    sink = ({}, {}, [])
    for delta in _drive_mixin_stream(server, request, output, sink):
        _harvest(delta, *sink)
    names, args, _content = sink
    extracted = ToolParserManager.get_tool_parser("minimax")(_GreedyTokenizer()).extract_tool_calls(output, request)
    assert extracted.tools_called and len(extracted.tool_calls) == 2
    for index, tool_call in enumerate(extracted.tool_calls):
        assert names.get(index) == tool_call.function.name
        assert args.get(index) == tool_call.function.arguments, f"args duplicated or lost: {args}"


def test_minimax_pretty_printed_nested_arguments_stream_to_final_json():
    """F4: a nested ``}`` followed by a newline must not advance to the next tool early."""
    output = (
        '<tool_calls>{"name": "get_weather", "arguments": {\n "location": "Paris",\n "a": {\n  "b": 1\n }\n}}'
        "</tool_calls>"
    )
    parser = ToolParserManager.get_tool_parser("minimax")(_GreedyTokenizer())
    request = _make_request()
    names: dict[int, str] = {}
    args: dict[int, str] = {}
    content: list[str] = []
    previous = ""
    for char in output:
        delta = parser.extract_tool_calls_streaming(previous, previous + char, char, [], [0], [0], request)
        _harvest(delta, names, args, content)
        previous += char
    assert names.get(0) == "get_weather"
    assert json.loads(args[0]) == {"location": "Paris", "a": {"b": 1}}


@pytest.mark.parametrize(
    ("parser_name", "prefix", "two_calls"),
    [
        (
            "granite",
            "<|tool_call|>[{",
            GRANITE_TWO,
        ),
        (
            "mistral",
            "[TOOL_CALLS][{",
            '[TOOL_CALLS][{"name":"get_weather","arguments":{"location":"Paris"}},'
            '{"name":"get_weather","arguments":{"location":"NYC"}}]',
        ),
        (
            "jamba",
            "<tool_calls>[{",
            '<tool_calls>[{"name":"get_weather","arguments":{"location":"Paris"}},'
            '{"name":"get_weather","arguments":{"location":"NYC"}}]</tool_calls>',
        ),
    ],
)
def test_multi_element_jump_announces_unnamed_tool_before_its_arguments(parser_name, prefix, two_calls):
    """F5: arguments for an index must never precede that index's name, id and type."""
    parser = ToolParserManager.get_tool_parser(parser_name)(_GreedyTokenizer())
    request = _make_request()
    assert parser.extract_tool_calls_streaming("", prefix, prefix, [], [0], [0], request) is None
    assert parser.current_tool_id == 0 and not parser.current_tool_name_sent
    delta = parser.extract_tool_calls_streaming(prefix, two_calls, two_calls[len(prefix) :], [], [0], [0], request)
    assert delta is not None and delta.tool_calls
    first = delta.tool_calls[0]
    assert first.index == 0
    assert first.function.name == "get_weather"
    assert first.id and first.type == "function"
    assert json.loads(first.function.arguments) == {"location": "Paris"}


def test_glm45_mixin_finished_flushes_withheld_start_marker_fragment():
    """F7: content ending in a possible start-marker prefix (``<``) is delivered at end of stream."""
    server = _mixin_server("glm45")
    request = _make_request()
    text = "x <"
    sink = ({}, {}, [])
    previous = ""
    for char in text:
        delta = server.extract_tool_calls_streaming("m", previous, previous + char, char, [], [0], [0], request)
        _harvest(delta, *sink)
        previous += char
    assert "".join(sink[2]) == "x ", "trailing '<' is withheld until the stream ends"
    delta = server.extract_tool_calls_streaming("m", text, text, "", [], [0], [], request, finished=True)
    _harvest(delta, *sink)
    assert "".join(sink[2]) == text


@pytest.mark.parametrize("parser_name", ["mistral", "jamba"])
def test_bot_token_only_delta_token_ids_early_return(parser_name):
    """F8: a delta consisting solely of the tool-start token id returns early without touching state."""
    parser = ToolParserManager.get_tool_parser(parser_name)(_GreedyTokenizer())
    request = _make_request()
    marker_id = parser.bot_token_id if parser_name == "mistral" else parser.tool_calls_start_token_id
    text = "[TOOL_CALLS][{" if parser_name == "mistral" else "<tool_calls>[{"
    assert parser.extract_tool_calls_streaming("", text, text, [], [marker_id], [marker_id], request) is None
    assert parser.current_tool_id == -1, "single start-token delta must not advance parser state"
    # The same text with additional token ids in the delta is processed normally.
    parser.extract_tool_calls_streaming("", text, text, [], [marker_id, 1], [marker_id, 1], request)
    assert parser.current_tool_id == 0
