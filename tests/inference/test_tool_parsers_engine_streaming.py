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

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar, get_args

import pytest

import easydel.inference.tools.parsers  # noqa: F401
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.esurge.mixins.parsing import EngineParsingMixin
from easydel.inference.openai_api_modules import ChatCompletionRequest, DeltaToolCall
from easydel.inference.parsing import DelegatingParser
from easydel.inference.tools import ToolParserManager
from easydel.inference.tools.abstract_tool import ToolParserName


class _ParsingHarness(EngineParsingMixin):
    pass


class _GreedyTokenizer:
    """Tokenizer stub that preserves parser boundary tokens as single IDs."""

    _SPECIAL_TOKENS: ClassVar[list[str]] = [
        "<tool_call>",
        "</tool_call>",
        "<longcat_tool_call>",
        "</longcat_tool_call>",
        "<minimax:tool_call>",
        "</minimax:tool_call>",
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁calls▁end｜>",
        "<｜tool▁call▁begin｜>",
        "<｜tool▁call▁end｜>",
        "<｜tool▁sep｜>",
        "<｜DSML｜function_calls>",
        "</｜DSML｜function_calls>",
        '<｜DSML｜invoke name="',
        "</｜DSML｜invoke>",
        '<｜DSML｜parameter name="',
        "</｜DSML｜parameter>",
        "<start_function_call>",
        "<end_function_call>",
        "<|tool_call|>",
        "<function_call>",
        "<|python_tag|>",
        "<|python_start|>",
        "<|python_end|>",
        "[TOOL_CALLS]",
        "<tool_calls>",
        "</tool_calls>",
        "<|action_start|><|plugin|>",
        "<|action_end|>",
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
        "<|tool_call_begin|>",
        "<|tool_call_argument_begin|>",
        "<|tool_call_end|>",
        "<seed:tool_call>",
        "</seed:tool_call>",
        "<seed:think>",
        "</seed:think>",
        "<function_calls>",
        "</function_calls>",
        "<｜tool_calls_begin｜>",
        "<｜tool_calls_end｜>",
        "<｜tool_call_begin｜>",
        "<｜tool_call_end｜>",
        "<｜tool_sep｜>",
        '<steptml:invoke name="',
        "</steptml:invoke>",
        '<steptml:parameter name="',
        "</steptml:parameter>",
        "<function=",
        "</function>",
        '<function name="',
        "<parameter=",
        "</parameter>",
        '<parameter name="',
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
        "functools",
        "<|tool_call>",
        "<tool_call|>",
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


@dataclass(frozen=True)
class _ToolParserStreamCase:
    name: str
    output: str
    visible_text: str = ""


_TOOL_STREAM_CASES = [
    _ToolParserStreamCase(
        "hermes",
        '<tool_call>{"name":"get_weather","arguments":{"location":"Paris"}}</tool_call>',
    ),
    _ToolParserStreamCase(
        "openai",
        '[{"name":"get_weather","arguments":{"location":"Paris"}}]',
    ),
    _ToolParserStreamCase(
        "mistral",
        '[TOOL_CALLS][{"name":"get_weather","arguments":{"location":"Paris"}}]',
    ),
    _ToolParserStreamCase(
        "qwen3_xml",
        '<function name="get_weather"><parameter name="location">Paris</parameter></function>',
    ),
    _ToolParserStreamCase(
        "qwen3_coder",
        "<tool_call><function=get_weather><parameter=location>Paris</parameter></function></tool_call>",
    ),
    _ToolParserStreamCase(
        "llama3_json",
        '<|python_tag|>{"name":"get_weather","arguments":{"location":"Paris"}}',
    ),
    _ToolParserStreamCase(
        "llama4_json",
        '<|python_tag|>{"name":"get_weather","arguments":{"location":"Paris"}}',
    ),
    _ToolParserStreamCase(
        "llama4_pythonic",
        '<|python_start|>[get_weather(location="Paris")]<|python_end|>',
    ),
    _ToolParserStreamCase(
        "pythonic",
        '[get_weather(location="Paris")]',
    ),
    _ToolParserStreamCase(
        "olmo3",
        '<function_calls>\nget_weather(location="Paris")\n</function_calls>',
    ),
    _ToolParserStreamCase(
        "functiongemma",
        '<start_function_call>call:get_weather{location:<escape>"Paris"<escape>}<end_function_call>',
    ),
    _ToolParserStreamCase(
        "gemma4",
        '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>',
    ),
    _ToolParserStreamCase(
        "phi4_mini_json",
        'functools[{"name":"get_weather","arguments":{"location":"Paris"}}]',
    ),
    _ToolParserStreamCase(
        "granite",
        '<|tool_call|>[{"name":"get_weather","arguments":{"location":"Paris"}}]',
    ),
    _ToolParserStreamCase(
        "granite-20b-fc",
        '<function_call>{"name":"get_weather","arguments":{"location":"Paris"}}',
    ),
    _ToolParserStreamCase(
        "deepseek_v3",
        (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>"
            "function<｜tool▁sep｜>get_weather\n```json\n"
            '{"location":"Paris"}'
            "\n```<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        ),
    ),
    _ToolParserStreamCase(
        "deepseek_v31",
        (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>"
            'get_weather<｜tool▁sep｜>{"location":"Paris"}'
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        ),
    ),
    _ToolParserStreamCase(
        "deepseek_v32",
        (
            "<｜DSML｜function_calls>"
            '<｜DSML｜invoke name="get_weather">'
            '<｜DSML｜parameter name="location" string="true">Paris</｜DSML｜parameter>'
            "</｜DSML｜invoke>"
            "</｜DSML｜function_calls>"
        ),
    ),
    _ToolParserStreamCase(
        "glm45",
        "<tool_call>get_weather\n<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>",
    ),
    _ToolParserStreamCase(
        "glm-4.5",
        "<tool_call>get_weather\n<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>",
    ),
    _ToolParserStreamCase(
        "glm47",
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>",
    ),
    _ToolParserStreamCase(
        "glm-4.7",
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>",
    ),
    _ToolParserStreamCase(
        "internlm",
        '<|action_start|><|plugin|>{"name":"get_weather","parameters":{"location":"Paris"}}<|action_end|>',
    ),
    _ToolParserStreamCase(
        "xlam",
        '[{"name":"get_weather","arguments":{"location":"Paris"}}]',
    ),
    _ToolParserStreamCase(
        "gigachat3",
        'function call{"name":"get_weather","arguments":{"location":"Paris"}}',
    ),
    _ToolParserStreamCase(
        "minimax",
        '<tool_calls>\n{"name":"get_weather","arguments":{"location":"Paris"}}\n</tool_calls>',
    ),
    _ToolParserStreamCase(
        "minimax_m2",
        '<minimax:tool_call><invoke name="get_weather"><parameter name="location">Paris</parameter></invoke></minimax:tool_call>',
    ),
    _ToolParserStreamCase(
        "ernie45",
        '<tool_call>{"name":"get_weather","arguments":{"location":"Paris"}}</tool_call>',
    ),
    _ToolParserStreamCase(
        "jamba",
        '<tool_calls>[{"name":"get_weather","arguments":{"location":"Paris"}}]</tool_calls>',
    ),
    _ToolParserStreamCase(
        "kimi_k2",
        (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>ns.get_weather:0"
            '<|tool_call_argument_begin|>{"location":"Paris"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        ),
    ),
    _ToolParserStreamCase(
        "longcat",
        '<longcat_tool_call>{"name":"get_weather","arguments":{"location":"Paris"}}</longcat_tool_call>',
    ),
    _ToolParserStreamCase(
        "hunyuan_a13b",
        '<tool_calls>[{"name":"get_weather","arguments":{"location":"Paris"}}]</tool_calls>',
    ),
    _ToolParserStreamCase(
        "seed_oss",
        "<seed:tool_call><function=get_weather><parameter=location>Paris</parameter></function></seed:tool_call>",
    ),
    _ToolParserStreamCase(
        "step3",
        (
            "<｜tool_calls_begin｜>"
            "<｜tool_call_begin｜>function<｜tool_sep｜>"
            '<steptml:invoke name="get_weather">'
            '<steptml:parameter name="location">Paris</steptml:parameter>'
            "</steptml:invoke>"
            "<｜tool_call_end｜>"
            "<｜tool_calls_end｜>"
        ),
    ),
    _ToolParserStreamCase(
        "step3p5",
        "<tool_call><function=get_weather><parameter=location>Paris</parameter></function></tool_call>",
    ),
    _ToolParserStreamCase(
        "step3.5",
        "<tool_call><function=get_weather><parameter=location>Paris</parameter></function></tool_call>",
    ),
]


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
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )


def _simulate_engine_stream(case: _ToolParserStreamCase) -> tuple[RequestOutput, str, list[str], list[str]]:
    tokenizer = _GreedyTokenizer()
    request = _make_request()
    parser_class = ToolParserManager.get_tool_parser(case.name)
    parser = parser_class(tokenizer)

    delegating_parser = DelegatingParser(
        tool_parser=parser,
        tool_request=request,
    )
    request_data = {
        "delegating_parser": delegating_parser,
        "parser_previous_text": "",
        "parser_previous_token_ids": [],
    }
    request_output = RequestOutput(
        request_id=f"req_{case.name}",
        prompt="hi",
        prompt_token_ids=[],
        outputs=[CompletionOutput(index=0, text="", token_ids=[])],
    )
    completion_output = request_output.outputs[0]
    harness = _ParsingHarness()

    visible_emitted = ""
    streamed_names: list[str] = []
    streamed_arguments: list[str] = []

    token_ids = tokenizer.encode(case.output, add_special_tokens=False)
    for index, token_id in enumerate(token_ids):
        current_token_ids = token_ids[: index + 1]
        accumulated_text = tokenizer.decode(current_token_ids)
        delta_text = tokenizer.decode([token_id])

        parsed = harness._run_output_parsers(
            rd=request_data,
            accumulated_text=accumulated_text,
            delta_text=delta_text,
            token_ids=current_token_ids,
            finished=False,
        )
        harness._update_outputs(
            completion_output,
            request_output,
            0,
            parsed,
            accumulated_text,
            delta_text,
        )

        visible_emitted += request_output.delta_text
        assert visible_emitted == request_output.accumulated_text

        for raw_tool_call in request_output.delta_tool_calls or []:
            tool_call = (
                raw_tool_call
                if isinstance(raw_tool_call, DeltaToolCall)
                else DeltaToolCall.model_validate(raw_tool_call)
            )
            if tool_call.function is None:
                continue
            if tool_call.function.name:
                streamed_names.append(tool_call.function.name)
            if tool_call.function.arguments:
                streamed_arguments.append(tool_call.function.arguments)

    final_parsed = harness._run_output_parsers(
        rd=request_data,
        accumulated_text=case.output,
        delta_text="",
        token_ids=token_ids,
        finished=True,
    )
    harness._update_outputs(
        completion_output,
        request_output,
        0,
        final_parsed,
        case.output,
        "",
    )
    visible_emitted += request_output.delta_text

    for raw_tool_call in request_output.delta_tool_calls or []:
        tool_call = (
            raw_tool_call if isinstance(raw_tool_call, DeltaToolCall) else DeltaToolCall.model_validate(raw_tool_call)
        )
        if tool_call.function is None:
            continue
        if tool_call.function.name:
            streamed_names.append(tool_call.function.name)
        if tool_call.function.arguments:
            streamed_arguments.append(tool_call.function.arguments)

    return request_output, visible_emitted, streamed_names, streamed_arguments


def test_tool_parser_stream_cases_cover_all_supported_parser_names():
    expected_names = set(get_args(ToolParserName))
    case_names = {case.name for case in _TOOL_STREAM_CASES}

    assert case_names == expected_names
    for name in sorted(expected_names):
        assert ToolParserManager.get_tool_parser(name) is not None


@pytest.mark.parametrize("case", _TOOL_STREAM_CASES, ids=[case.name for case in _TOOL_STREAM_CASES])
def test_tool_parser_streaming_through_engine_pipeline(case: _ToolParserStreamCase):
    request_output, visible_emitted, streamed_names, streamed_arguments = _simulate_engine_stream(case)

    assert visible_emitted == case.visible_text
    assert request_output.accumulated_text == case.visible_text
    assert request_output.raw_accumulated_text == case.output

    assert request_output.tool_calls is not None
    assert request_output.tool_calls[0].function.name == "get_weather"
    assert json.loads(request_output.tool_calls[0].function.arguments) == {"location": "Paris"}

    assert "".join(streamed_names) == "get_weather"
    assert streamed_arguments
