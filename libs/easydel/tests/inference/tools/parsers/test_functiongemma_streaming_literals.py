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

"""FunctionGemma argument deltas must retain the final JSON value types."""

from __future__ import annotations

import json

import pytest
from easydel.inference.openai_api_modules import ChatCompletionRequest
from easydel.inference.tools.parsers.functiongemma_tool_parser import FunctionGemmaToolParser

from .test_engine_streaming import _GreedyTokenizer, _simulate_engine_stream, _ToolParserStreamCase

pytestmark = pytest.mark.cpu_ok

_CASES = [
    ("enabled:true", {"enabled": True}),
    ("enabled:false", {"enabled": False}),
    ("value:null", {"value": None}),
    ("value:-5", {"value": -5}),
    ("value:-5.5e-2", {"value": -0.055}),
    (
        "enabled:true,disabled:false,value:null,count:-5",
        {"enabled": True, "disabled": False, "value": None, "count": -5},
    ),
    (
        'enabled:true,payload:<escape>{"nested":[false,null,-5]}<escape>,count:-2',
        {"enabled": True, "payload": {"nested": [False, None, -5]}, "count": -2},
    ),
]


@pytest.mark.parametrize(("arguments", "expected"), _CASES)
def test_functiongemma_literal_deltas_match_final_arguments(arguments: str, expected: dict) -> None:
    """Split bare literals at every character and concatenate public parser deltas."""
    tokenizer = _GreedyTokenizer()
    parser = FunctionGemmaToolParser(tokenizer)
    request = ChatCompletionRequest(model="dummy", messages=[{"role": "user", "content": "hi"}])
    output = f"<start_function_call>call:get_weather{{{arguments}}}<end_function_call>"
    token_ids = tokenizer.encode(output)
    previous_text = ""
    previous_token_ids: list[int] = []
    deltas: list[str] = []
    for index, token_id in enumerate(token_ids):
        current_token_ids = token_ids[: index + 1]
        current_text = tokenizer.decode(current_token_ids)
        result = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            tokenizer.decode([token_id]),
            previous_token_ids,
            current_token_ids,
            [token_id],
            request,
        )
        for call in result.tool_calls or [] if result else []:
            if call.function and call.function.arguments:
                deltas.append(call.function.arguments)
        previous_text = current_text
        previous_token_ids = current_token_ids

    final = parser.extract_tool_calls(output, request)
    assert final.tools_called and len(final.tool_calls) == 1
    final_arguments = final.tool_calls[0].function.arguments
    concatenated = "".join(deltas)
    assert json.loads(concatenated) == json.loads(final_arguments) == expected
    assert concatenated == final_arguments


@pytest.mark.parametrize(("arguments", "expected"), _CASES)
def test_functiongemma_literal_deltas_match_engine_final_arguments(arguments: str, expected: dict) -> None:
    """Terminal reconciliation must not append a second, incompatible JSON payload."""
    case = _ToolParserStreamCase(
        "functiongemma",
        f"<start_function_call>call:get_weather{{{arguments}}}<end_function_call>",
    )
    output, _, names, deltas = _simulate_engine_stream(case)
    assert "".join(names) == "get_weather"
    final_arguments = output.tool_calls[0].function.arguments
    concatenated = "".join(deltas)
    assert json.loads(concatenated) == json.loads(final_arguments) == expected
    assert concatenated == final_arguments
