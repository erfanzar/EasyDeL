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

"""Minimax multi-tool streaming must deliver complete, nonduplicated JSON arguments."""

from __future__ import annotations

import json

import pytest
from easydel.inference.openai_api_modules import ChatCompletionRequest
from easydel.inference.tools.parsers.minimax_tool_parser import MinimaxToolParser


class _Tokenizer:
    chat_template = ""

    def get_vocab(self) -> dict[str, int]:
        return {"<tool_calls>": 1, "</tool_calls>": 2}


_OUTPUT = (
    '<tool_calls>{"name": "lookup", "arguments": {"text": "abab", "inner": {"v": 1}}}\n'
    '{"name": "lookup", "arguments": {"text": "abab", "inner": {"v": 2}}}</tool_calls>'
)


@pytest.mark.parametrize("granularity", ["token", "char"])
def test_minimax_multi_tool_stream_args_match_final_json(granularity: str) -> None:
    """A caller concatenating each tool's argument deltas gets the final JSON."""
    parser = MinimaxToolParser(_Tokenizer())
    request = ChatCompletionRequest(model="dummy", messages=[{"role": "user", "content": "hi"}])
    chunks = list(_OUTPUT)
    # For token mode, model each boundary marker as one token and all remaining
    # text as individual tokens: a token decoder can split anywhere else.
    if granularity == "token":
        chunks = []
        rest = _OUTPUT
        for marker in ("<tool_calls>", "</tool_calls>"):
            rest = rest.replace(marker, f"\x00{marker}\x00")
        for piece in rest.split("\x00"):
            if piece in {"<tool_calls>", "</tool_calls>"}:
                chunks.append(piece)
            else:
                chunks.extend(piece)
    names: dict[int, str] = {}
    args: dict[int, str] = {}
    previous = ""
    for chunk in chunks:
        current = previous + chunk
        result = parser.extract_tool_calls_streaming(previous, current, chunk, [], [0], [0], request)
        for call in result.tool_calls or [] if result else []:
            if call.function is None:
                continue
            if call.function.name:
                names[call.index] = call.function.name
            if call.function.arguments:
                args[call.index] = args.get(call.index, "") + call.function.arguments
        previous = current

    final = parser.extract_tool_calls(_OUTPUT, request)
    assert final.tools_called and len(final.tool_calls) == 2
    for index, call in enumerate(final.tool_calls):
        assert names[index] == call.function.name
        assert args[index] == call.function.arguments
        assert json.loads(args[index]) == json.loads(call.function.arguments)
