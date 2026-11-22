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

"""Public parser metadata remains available and reflects real streaming progress."""

import pytest
from easydel.inference.tools.parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser
from easydel.inference.tools.parsers.functiongemma_tool_parser import FunctionGemmaToolParser
from easydel.inference.tools.parsers.gemma4_tool_parser import Gemma4ToolParser
from easydel.inference.tools.parsers.hunyuan_a13b_tool_parser import HunyuanA13BToolParser

from ._helpers import DummyTokenizer, make_request, stream

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    "parser_cls,start,end",
    [
        (FunctionGemmaToolParser, "<start_function_call>", "<end_function_call>"),
        (Gemma4ToolParser, "<|tool_call>", "<tool_call|>"),
    ],
)
def test_gemma_marker_token_ids_are_still_available(parser_cls, start, end):
    tokenizer = DummyTokenizer()
    parser = parser_cls(tokenizer)
    assert tokenizer.decode(parser.tool_call_start_token_ids) == start
    assert tokenizer.decode(parser.tool_call_end_token_ids) == end


def test_hunyuan_public_tracking_is_populated_by_real_streaming():
    parser = HunyuanA13BToolParser(DummyTokenizer())
    output = '<tool_calls>[{"name":"ping","arguments":{}}]</tool_calls>'
    deltas = stream(parser, [output], make_request())
    assert [tc.function.name for delta in deltas for tc in delta.tool_calls or []] == ["ping"]
    assert parser.current_tools_sent == [True]
    assert parser.prev_tool_calls == [{"name": "ping", "arguments": {}}]


def test_dsml_public_state_tracks_open_and_completed_invocation():
    parser = DeepSeekV32ToolParser(DummyTokenizer())
    request = make_request()
    partial = '<｜DSML｜function_calls><｜DSML｜invoke name="ping"><｜DSML｜parameter name="q" string="true">Par'
    assert parser.extract_tool_calls_streaming("", partial, partial, [], [], [], request) is None
    assert parser.in_function and parser.in_param and parser.json_started
    assert parser.current_function_name == "ping"
    assert parser.current_param_name == "q"
    assert parser.current_param_value == "Par"
    assert parser.param_count == 1
    suffix = "is</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>"
    delta = parser.extract_tool_calls_streaming(partial, partial + suffix, suffix, [], [], [], request)
    assert delta is not None and delta.tool_calls[0].function.name == "ping"
    assert not parser.in_function and not parser.in_param
    assert parser.json_closed and parser.header_sent
    assert parser.accumulated_params == {"q": "Paris"}
    assert parser.current_tool_id == parser.invoke_index == 0
    assert parser.streaming_request is request
