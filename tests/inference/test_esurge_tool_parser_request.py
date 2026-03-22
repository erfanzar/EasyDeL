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

from easydel.inference.esurge.mixins.io import EngineIOMixin
from easydel.inference.esurge.mixins.parsing import EngineParsingMixin
from easydel.inference.openai_api_modules import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)


class _ParsingHarness(EngineParsingMixin):
    pass


class _RecordingToolParser:
    def __init__(self):
        self.batch_tools = None
        self.streaming_tools = None

    def extract_tool_calls(self, _content, request):
        self.batch_tools = request.tools
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[
                ToolCall(
                    type="function",
                    function=FunctionCall(name="lookup", arguments='{"limit": 5}'),
                )
            ],
            content=None,
        )

    def extract_tool_calls_streaming(self, **kwargs):
        self.streaming_tools = kwargs["request"].tools
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    id="call_test",
                    type="function",
                    function=DeltaFunctionCall(name="lookup", arguments='{"limit": 5}'),
                )
            ]
        )


def _make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "limit": {"type": "integer"},
                            },
                        },
                    },
                }
            ],
        }
    )


def test_run_output_parsers_uses_request_tools_for_batch_tool_parsing():
    engine = _ParsingHarness()
    parser = _RecordingToolParser()
    request = _make_request()
    rd = {
        "reasoning_parser_instance": None,
        "tool_parser_instance": parser,
        "tool_parser_request": request,
        "parser_previous_text": "",
        "parser_previous_token_ids": [],
        "accumulated_reasoning": "",
        "accumulated_content": "",
    }

    result = engine._run_output_parsers(
        rd=rd,
        accumulated_text="<function=lookup><parameter=limit>5</parameter></function>",
        delta_text="<function=lookup><parameter=limit>5</parameter></function>",
        token_ids=[],
        finished=True,
    )

    assert parser.batch_tools is not None
    assert parser.batch_tools[0].function.name == "lookup"
    assert result["tool_calls"] is not None
    assert result["tool_calls"][0].function.name == "lookup"


def test_run_output_parsers_uses_request_tools_for_streaming_tool_parsing():
    engine = _ParsingHarness()
    parser = _RecordingToolParser()
    request = _make_request()
    rd = {
        "reasoning_parser_instance": None,
        "tool_parser_instance": parser,
        "tool_parser_request": request,
        "parser_previous_text": "",
        "parser_previous_token_ids": [],
        "accumulated_reasoning": "",
        "accumulated_content": "",
    }

    result = engine._run_output_parsers(
        rd=rd,
        accumulated_text="<function=lookup>",
        delta_text="<function=lookup>",
        token_ids=[],
        finished=False,
    )

    assert parser.streaming_tools is not None
    assert parser.streaming_tools[0].function.name == "lookup"
    assert result["delta_tool_calls"] is not None
    assert result["delta_tool_calls"][0].function.name == "lookup"


def test_build_tool_parser_request_wraps_chat_template_tool_dicts():
    request = EngineIOMixin._build_tool_parser_request(
        prompt="hi",
        tools=[
            {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                },
            }
        ],
        tool_choice="auto",
    )

    assert request is not None
    assert request.tools is not None
    assert request.tools[0].function.name == "lookup"
    assert request.tool_choice == "auto"
