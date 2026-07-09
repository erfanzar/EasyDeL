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

from easydel.inference.esurge.engine.admission import RequestAdmission
from easydel.inference.esurge.engine.output_pipeline import OutputPipeline
from easydel.inference.esurge.engine.registry import RequestRecord, RequestRegistry
from easydel.inference.esurge.esurge_engine import eSurge
from easydel.inference.openai_api_modules import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from easydel.inference.parsing import DelegatingParser
from easydel.inference.sampling_params import SamplingParams


def _make_pipeline():
    """Build an OutputPipeline with inert fakes for parser-drive tests."""
    return OutputPipeline(
        registry=RequestRegistry(),
        detokenizer_client=None,
        eos_token_ids=[],
        decode_interval_tokens=1,
        decode_interval_secs=0.0,
        on_stop_strings=lambda stops: None,
        on_activity=lambda: None,
        on_fatal=lambda exc, tb: None,
    )


class _RecordingToolParser:
    """Stub tool parser that records the ``tools`` list from each request.

    Always returns a fixed ``lookup`` tool call so tests can verify that the
    engine correctly wires tool-parser instances and forwards request metadata.
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.model_tokenizer = tokenizer
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


class _RecordingSubmit:
    """Minimal scheduler-submit stub that records submitted requests.

    Attributes:
        requests: List of requests passed to the submit callable, in order.
    """

    def __init__(self):
        self.requests = []

    def __call__(self, scheduler_requests):
        """Append the submitted requests to the internal recording list.

        Args:
            scheduler_requests: The request batch forwarded by admission.
        """
        self.requests.extend(scheduler_requests)


class _TokenizerClientStub:
    """Worker tokenizer-client stub returning a fixed dummy token list."""

    def tokenize(self, request_id, prompt):
        """Return a fixed ``[1, 2, 3]`` token list, ignoring the inputs."""
        del request_id, prompt
        return [1, 2, 3]


class _ContextConfigStub:
    """Context-config stub matching the fields admission reads."""

    auto_truncate_prompt = False
    strict_context = False
    truncate_mode = "left"
    auto_cap_new_tokens = True
    prefer_preserve_prompt = True
    decode_truncated_prompt = False


def _make_admission():
    """Build a RequestAdmission over recording fakes for add_request tests."""
    submit = _RecordingSubmit()
    admission = RequestAdmission(
        registry=RequestRegistry(),
        scheduler_submit=submit,
        tokenizer_client=_TokenizerClientStub(),
        tokenizer=type("Tokenizer", (), {"eos_token_id": 2})(),
        context_config=_ContextConfigStub(),
        reserve_tokens=0,
        max_model_len=128,
        tool_parser_class=_RecordingToolParser,
        reasoning_parser_class=None,
        sampling_params_callback=lambda: None,
        generation_config_dict={},
        primary_eos_token_id=None,
        eos_token_ids=[],
        extra_stops=[],
        ignore_stop_strings_in_reasoning=False,
        on_activity=lambda: None,
        info=lambda *args, **kwargs: None,
        callback_engine=None,
    )
    return admission, submit


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
    pipeline = _make_pipeline()
    parser = _RecordingToolParser()
    request = _make_request()
    rd = RequestRecord(**{
        "delegating_parser": DelegatingParser(
            reasoning_parser=None,
            tool_parser=parser,
            tool_request=request,
        ),
        "parser_previous_text": "",
        "parser_previous_token_ids": [],
    })

    result = pipeline._run_output_parsers(
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
    pipeline = _make_pipeline()
    parser = _RecordingToolParser()
    request = _make_request()
    rd = RequestRecord(**{
        "delegating_parser": DelegatingParser(
            reasoning_parser=None,
            tool_parser=parser,
            tool_request=request,
        ),
        "parser_previous_text": "",
        "parser_previous_token_ids": [],
    })

    result = pipeline._run_output_parsers(
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
    request = eSurge._build_tool_parser_request(
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


def test_add_request_creates_delegating_parser_for_single_request():
    admission, _submit = _make_admission()

    admission.add_request(
        request_id="req-1",
        prompt="hi",
        sampling_params=SamplingParams(max_tokens=1),
        tool_parser_request=None,
    )

    request_state = admission._active_requests["req-1"]
    dp = request_state.delegating_parser
    assert dp is not None
    assert isinstance(dp, DelegatingParser)
    assert dp.tool_parser is not None
    assert dp.tool_request is None


def test_add_request_can_defer_scheduler_enqueue_for_batching():
    admission, submit = _make_admission()

    scheduler_requests = admission.add_request(
        request_id="req-1",
        prompt="hi",
        sampling_params=SamplingParams(max_tokens=1),
        tool_parser_request=None,
        defer_scheduler_enqueue=True,
    )

    assert scheduler_requests is not None
    assert len(scheduler_requests) == 1
    assert scheduler_requests[0].request_id == "req-1"
    assert submit.requests == []
    assert "req-1" in admission._active_requests


def test_build_tool_parser_request_returns_none_without_tools():
    """When both tools and tool_choice are None, should return None."""
    result = eSurge._build_tool_parser_request(prompt="hi", tools=None, tool_choice=None)
    assert result is None


def test_build_tool_parser_request_with_tool_choice_only():
    """When tool_choice is set but tools is None, should still create a request."""
    result = eSurge._build_tool_parser_request(prompt="hi", tools=None, tool_choice="auto")
    assert result is not None
    assert result.tool_choice == "auto"
    assert result.tools is None


def test_build_tool_parser_request_skips_non_dict_tools():
    """Non-dict/non-model tools should be silently skipped."""
    result = eSurge._build_tool_parser_request(
        prompt="hi",
        tools=[
            123,  # not a dict
            "bad",  # not a dict
            {"name": "valid_func", "parameters": {}},  # dict without "function" key -> wraps
        ],
    )
    assert result is not None
    tools = result.tools
    assert tools is not None
    assert len(tools) == 1
    assert tools[0].function.name == "valid_func"


def test_build_tool_parser_request_normalizes_nested_function():
    """Tools with function.name structure should be preserved."""
    result = eSurge._build_tool_parser_request(
        prompt="hi",
        tools=[
            {
                "type": "function",
                "function": {"name": "search", "parameters": {"type": "object", "properties": {}}},
            }
        ],
    )
    assert result is not None
    tools = result.tools
    assert tools is not None
    assert tools[0].function.name == "search"


def test_run_output_parsers_no_delegating_parser():
    """When rd has no delegating_parser, should return passthrough result."""
    pipeline = _make_pipeline()
    rd = RequestRecord(**{
        "delegating_parser": None,
        "parser_previous_text": "",
        "parser_previous_token_ids": [],
    })
    result = pipeline._run_output_parsers(
        rd=rd,
        accumulated_text="hello world",
        delta_text="hello world",
        token_ids=[],
        finished=False,
    )
    assert result["accumulated_content"] == "hello world"
    assert result["delta_content"] == "hello world"
    assert result["tool_calls"] is None
    assert result["delta_reasoning"] is None
