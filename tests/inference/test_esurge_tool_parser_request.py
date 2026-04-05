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

import threading

from easydel.inference.esurge.mixins.io import EngineIOMixin
from easydel.inference.esurge.mixins.parsing import EngineParsingMixin
from easydel.inference.esurge.mixins.requests import EngineRequestsMixin
from easydel.inference.openai_api_modules import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from easydel.inference.sampling_params import SamplingParams


class _ParsingHarness(EngineParsingMixin):
    pass


class _RecordingToolParser:
    """Stub tool parser that records the ``tools`` list from each request.

    Always returns a fixed ``lookup`` tool call so tests can verify that the
    engine correctly wires tool-parser instances and forwards request metadata.

    Attributes:
        tokenizer: Tokenizer passed at construction (may be ``None``).
        batch_tools: Tools list captured by the last ``extract_tool_calls`` call.
        streaming_tools: Tools list captured by the last streaming extraction.
    """

    def __init__(self, tokenizer=None):
        """Initialize the recording tool parser.

        Args:
            tokenizer: Optional tokenizer instance (stored but unused).
        """
        self.tokenizer = tokenizer
        self.batch_tools = None
        self.streaming_tools = None

    def extract_tool_calls(self, _content, request):
        """Record ``request.tools`` and return a fixed tool-call result.

        Args:
            _content: Model output text (ignored).
            request: Request object whose ``tools`` field is captured.

        Returns:
            An ``ExtractedToolCallInformation`` with a single ``lookup`` call.
        """
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
        """Record ``request.tools`` and return a fixed streaming delta.

        Args:
            **kwargs: Must contain a ``request`` key whose ``tools`` field
                is captured.

        Returns:
            A ``DeltaMessage`` with a single ``lookup`` delta tool call.
        """
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


class _RecordingScheduler:
    """Minimal scheduler stub that records submitted requests for assertions.

    Attributes:
        requests: List of requests passed to ``add_request``, in order.
    """

    def __init__(self):
        self.requests = []

    def add_request(self, request):
        """Append *request* to the internal recording list.

        Args:
            request: The request object forwarded by the engine harness.
        """
        self.requests.append(request)


class _RequestsHarness(EngineRequestsMixin):
    """Lightweight ``EngineRequestsMixin`` implementation for unit tests.

    Provides the minimal state and stub methods required by the mixin so
    that ``add_request`` / tokenisation paths can be exercised without
    spinning up a real eSurge engine.

    Attributes:
        runner: Fake runner object exposing ``max_model_len``.
        scheduler: A ``_RecordingScheduler`` that captures forwarded requests.
        tokenizer: Stub tokenizer with a fixed ``eos_token_id``.
    """

    def __init__(self):
        self.runner = type("Runner", (), {"max_model_len": 128})()
        self.reserve_tokens = 0
        self.auto_truncate_prompt = False
        self.strict_context = False
        self.truncate_mode = "left"
        self.auto_cap_new_tokens = True
        self.prefer_preserve_prompt = True
        self.decode_truncated_prompt = False
        self._request_lock = threading.Lock()
        self._output_lock = threading.Lock()
        self._scheduler_lock = threading.Lock()
        self._request_events = {}
        self._active_requests = {}
        self._request_outputs = {}
        self._tool_parser_class = _RecordingToolParser
        self._reasoning_parser_class = None
        self.scheduler = _RecordingScheduler()
        self.tokenizer = type("Tokenizer", (), {"eos_token_id": 2})()
        self._eos_ids = []
        self._request_counter = 0

    def _touch_activity(self):
        """No-op activity timestamp update for the test harness.

        Returns:
            Always ``None``.
        """
        return None

    def _tokenize_prompt(self, request_id, prompt):
        """Return a fixed dummy token list, ignoring the actual prompt.

        Args:
            request_id: Ignored request identifier.
            prompt: Ignored prompt text.

        Returns:
            A hardcoded ``[1, 2, 3]`` token list.
        """
        del request_id, prompt
        return [1, 2, 3]

    def _tokenize_prompt_segments(self, prompt):
        """Return a fixed dummy token list for multi-segment prompts.

        Args:
            prompt: Ignored prompt segments.

        Returns:
            A hardcoded ``[1, 2, 3]`` token list.
        """
        del prompt
        return [1, 2, 3]

    def _info(self, *_args, **_kwargs):
        """No-op logger stub.

        Returns:
            Always ``None``.
        """
        return None


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


def test_add_request_keeps_tool_parser_for_single_request_without_request_object():
    engine = _RequestsHarness()

    engine._add_request(
        request_id="req-1",
        prompt="hi",
        sampling_params=SamplingParams(max_tokens=1),
        tool_parser_request=None,
    )

    request_state = engine._active_requests["req-1"]
    assert request_state["tool_parser_request"] is None
    assert request_state["tool_parser_instance"] is not None
