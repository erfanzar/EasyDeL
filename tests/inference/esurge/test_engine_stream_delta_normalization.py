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

from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.esurge.mixins.io import EngineIOMixin
from easydel.inference.esurge.mixins.utils import EngineUtilsMixin
from easydel.inference.inference_engine_interface import BaseInferenceApiServer
from easydel.inference.openai_api_modules import (
    ChatCompletionRequest,
    ChatMessage,
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from easydel.inference.stream_protocol import iter_chat_completion_stream_responses, iter_responses_stream_frames
from easydel.inference.tools.tool_calling_mixin import ToolCallingMixin
from easydel.inference.typed_models import ResponsesFinalizationOptions


def test_jsonify_tool_calls_filters_non_mapping_entries():
    payload = BaseInferenceApiServer._jsonify_tool_calls(
        [
            "bad-entry",
            {"index": 0, "function": {"name": "lookup", "arguments": "{}"}},
            123,
        ]
    )
    assert payload == [{"index": 0, "function": {"name": "lookup", "arguments": "{}"}}]


def test_coerce_stream_delta_message_strips_text_when_tool_calls_are_present():
    delta = DeltaMessage(content="ok", tool_calls=["bad", {"index": 0, "function": {"arguments": "{}"}}])
    normalized = BaseInferenceApiServer._coerce_stream_delta_message(
        delta, fallback_text="fallback", default_role="assistant"
    )

    assert normalized is not None
    assert normalized.role == "assistant"
    assert normalized.content is None
    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].index == 0
    assert normalized.tool_calls[0].function is not None
    assert normalized.tool_calls[0].function.arguments == "{}"

    normalized_bad = BaseInferenceApiServer._coerce_stream_delta_message(object(), fallback_text="fallback")
    assert normalized_bad is not None
    assert normalized_bad.content == "fallback"


class _BrokenStreamingParser:
    def extract_tool_calls_streaming(self, **_kwargs):
        raise AttributeError("'str' object has no attribute 'items'")


class _StringStreamingParser:
    def extract_tool_calls_streaming(self, **_kwargs):
        return "delta-from-parser"


class _BatchToolParser:
    def extract_tool_calls(self, _response_text, _request):
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[
                ToolCall(
                    type="function",
                    function=FunctionCall(name="lookup", arguments="{}"),
                )
            ],
            content=None,
        )


class _Server(ToolCallingMixin):
    pass


class _IOStreamHarness(EngineIOMixin):
    def __init__(self, outputs):
        self.outputs = outputs
        self.chat_calls = []

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        return iter(self.outputs)


def test_tool_calling_mixin_streaming_falls_back_on_parser_exception():
    server = _Server()
    server.tool_parsers = {"test-model": _BrokenStreamingParser()}

    delta = server.extract_tool_calls_streaming(
        model_name="test-model",
        previous_text="",
        current_text="abc",
        delta_text="abc",
        request=None,
    )
    assert isinstance(delta, DeltaMessage)
    assert delta.content == "abc"


def test_tool_calling_mixin_streaming_coerces_string_delta():
    server = _Server()
    server.tool_parsers = {"test-model": _StringStreamingParser()}

    delta = server.extract_tool_calls_streaming(
        model_name="test-model",
        previous_text="old",
        current_text="new",
        delta_text="raw",
        request=None,
    )
    assert isinstance(delta, DeltaMessage)
    assert delta.content == "delta-from-parser"


def test_tool_calling_mixin_batch_uses_tool_calls_finish_reason():
    server = _Server()
    server.tool_parsers = {"test-model": _BatchToolParser()}
    request = ChatCompletionRequest(model="test-model", messages=[ChatMessage(role="user", content="hi")])

    message, finish_reason = server.extract_tool_calls_batch(
        response_text="<tool_call>ignored</tool_call>",
        request=request,
        model_name="test-model",
    )

    assert finish_reason == "tool_calls"
    assert message.tool_calls is not None
    assert message.tool_calls[0].function.name == "lookup"


def test_compute_delta_text_handles_overlap_and_shrink_without_warnings():
    # Overlap recovery.
    overlap_delta = BaseInferenceApiServer._compute_delta_text(
        current_text="world!",
        previous_text="hello world",
        fallback_delta="!",
    )
    assert overlap_delta == "!"

    # Snapshot rewrite shrink: prefer fallback if it is not a replay.
    shrink_delta = BaseInferenceApiServer._compute_delta_text(
        current_text="answer",
        previous_text="very long prior content",
        fallback_delta="answer",
    )
    assert shrink_delta == "answer"

    # Snapshot rewrite shrink without fallback should emit nothing.
    shrink_empty = BaseInferenceApiServer._compute_delta_text(
        current_text="answer",
        previous_text="very long prior content",
        fallback_delta="",
    )
    assert shrink_empty == ""


def test_compute_snapshot_delta_text_handles_shrink_without_reset_noise():
    delta = EngineUtilsMixin._compute_snapshot_delta_text(
        current_text="answer",
        previous_text="very long prior content",
        fallback_delta="answer",
    )
    assert delta == "answer"


def test_compute_snapshot_delta_text_equal_length_normalization():
    """When parser normalization rewrites content to equal-length different text,
    treat as benign realignment (no warning, return empty delta)."""
    delta = EngineUtilsMixin._compute_snapshot_delta_text(
        current_text="hello world!",  # 12 chars, different from prev
        previous_text="hello_world!",  # 12 chars
        fallback_delta="",
    )
    assert delta == ""

    delta2 = EngineUtilsMixin._compute_snapshot_delta_text(
        current_text="hello world!",
        previous_text="hello_world!",
        fallback_delta="!",  # suffix of prev, so treated as replay
    )
    assert delta2 == ""

    delta3 = EngineUtilsMixin._compute_snapshot_delta_text(
        current_text="hello world!",
        previous_text="hello_world!",
        fallback_delta="xyz",
    )
    assert delta3 == "xyz"


def test_compute_delta_text_equal_length_normalization():
    """Same test for BaseInferenceApiServer._compute_delta_text."""
    delta = BaseInferenceApiServer._compute_delta_text(
        current_text="hello world!",
        previous_text="hello_world!",
        fallback_delta="",
    )
    assert delta == ""

    delta3 = BaseInferenceApiServer._compute_delta_text(
        current_text="hello world!",
        previous_text="hello_world!",
        fallback_delta="xyz",
    )
    assert delta3 == "xyz"


def test_chat_stream_protocol_forwards_engine_deltas_without_recompute():
    outputs = iter(
        [
            RequestOutput(
                request_id="req-chat",
                prompt="hi",
                prompt_token_ids=[1, 2],
                outputs=[CompletionOutput(index=0, text="Hello!", token_ids=[10], finish_reason=None)],
                accumulated_text="Hello!",
                delta_text="Hello!",
                num_generated_tokens=1,
                tokens_per_second=5.0,
                processing_time=0.1,
                first_token_time=0.01,
            ),
            RequestOutput(
                request_id="req-chat",
                prompt="hi",
                prompt_token_ids=[1, 2],
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="Hello! How can I assist you today?",
                        token_ids=[10, 11, 12],
                        finish_reason="stop",
                    )
                ],
                finished=True,
                accumulated_text="Hello! How can I assist you today?",
                delta_text=" How can I assist you today?",
                num_generated_tokens=3,
                tokens_per_second=6.0,
                processing_time=0.2,
                first_token_time=0.01,
            ),
        ]
    )

    chunks = list(iter_chat_completion_stream_responses(outputs, model="demo-model"))

    assert [chunk.choices[0].delta.content for chunk in chunks[:-1]] == [
        "Hello!",
        " How can I assist you today?",
    ]
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage.completion_tokens == 3


def test_engine_io_iter_chat_completion_stream_executes_real_wrapper():
    engine = _IOStreamHarness(
        [
            RequestOutput(
                request_id="req-chat",
                prompt="hi",
                prompt_token_ids=[1, 2],
                outputs=[CompletionOutput(index=0, text="Hello!", token_ids=[10], finish_reason=None)],
                accumulated_text="Hello!",
                delta_text="Hello!",
                num_generated_tokens=1,
                tokens_per_second=5.0,
                processing_time=0.1,
                first_token_time=0.01,
            ),
            RequestOutput(
                request_id="req-chat",
                prompt="hi",
                prompt_token_ids=[1, 2],
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="Hello!",
                        token_ids=[10],
                        finish_reason="stop",
                    )
                ],
                finished=True,
                accumulated_text="Hello!",
                delta_text="",
                num_generated_tokens=1,
                tokens_per_second=5.0,
                processing_time=0.2,
                first_token_time=0.01,
            ),
        ]
    )

    chunks = list(
        engine.iter_chat_completion_stream(
            model="demo-model",
            messages=[{"role": "user", "content": "hi"}],
            request_id="req-chat",
        )
    )

    assert chunks[0].choices[0].delta.content == "Hello!"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert engine.chat_calls[0]["stream"] is True


def test_responses_stream_protocol_emits_reasoning_and_message_events_in_order():
    outputs = iter(
        [
            RequestOutput(
                request_id="req-response",
                prompt="hi",
                prompt_token_ids=[1, 2],
                outputs=[CompletionOutput(index=0, text="", token_ids=[10], reasoning_content="short")],
                delta_reasoning_content="short",
                reasoning_content="short",
                num_generated_tokens=1,
            ),
            RequestOutput(
                request_id="req-response",
                prompt="hi",
                prompt_token_ids=[1, 2],
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="Hello!",
                        token_ids=[10, 11],
                        finish_reason="stop",
                        reasoning_content="short",
                    )
                ],
                finished=True,
                accumulated_text="Hello!",
                delta_text="Hello!",
                reasoning_content="short",
                num_generated_tokens=2,
            ),
        ]
    )

    frames = list(
        iter_responses_stream_frames(
            outputs,
            response_id="resp_1",
            model="demo-model",
            include_reasoning_summary=True,
            final_response_overrides=ResponsesFinalizationOptions(store=False),
            created_at=123,
        )
    )

    event_names = [frame.event for frame in frames]
    assert event_names[0] == "response.created"
    assert event_names[-1] == "response.completed"
    assert event_names.count("response.output_item.added") == 2
    assert event_names.count("response.output_item.done") == 2
    assert "response.content_part.added" in event_names
    assert event_names.index("response.reasoning_summary_text.delta") < event_names.index(
        "response.reasoning_summary_text.done"
    )
    assert event_names.index("response.output_text.delta") < event_names.index("response.output_text.done")
    assert frames[-1].payload.response.output[-1].content[0].text == "Hello!"


def test_engine_io_iter_responses_stream_executes_real_wrapper():
    engine = _IOStreamHarness(
        [
            RequestOutput(
                request_id="req-response",
                prompt="hi",
                prompt_token_ids=[1, 2],
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="Hello!",
                        token_ids=[10],
                        finish_reason="stop",
                    )
                ],
                finished=True,
                accumulated_text="Hello!",
                delta_text="Hello!",
                num_generated_tokens=1,
            )
        ]
    )

    frames = list(
        engine.iter_responses_stream(
            response_id="resp-test",
            model="demo-model",
            messages=[{"role": "user", "content": "hi"}],
            request_id="req-response",
            created_at=123,
        )
    )

    assert frames[0].event == "response.created"
    assert frames[-1].event == "response.completed"
    assert frames[-1].payload.response.id == "resp-test"
    assert engine.chat_calls[0]["stream"] is True


def test_responses_stream_protocol_emits_function_call_argument_events():
    tool_delta = [{"index": 0, "id": "call_1", "function": {"name": "lookup", "arguments": '{"q":"AI"}'}}]
    tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q":"AI"}'}}]
    outputs = iter(
        [
            RequestOutput(
                request_id="req-tools",
                prompt="hi",
                prompt_token_ids=[1, 2],
                outputs=[
                    CompletionOutput(index=0, text="", token_ids=[10], finish_reason="tool_calls", tool_calls=tool_calls)
                ],
                finished=True,
                delta_tool_calls=tool_delta,
                tool_calls=tool_calls,
                num_generated_tokens=1,
            )
        ]
    )

    frames = list(
        iter_responses_stream_frames(
            outputs,
            response_id="resp_tools",
            model="demo-model",
            include_reasoning_summary=False,
            created_at=123,
        )
    )

    event_names = [frame.event for frame in frames]
    assert "response.function_call_arguments.delta" in event_names
    assert "response.function_call_arguments.done" in event_names
    assert frames[-1].payload.response.output[0].type == "function_call"
