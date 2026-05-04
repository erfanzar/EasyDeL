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

from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.esurge.mixins.io import EngineIOMixin
from easydel.inference.esurge.mixins.utils import EngineUtilsMixin
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.inference.openai_api_modules import DeltaMessage
from easydel.inference.sampling_params import SamplingParams


class _StreamHarness(EngineIOMixin, EngineUtilsMixin):
    def __init__(self):
        self._request_lock = threading.Lock()
        self._output_lock = threading.Lock()
        self._request_events = {}
        self._request_outputs = {}
        self._scheduler_running = True
        self._max_request_outputs = None

    def _generate_request_id(self):
        return "req-1"

    def _tokenize_prompt(self, request_id, prompt):
        del request_id, prompt
        return [1]

    def _prepare_sampling_params_for_request(self, sampling_params, request_id=None, prompt=None):
        del request_id, prompt
        return sampling_params

    def _add_request(
        self,
        request_id,
        prompt,
        sampling_params,
        prompt_token_ids=None,
        tool_parser_request=None,
        defer_scheduler_enqueue=False,
    ):
        del sampling_params, tool_parser_request
        assert not defer_scheduler_enqueue
        event = threading.Event()
        event.set()
        with self._request_lock:
            self._request_events[request_id] = event
        with self._output_lock:
            self._request_outputs[request_id] = RequestOutput(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids or [],
                outputs=[CompletionOutput(index=0, text="", token_ids=[])],
                accumulated_text="visible",
                delta_text="visible",
                raw_accumulated_text="<think>one</think>visible",
                raw_delta_text="<think>one</think>visible",
                update_seq=1,
                finished=False,
            )

    def _ensure_scheduler_running(self, context=""):
        del context
        return None

    def _raise_if_scheduler_failed(self):
        return None

    def _recover_orphaned_request(self, request_id):
        del request_id
        return False

    def abort_request(self, request_id):
        with self._output_lock:
            self._request_outputs.pop(request_id, None)
        with self._request_lock:
            self._request_events.pop(request_id, None)

    def _track_finished_output(self, request_id):
        del request_id
        return None


def test_stream_debug_context_captures_key_shapes():
    output = RequestOutput(
        request_id="req-1",
        prompt="hello",
        prompt_token_ids=[1, 2],
        outputs=[CompletionOutput(index=0, text="abc", token_ids=[11, 12, 13])],
        accumulated_text="abcdef",
        delta_text="def",
        num_generated_tokens=3,
        delta_tool_calls=[{"index": 0, "function": {"arguments": "{}"}}],
    )
    delta_message = DeltaMessage(role="assistant", content="def", tool_calls=[{"index": 0}])

    context = eSurgeApiServer._build_stream_debug_context(
        endpoint="/v1/chat/completions",
        request_id="req-1",
        model="test-model",
        queue_kind="data",
        disconnected=False,
        output=output,
        previous_text="abc",
        current_text="abcdef",
        delta_text="def",
        previous_token_ids=[11, 12],
        current_token_ids=[11, 12, 13],
        delta_token_ids=[13],
        raw_delta_message=delta_message,
        delta_message=delta_message,
        delta_tool_calls_raw=[{"index": 0}],
        saw_tool_call_delta=True,
    )

    assert context["endpoint"] == "/v1/chat/completions"
    assert context["request_id"] == "req-1"
    assert context["model"] == "test-model"
    assert context["delta_text_len"] == 3
    assert context["previous_text_len"] == 3
    assert context["current_text_len"] == 6
    assert context["output_num_generated_tokens"] == 3
    assert context["output_primary_token_ids_len"] == 3
    assert context["delta_message_tool_calls_len"] == 1


def test_stream_debug_context_truncates_preview():
    text = "x" * 200
    preview = eSurgeApiServer._stream_debug_preview(text, max_chars=50)
    assert preview is not None
    assert preview.endswith("...")
    assert len(preview) == 53


def test_stream_debug_context_includes_stream_error_and_tools_shape():
    err = RuntimeError("boom")
    err.__stream_producer_traceback__ = "Traceback (most recent call last):\n  ...\nRuntimeError: boom\n"
    tools = [{"name": "lookup"}]

    context = eSurgeApiServer._build_stream_debug_context(
        endpoint="/v1/chat/completions",
        request_id="req-err",
        model="test-model",
        queue_kind="error",
        disconnected=False,
        stream_error=err,
        tools=tools,
    )

    assert context["queue_kind"] == "error"
    assert context["stream_error_type"] == "RuntimeError"
    assert context["stream_error_message"] == "boom"
    assert context["stream_error_producer_traceback"] is not None
    assert context["tools_type"] == "list"
    assert context["tools_len"] == 1
    assert context["first_tool_type"] == "dict"


def test_request_output_keeps_raw_and_parsed_text_separately():
    output = RequestOutput(
        request_id="req-raw",
        prompt="hello",
        prompt_token_ids=[1, 2],
        outputs=[CompletionOutput(index=0, text="answer", token_ids=[11, 12, 13])],
        accumulated_text="answer",
        delta_text="answer",
        raw_accumulated_text="<think>plan</think>answer",
        raw_delta_text="</think>answer",
    )

    assert output.accumulated_text == "answer"
    assert output.delta_text == "answer"
    assert output.raw_accumulated_text == "<think>plan</think>answer"
    assert output.raw_delta_text == "</think>answer"


def test_stream_snapshot_keeps_exact_raw_delta_text():
    harness = _StreamHarness()
    stream = harness.stream("hello", sampling_params=SamplingParams(max_tokens=4))

    first = next(stream)
    assert first.raw_delta_text == "<think>one</think>visible"

    with harness._output_lock:
        output = harness._request_outputs["req-1"]
        output.accumulated_text = "visible final"
        output.delta_text = " final"
        output.raw_accumulated_text = "<think>done</think>visible final"
        output.raw_delta_text = "</think> final"
        output.update_seq = 2
        output.finished = True
    with harness._request_lock:
        harness._request_events["req-1"].set()

    second = next(stream)
    assert second.raw_delta_text == "</think> final"


def test_stream_snapshot_does_not_replay_stale_raw_delta_text():
    harness = _StreamHarness()
    stream = harness.stream("hello", sampling_params=SamplingParams(max_tokens=4))

    first = next(stream)
    assert first.raw_delta_text == "<think>one</think>visible"

    with harness._output_lock:
        output = harness._request_outputs["req-1"]
        output.delta_text = ""
        output.raw_delta_text = "<think>one</think>visible"
        output.delta_reasoning_content = " reasoning"
        output.reasoning_content = "reasoning"
        output.update_seq = 2
        output.finished = True
    with harness._request_lock:
        harness._request_events["req-1"].set()

    second = next(stream)
    assert second.raw_delta_text == ""
    assert second.delta_reasoning_content == "reasoning"
