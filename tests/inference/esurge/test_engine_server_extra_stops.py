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

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.inference.inference_engine_interface import BaseInferenceApiServer, ServerStatus
from easydel.inference.openai_api_modules import ChatCompletionRequest, CompletionRequest, ResponsesRequest
from easydel.inference.sampling_params import SamplingParams


def _make_server(extra_stops):
    server = object.__new__(eSurgeApiServer)
    server._extra_stops = eSurgeApiServer._normalize_stop_sequences(extra_stops)
    return server


def _make_request_output(
    *,
    request_id: str = "req-1",
    prompt: str = "hi",
    prompt_token_ids: list[int] | None = None,
    text: str = "pong",
    finish_reason: str | None = "stop",
) -> RequestOutput:
    return RequestOutput(
        request_id=request_id,
        prompt=prompt,
        prompt_token_ids=prompt_token_ids or [1, 2],
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[10, 11],
                finish_reason=finish_reason,
            )
        ],
        finished=True,
        accumulated_text=text,
        tokens_per_second=12.5,
        num_generated_tokens=2,
        first_token_time=0.02,
        processing_time=0.08,
    )


def _make_handler_server() -> eSurgeApiServer:
    server = object.__new__(eSurgeApiServer)
    server.thread_pool = ThreadPoolExecutor(max_workers=1)
    server.metrics = SimpleNamespace(
        total_tokens_generated=0,
        average_tokens_per_second=0.0,
    )
    server._active_requests = set()
    server._extra_stops = []
    server._generation_slots = asyncio.Queue(maxsize=1)
    server._generation_slots.put_nowait(0)
    server._max_generation_slots = 128
    server._overload_message = "Server is busy, please try again later"
    server._record_api_key_usage = lambda *_args, **_kwargs: None
    server._default_store_responses = False
    server._enable_response_store = False
    return server


class _DummyApiServer(BaseInferenceApiServer):
    async def chat_completions(self, request, raw_request):
        return JSONResponse({})

    async def completions(self, request, raw_request):
        return JSONResponse({})

    async def health_check(self, raw_request):
        return JSONResponse({})

    async def get_metrics(self, raw_request):
        return JSONResponse({})

    async def list_models(self, raw_request):
        return JSONResponse({})

    async def get_model(self, model_id, raw_request):
        return JSONResponse({})

    async def list_tools(self, raw_request):
        return JSONResponse({})

    async def execute_tool(self, request):
        return JSONResponse({})

    def _create_sampling_params(self, request):
        return SamplingParams(max_tokens=8)

    def _count_tokens(self, content: str, model_name: str | None = None) -> int:
        return len(content)

    async def generate(self, request):
        return JSONResponse({})


def test_normalize_stop_sequences_handles_common_inputs():
    assert eSurgeApiServer._normalize_stop_sequences(None) == []
    assert eSurgeApiServer._normalize_stop_sequences("<user>") == ["<user>"]
    assert eSurgeApiServer._normalize_stop_sequences(["", "<user>", "<user>", 42, None]) == ["<user>", "42"]


def test_apply_extra_stops_appends_and_deduplicates():
    server = _make_server(["<user>", "</assistant>"])
    sampling_params = SamplingParams(max_tokens=32, stop=["</assistant>", "DONE"])

    updated = server._apply_extra_stops_to_sampling_params(sampling_params)

    assert updated.stop == ["</assistant>", "DONE", "<user>"]


def test_apply_extra_stops_populates_empty_stop_list():
    server = _make_server("<user>")
    sampling_params = SamplingParams(max_tokens=16)

    updated = server._apply_extra_stops_to_sampling_params(sampling_params)

    assert updated.stop == ["<user>"]


def test_create_sampling_params_honors_special_token_flags():
    server = _make_server(None)
    request = ChatCompletionRequest.model_validate(
        {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "hi"}],
            "skip_special_tokens": "true",
            "spaces_between_special_tokens": "off",
        }
    )

    sampling_params = server._create_sampling_params(request)

    assert sampling_params.skip_special_tokens is True
    assert sampling_params.spaces_between_special_tokens is False


def test_api_server_defaults_generation_slots_to_runtime_request_cap(monkeypatch):
    captured = {}

    class FakeMetadata:
        def get_max_num_seqs(self):
            return 32

    class FakeSurge:
        def __init__(self, max_num_seqs):
            self.esurge_name = "fake-model"
            self.tokenizer = object()
            self.max_num_seqs = max_num_seqs
            self.runner = SimpleNamespace(metadata=FakeMetadata(), num_reqs_max_model_len=32)
            self.distributed_mode = False
            self.distributed_role = None

    class DummyAuthManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_base_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("easydel.inference.esurge.server.api_server.eSurge", FakeSurge)
    monkeypatch.setattr("easydel.inference.esurge.server.api_server.EnhancedApiKeyManager", DummyAuthManager)
    monkeypatch.setattr(BaseInferenceApiServer, "__init__", fake_base_init)

    eSurgeApiServer(FakeSurge(128), enable_function_calling=False, max_workers=8)

    assert captured["max_workers"] == 64
    assert captured["max_concurrent_generations"] == 32


def test_base_server_initializes_generation_slot_state_when_limit_is_set():
    server = _DummyApiServer(
        max_workers=1,
        enable_cors=False,
        enable_auth_ui=False,
        max_concurrent_generations=128,
    )
    try:
        assert server._generation_slots is not None
        assert server._max_generation_slots == 128
        assert server._generation_slots.qsize() == 128
    finally:
        server.thread_pool.shutdown(wait=True)


def test_generation_slot_503_logs_overload_context(monkeypatch):
    warnings = []

    def _capture_warning(message, *args):
        warnings.append(message % args if args else message)

    monkeypatch.setattr("easydel.inference.inference_engine_interface.logger.warning", _capture_warning)

    server = object.__new__(eSurgeApiServer)
    server._generation_slots = asyncio.Queue(maxsize=1)
    server._max_generation_slots = 1
    server._active_requests = {"req-a", "req-b", "req-c"}
    server.status = ServerStatus.READY
    server._overload_message = "Server is busy, please try again later"

    raw_request = SimpleNamespace(
        client=SimpleNamespace(host="203.0.113.9", port=4567),
        headers={"x-forwarded-for": "198.51.100.1"},
    )

    async def _attempt():
        async with server._acquire_generation_slot(
            endpoint="/v1/chat/completions",
            request_id="req-overload",
            model="demo-model",
            raw_request=raw_request,
            stream=True,
        ):
            raise AssertionError("Expected slot acquisition to fail")

    try:
        asyncio.run(_attempt())
    except HTTPException as exc:
        assert exc.status_code == 503
        assert exc.detail == "Server is busy, please try again later"
    else:
        raise AssertionError("Expected HTTPException when no generation slots are available")

    assert len(warnings) == 1
    warning = warnings[0]
    assert "all generation slots are busy" in warning
    assert "endpoint=/v1/chat/completions" in warning
    assert "request_id=req-overload" in warning
    assert "model=demo-model" in warning
    assert "stream=True" in warning
    assert "client_host=203.0.113.9" in warning
    assert "client_port=4567" in warning
    assert "forwarded_for=198.51.100.1" in warning
    assert "server_status=ready" in warning
    assert "active_http_requests=3" in warning
    assert "max_generation_slots=1" in warning
    assert "available_generation_slots=0" in warning
    assert "active_generation_slots=1" in warning


def test_non_stream_disconnect_watcher_aborts_until_done(monkeypatch):
    warnings = []

    class FakeRequest:
        def __init__(self):
            self.client = SimpleNamespace(host="203.0.113.10", port=7654)

        async def is_disconnected(self):
            return True

    class FakeSurge:
        def __init__(self):
            self.abort_calls = []

        def abort_request(self, request_id):
            self.abort_calls.append(request_id)

    def _capture_warning(message, *args):
        warnings.append(message % args if args else message)

    monkeypatch.setattr("easydel.inference.esurge.server.api_server.logger.warning", _capture_warning)

    server = object.__new__(eSurgeApiServer)
    surge = FakeSurge()
    done_event = asyncio.Event()

    async def _exercise():
        task = asyncio.create_task(
            server._abort_non_stream_request_on_disconnect(
                raw_request=FakeRequest(),
                esurge=surge,
                request_id="req-disconnect",
                endpoint="/v1/chat/completions",
                model="demo-model",
                done_event=done_event,
                poll_interval_s=0.001,
            )
        )
        await asyncio.sleep(0.01)
        done_event.set()
        await task

    asyncio.run(_exercise())

    assert len(surge.abort_calls) >= 1
    assert set(surge.abort_calls) == {"req-disconnect"}
    assert len(warnings) == 1
    assert "Client disconnected during non-stream request" in warnings[0]
    assert "endpoint=/v1/chat/completions" in warnings[0]
    assert "request_id=req-disconnect" in warnings[0]


def test_handler_cancel_abort_helper_aborts_engine_request(monkeypatch):
    warnings = []

    class FakeSurge:
        def __init__(self):
            self.abort_calls = []

        def abort_request(self, request_id):
            self.abort_calls.append(request_id)

    def _capture_warning(message, *args):
        warnings.append(message % args if args else message)

    monkeypatch.setattr("easydel.inference.esurge.server.api_server.logger.warning", _capture_warning)

    surge = FakeSurge()

    eSurgeApiServer._abort_request_after_handler_cancel(
        esurge=surge,
        request_id="req-cancelled",
        endpoint="/v1/chat/completions",
        model="demo-model",
    )

    assert surge.abort_calls == ["req-cancelled"]
    assert len(warnings) == 1
    assert "HTTP handler cancelled; aborting engine request." in warnings[0]
    assert "endpoint=/v1/chat/completions" in warnings[0]
    assert "request_id=req-cancelled" in warnings[0]


def test_health_check_reports_http_engine_and_slot_occupancy():
    server = object.__new__(eSurgeApiServer)
    server._authorize_request = lambda _raw_request: None
    server.metrics = SimpleNamespace(start_time=0.0, uptime_seconds=0.0)
    server.status = ServerStatus.READY
    server._active_requests = {"http-1", "http-2"}
    server._max_generation_slots = 8
    server._generation_slots = asyncio.Queue(maxsize=8)
    for slot in range(3):
        server._generation_slots.put_nowait(slot)
    server.adapters = {
        "demo-model": SimpleNamespace(
            esurge=SimpleNamespace(num_pending_requests=5, num_running_requests=3),
            get_model_info=lambda: {"type": "causal_lm", "max_model_len": 4096},
        )
    }

    response = asyncio.run(server.health_check(SimpleNamespace()))
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["active_requests"] == 2
    assert payload["active_http_requests"] == 2
    assert payload["pending_requests"] == 5
    assert payload["running_requests"] == 3
    assert payload["max_generation_slots"] == 8
    assert payload["available_generation_slots"] == 3
    assert payload["active_generation_slots"] == 5
    assert payload["models"]["demo-model"]["pending_requests"] == 5
    assert payload["models"]["demo-model"]["running_requests"] == 3


def test_chat_completion_handler_ignores_stale_slot_state():
    server = _make_handler_server()
    raw_request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1", port=8000))
    output = _make_request_output(request_id="chat-req", text="chat-ok")
    calls = {"chat": 0}

    async def _no_disconnect(**_kwargs):
        return None

    class FakeSurge:
        def chat(self, **kwargs):
            calls["chat"] += 1
            assert kwargs["request_id"] == "chat-req"
            return output

    server._prepare_sampling_params = lambda request, esurge: SamplingParams(max_tokens=8)
    server._abort_non_stream_request_on_disconnect = _no_disconnect
    server.extract_tools = lambda request: None

    request = ChatCompletionRequest.model_validate(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "ping"}],
        }
    )

    try:
        response = asyncio.run(
            server._handle_chat_completion(
                request,
                FakeSurge(),
                [{"role": "user", "content": "ping"}],
                "chat-req",
                raw_request,
            )
        )
    finally:
        server.thread_pool.shutdown(wait=True)

    assert calls["chat"] == 1
    assert response.choices[0].message.content == "chat-ok"


def test_completion_handler_ignores_stale_slot_state():
    server = _make_handler_server()
    raw_request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1", port=8000))
    output = _make_request_output(request_id="completion-req", text="completion-ok")
    calls = {"generate": 0}

    async def _no_disconnect(**_kwargs):
        return None

    class FakeSurge:
        @staticmethod
        def tokenizer(prompt):
            assert prompt == "ping"
            return {"input_ids": [1, 2]}

        def generate(self, prompt, sampling_params, request_id=None, use_tqdm=False):
            calls["generate"] += 1
            assert prompt == "ping"
            assert request_id == "completion-req"
            assert use_tqdm is False
            return [output]

    server._prepare_sampling_params = lambda request, esurge: SamplingParams(max_tokens=8)
    server._abort_non_stream_request_on_disconnect = _no_disconnect

    request = CompletionRequest.model_validate({"model": "demo-model", "prompt": "ping"})

    try:
        response = asyncio.run(
            server._handle_completion_response(request, FakeSurge(), "ping", "completion-req", raw_request)
        )
    finally:
        server.thread_pool.shutdown(wait=True)

    assert calls["generate"] == 1
    assert response.choices[0].text == "completion-ok"


def test_responses_handler_ignores_stale_slot_state():
    server = _make_handler_server()
    raw_request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1", port=8000))
    output = _make_request_output(request_id="resp-req", text="responses-ok")
    calls = {"chat": 0}

    async def _no_disconnect(**_kwargs):
        return None

    class FakeSurge:
        def chat(self, **kwargs):
            calls["chat"] += 1
            assert kwargs["request_id"].startswith("resp_")
            return output

    server._extract_payload_api_keys = lambda request: []
    server._get_adapter = lambda model: SimpleNamespace(esurge=FakeSurge())
    server._parse_responses_max_tokens = lambda payload, esurge: (8, 8)
    server._authorize_request = lambda *args, **kwargs: None
    server._responses_payload_to_messages = lambda payload, include_instructions=False: [
        {"role": "user", "content": "ping"}
    ]
    server._extract_responses_tools = lambda payload: (None, None)
    server._create_sampling_params_from_responses = lambda payload, max_tokens: SamplingParams(max_tokens=max_tokens)
    server._apply_extra_stops_to_sampling_params = lambda sampling_params: sampling_params
    server._responses_reasoning_summary_requested = lambda payload: False
    server._abort_non_stream_request_on_disconnect = _no_disconnect

    request = ResponsesRequest.model_validate({"model": "demo-model", "input": "ping", "store": False})

    try:
        response = asyncio.run(server.responses(request, raw_request))
    finally:
        server.thread_pool.shutdown(wait=True)

    assert calls["chat"] == 1
    assert response["status"] == "completed"
    assert response["usage"]["output_tokens"] == 2
