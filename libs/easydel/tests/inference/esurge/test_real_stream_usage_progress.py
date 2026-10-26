"""Account real local/remote protocol-wrapper progress through the server queue.

Only raw request snapshots are stubbed; production engine wrappers, converters,
producer queue, ASGI response cleanup and direct/worker auth accounting run.
No model construction or array computation is involved.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.distributed.remote_engine import RemoteEngineHandle
from easydel.inference.esurge.distributed.routed_engine import RoutedEngine
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput, eSurge
from easydel.inference.openai_api_modules import ChatCompletionRequest, ResponsesRequest
from easydel.workers.esurge.auth import QuotaConfig

from . import test_cross_auth_permissions as auth_tests
from .test_cross_auth_permissions import _raw_request
from .test_stream_usage_progress import _connected, _disconnect
from .test_stream_usage_read_ahead import _synchronize_teardown, _threaded_server

pytestmark = pytest.mark.cpu_ok
auth_manager = auth_tests.auth_manager


@pytest.mark.parametrize("wrapper", [eSurge, RemoteEngineHandle], ids=["local", "remote"])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("routed", [False, True], ids=["direct-engine", "routed-engine"])
@pytest.mark.parametrize("mode", ["body-close", "send-error", "cancel", "engine-error"])
def test_real_wrappers_account_invisible_partial_usage(auth_manager, wrapper, endpoint, routed, mode):
    key, metadata = auth_manager.generate_api_key(name="real-progress", quota=QuotaConfig(max_total_requests=1))
    server = _threaded_server(auth_manager)
    reached_inflight = threading.Event()
    abort = threading.Event()
    engine_closed = threading.Event()

    class Engine:
        iter_chat_completion_stream = wrapper.iter_chat_completion_stream
        iter_responses_stream = wrapper.iter_responses_stream

        def chat(self, **kwargs):
            def snapshot(count, delta):
                return RequestOutput(
                    request_id=kwargs["request_id"],
                    prompt="p",
                    prompt_token_ids=[1, 2, 3],
                    outputs=[CompletionOutput(index=0, text="observed", token_ids=list(range(count)))],
                    finished=False,
                    accumulated_text="observed",
                    delta_text=delta,
                    num_generated_tokens=count,
                )

            try:
                yield snapshot(2, "observed")
                reached_inflight.set()
                if mode != "engine-error":
                    assert abort.wait(5), "Consumer did not abort the raw engine stream"
                # No visible delta and no terminal usage frame: accounting must
                # come from the real wrapper -> converter progress callback.
                yield snapshot(5, "")
                if mode == "engine-error":
                    raise RuntimeError("raw engine interrupted")
            finally:
                engine_closed.set()

        def abort_request(self, request_id):
            abort.set()

    engine = RoutedEngine([Engine()]) if routed else Engine()
    server._get_adapter = lambda _: SimpleNamespace(esurge=engine)
    server._default_store_responses = False
    server._parse_responses_max_tokens = lambda *_: (8, 8)
    server._responses_payload_to_messages = lambda *_, **__: [{"role": "user", "content": "p"}]
    server._extract_responses_tools = lambda _: (None, None)
    server._create_sampling_params_from_responses = lambda *_, **__: None
    server._apply_extra_stops_to_sampling_params = lambda value: value
    server._responses_reasoning_summary_requested = lambda _: False

    async def run():
        raw = _raw_request(key)
        raw.is_disconnected = _connected
        if endpoint == "chat":
            server._authorize_request(raw, model="m", requested_tokens=8)
            response = await server._handle_chat_streaming(
                ChatCompletionRequest(model="m", messages=[], stream=True), engine, [], "real-progress", raw
            )
            response = server._track_api_key_stream(response, raw)
        else:
            response = await server.responses(ResponsesRequest(model="m", input="p", stream=True, store=False), raw)
        if mode == "engine-error":
            body = "".join([event async for event in response.body_iterator])
            assert "raw engine interrupted" in body
        else:
            _synchronize_teardown(response, "observed", reached_inflight)
            await _disconnect(response, mode, "observed")
        assert engine_closed.is_set(), "Usage reconciled before in-flight raw snapshot observation finished"
        current = auth_manager.get_key_by_id(metadata.key_id)
        assert (current.total_requests, current.total_prompt_tokens, current.total_completion_tokens) == (1, 3, 5)
        await response.body_iterator.aclose()
        server._release_api_key_reservation(raw)
        current = auth_manager.get_key_by_id(metadata.key_id)
        assert (current.total_requests, current.total_prompt_tokens, current.total_completion_tokens) == (1, 3, 5)

    try:
        asyncio.run(run())
    finally:
        abort.set()
        server._stream_pool.shutdown(wait=True)
