"""Real producer-queue admission accounting, without model/array computation."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.openai_api_modules import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    CompletionRequest,
    DeltaMessage,
    ResponsesRequest,
    UsageInfo,
)
from easydel.inference.stream_protocol import StreamEventFrame
from easydel.inference.typed_models import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponsesResponse,
    ResponsesUsage,
)
from easydel.workers.esurge.auth import QuotaConfig

from . import test_cross_auth_permissions as auth_tests
from .test_cross_auth_permissions import _inference_server, _raw_request
from .test_stream_usage_progress import _connected, _disconnect

pytestmark = pytest.mark.cpu_ok
auth_manager = auth_tests.auth_manager


def _threaded_server(manager):
    server = _inference_server(manager)
    server.metrics = SimpleNamespace(total_tokens_generated=0)
    server._prepare_sampling_params = lambda *_: None
    server._mark_stream_failure = lambda: None
    server._build_stream_debug_context = lambda **_: {}
    server._stream_pool = ThreadPoolExecutor(max_workers=1)
    server._stream_pool_lock = threading.Lock()
    server._max_stream_producers = 1
    server.stream_queue_maxsize = 1

    @asynccontextmanager
    async def slot(**_):
        yield

    server._acquire_generation_slot = slot
    return server


def _synchronize_teardown(response, target, producer_event):
    # Preserve the actual response and producer. Only gate consumer teardown
    # until the producer has reached the later snapshot/blocked next().
    source = response.body_iterator

    class SynchronizedBody:
        def __aiter__(self):
            return self

        async def __anext__(self):
            event = await anext(source)
            if target in event:
                assert await asyncio.to_thread(producer_event.wait, 5)
            return event

        async def aclose(self):
            await source.aclose()

    response.body_iterator = SynchronizedBody()


@pytest.mark.parametrize("mode", ["body-close", "send-error", "cancel"])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("timing", ["queued", "after-abort"])
def test_completion_read_ahead_and_inflight_abort_reconciled_once(auth_manager, mode, batch, timing):
    key, metadata = auth_manager.generate_api_key(name="read-ahead", quota=QuotaConfig(max_total_requests=1))
    server = _threaded_server(auth_manager)
    reached_later_snapshot = threading.Event()
    abort = threading.Event()
    engine_closed = threading.Event()

    class Engine:
        tokenizer = staticmethod(lambda _: {"input_ids": [1, 2, 3]})
        completion_tokens = 0

        def stream(self, prompt, sampling_params, request_id):
            def snapshot(count, *, finished=False):
                self.completion_tokens = count
                return RequestOutput(
                    request_id=request_id,
                    prompt=prompt,
                    prompt_token_ids=[1, 2, 3],
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=prompt if count == 2 else f"{prompt}-later",
                            token_ids=list(range(count)),
                            finish_reason="stop" if finished else None,
                        )
                    ],
                    finished=finished,
                    accumulated_text=prompt,
                    num_generated_tokens=count,
                )

            if prompt == "first":
                yield snapshot(2, finished=True)
                return
            try:
                yield snapshot(2)
                if timing == "after-abort":
                    reached_later_snapshot.set()
                    assert abort.wait(5), "Consumer did not abort the in-flight engine"
                    yield snapshot(5)
                else:
                    yield snapshot(5)
                    # Resumption proves the real bounded producer enqueued the
                    # newer snapshot before the consumer closes its body.
                    reached_later_snapshot.set()
                    assert abort.wait(5), "Consumer did not close the read-ahead stream"
            finally:
                engine_closed.set()

        def abort_request(self, request_id):
            if not batch or request_id.endswith("-1"):
                abort.set()

    engine = Engine()
    server._get_adapter = lambda _: SimpleNamespace(esurge=engine)

    async def run():
        raw = _raw_request(key)
        raw.is_disconnected = _connected
        response = await server.completions(
            CompletionRequest(
                model="m", prompt=["first", "inflight"] if batch else "inflight", max_tokens=8, stream=True
            ),
            raw,
        )

        _synchronize_teardown(response, '"text": "inflight"', reached_later_snapshot)
        await _disconnect(response, mode, '"text": "inflight"')
        assert engine_closed.is_set(), "Admission was released before the producer stopped"
        assert engine.completion_tokens == 5
        expected = (1, 6 if batch else 3, 7 if batch else 5)

        def assert_usage():
            current = auth_manager.get_key_by_id(metadata.key_id)
            assert (current.total_requests, current.total_prompt_tokens, current.total_completion_tokens) == expected

        assert_usage()
        # All cleanup routes can converge more than once; never charge twice.
        await response.body_iterator.aclose()
        server._release_api_key_reservation(raw)
        assert_usage()

    try:
        asyncio.run(run())
    finally:
        abort.set()
        server._stream_pool.shutdown(wait=True)


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("mode", ["body-close", "send-error", "cancel"])
@pytest.mark.parametrize("progress", ["queued-terminal", "callback-after-abort"])
def test_converted_stream_producer_usage_reconciled_before_release(auth_manager, endpoint, mode, progress):
    key, metadata = auth_manager.generate_api_key(name="converted", quota=QuotaConfig(max_total_requests=1))
    server = _threaded_server(auth_manager)
    reached_later_snapshot = threading.Event()
    abort = threading.Event()
    engine_closed = threading.Event()

    class LegacyEngine:
        def _outputs(self, endpoint, model, response_id):
            if endpoint == "chat":
                yield ChatCompletionStreamResponse(
                    model=model,
                    choices=[ChatCompletionStreamResponseChoice(index=0, delta=DeltaMessage(content="observed"))],
                )
                yield ChatCompletionStreamResponse(
                    model=model,
                    choices=[ChatCompletionStreamResponseChoice(index=0, delta=DeltaMessage(), finish_reason="stop")],
                    usage=UsageInfo(prompt_tokens=3, completion_tokens=5, total_tokens=8),
                )
            else:
                obj = ResponsesResponse(id=response_id, model=model, created_at=123, output=[], status="in_progress")
                yield StreamEventFrame(event="response.created", payload=ResponseCreatedEvent(response=obj))
                obj = obj.model_copy(
                    update={
                        "status": "completed",
                        "usage": ResponsesUsage(input_tokens=3, output_tokens=5, total_tokens=8),
                    }
                )
                yield StreamEventFrame(event="response.completed", payload=ResponseCompletedEvent(response=obj))

        def _stream(self, endpoint, model, response_id, usage_callback=None):
            outputs = self._outputs(endpoint, model, response_id)
            try:
                yield next(outputs)
                if progress == "callback-after-abort":
                    reached_later_snapshot.set()
                    assert abort.wait(5)
                    usage_callback(3, 5)
                else:
                    yield next(outputs)
                    reached_later_snapshot.set()
                    assert abort.wait(5)
            finally:
                engine_closed.set()

        # No callback extension is required to retain a legacy engine's queued
        # terminal counts; production-compatible preterminal callbacks are tested
        # separately below, including callbacks arriving in response to abort.
        def iter_chat_completion_stream(self, *, model, request_id, **kwargs):
            return self._stream("chat", model, request_id)

        def iter_responses_stream(self, *, model, response_id, **kwargs):
            return self._stream("responses", model, response_id)

        def abort_request(self, request_id):
            abort.set()

    class CallbackEngine(LegacyEngine):
        def iter_chat_completion_stream(self, *, model, request_id, usage_callback=None, **kwargs):
            return self._stream("chat", model, request_id, usage_callback)

        def iter_responses_stream(self, *, model, response_id, usage_callback=None, **kwargs):
            return self._stream("responses", model, response_id, usage_callback)

    engine = LegacyEngine() if progress == "queued-terminal" else CallbackEngine()
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
                ChatCompletionRequest(model="m", messages=[], stream=True), engine, [], "chat", raw
            )
            response = server._track_api_key_stream(response, raw)
            target = "observed"
        else:
            response = await server.responses(ResponsesRequest(model="m", input="p", stream=True, store=False), raw)
            target = "response.created"
        _synchronize_teardown(response, target, reached_later_snapshot)
        await _disconnect(response, mode, target)
        assert engine_closed.is_set(), "Admission was released before the producer stopped"
        current = auth_manager.get_key_by_id(metadata.key_id)
        assert (current.total_requests, current.total_prompt_tokens, current.total_completion_tokens) == (1, 3, 5)
        await response.body_iterator.aclose()
        server._release_api_key_reservation(raw)
        assert auth_manager.get_key_by_id(metadata.key_id).total_completion_tokens == 5

    try:
        asyncio.run(run())
    finally:
        abort.set()
        server._stream_pool.shutdown(wait=True)


def test_cancelling_body_close_waits_for_final_producer_snapshot(auth_manager):
    key, metadata = auth_manager.generate_api_key(name="cancel-close", quota=QuotaConfig(max_total_requests=1))
    server = _threaded_server(auth_manager)
    next_started = threading.Event()
    abort_received = threading.Event()
    allow_final_snapshot = threading.Event()
    engine_closed = threading.Event()

    class Engine:
        tokenizer = staticmethod(lambda _: {"input_ids": [1, 2, 3]})

        def stream(self, prompt, sampling_params, request_id):
            try:
                for count in [2, 5]:
                    if count == 5:
                        next_started.set()
                        assert abort_received.wait(5)
                        assert allow_final_snapshot.wait(5)
                    yield RequestOutput(
                        request_id=request_id,
                        prompt=prompt,
                        prompt_token_ids=[1, 2, 3],
                        outputs=[CompletionOutput(index=0, text=f"observed-{count}", token_ids=list(range(count)))],
                        accumulated_text=f"observed-{count}",
                        num_generated_tokens=count,
                    )
            finally:
                engine_closed.set()

        def abort_request(self, request_id):
            abort_received.set()

    server._get_adapter = lambda _: SimpleNamespace(esurge=Engine())

    async def run():
        raw = _raw_request(key)
        raw.is_disconnected = _connected
        response = await server.completions(CompletionRequest(model="m", prompt="p", max_tokens=8, stream=True), raw)
        assert "observed-2" in await anext(response.body_iterator)
        assert await asyncio.to_thread(next_started.wait, 5)
        closing = asyncio.create_task(response.body_iterator.aclose())
        try:
            assert await asyncio.to_thread(abort_received.wait, 5)
            closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done(), "Cancellation let admission cleanup overtake the engine producer"
            assert not raw.state.auth_reservation_completed
            allow_final_snapshot.set()
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert engine_closed.is_set()
            current = auth_manager.get_key_by_id(metadata.key_id)
            assert (current.total_requests, current.total_prompt_tokens, current.total_completion_tokens) == (1, 3, 5)
        finally:
            allow_final_snapshot.set()
            try:
                await closing
            except asyncio.CancelledError:
                pass

    try:
        asyncio.run(run())
    finally:
        abort_received.set()
        allow_final_snapshot.set()
        server._stream_pool.shutdown(wait=True)
