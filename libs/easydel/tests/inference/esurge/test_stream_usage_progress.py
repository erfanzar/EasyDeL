"""Admission accounting survives disconnects after observed streaming work (no arrays)."""

import asyncio
import json
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
from starlette.requests import ClientDisconnect

from . import test_cross_auth_permissions as auth_tests
from .test_cross_auth_permissions import _inference_server, _raw_request

pytestmark = pytest.mark.cpu_ok
auth_manager = auth_tests.auth_manager


def _server(manager):
    server = _inference_server(manager)
    server.metrics = SimpleNamespace(total_tokens_generated=0)
    server._prepare_sampling_params = lambda *_: None
    server._close_stream_queue = lambda *_: None
    server._mark_stream_failure = lambda: None
    server._build_stream_debug_context = lambda **_: {}

    @asynccontextmanager
    async def slot(**_):
        yield

    server._acquire_generation_slot = slot

    # Unlike a producer that drains its entire source, advance exactly one engine
    # snapshot per consumer pull, so interrupted-child expectations are exact.
    class Queue:
        def __init__(self, source):
            self.source = iter(source)

        async def get(self):
            try:
                return "data", next(self.source)
            except StopIteration:
                return "end", None
            except Exception as exc:
                return "error", exc

    server._start_stream_task = lambda generate, abort: Queue(generate())
    return server


async def _disconnect(response, mode, target):
    def matches(event):
        text = event.decode() if isinstance(event, bytes) else event
        return target in text

    if mode == "body-close":
        async for event in response.body_iterator:
            if matches(event):
                break
        else:
            pytest.fail("Target stream event was never emitted")
        await response.body_iterator.aclose()
    else:

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.body" and matches(message["body"]):
                if mode == "cancel":
                    raise asyncio.CancelledError
                raise OSError("consumer disconnected")

        with pytest.raises(asyncio.CancelledError if mode == "cancel" else ClientDisconnect):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    await response.body_iterator.aclose()


@pytest.mark.parametrize("mode", ["body-close", "send-error", "cancel"])
@pytest.mark.parametrize("terminal", [False, True])
def test_chat_disconnect_reconciles_before_terminal_or_partial_yield(auth_manager, mode, terminal):
    key, metadata = auth_manager.generate_api_key(name="chat", quota=QuotaConfig(max_total_requests=1))
    server = _server(auth_manager)

    class Engine:
        def iter_chat_completion_stream(self, *, usage_callback, **kwargs):
            usage_callback(3, 2)
            yield ChatCompletionStreamResponse(
                model="m",
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0, delta=DeltaMessage(content="observed"), finish_reason="stop" if terminal else None
                    )
                ],
                usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5) if terminal else None,
            )
            raise RuntimeError("must not advance after client closes")

        def abort_request(self, request_id):
            pass

    async def run():
        raw = _raw_request(key)
        raw.is_disconnected = _connected
        server._authorize_request(raw, model="m", requested_tokens=8)
        response = await server._handle_chat_streaming(
            ChatCompletionRequest(model="m", messages=[], stream=True), Engine(), [], "chat", raw
        )
        response = server._track_api_key_stream(response, raw)
        await _disconnect(response, mode, "observed")

    asyncio.run(run())
    current = auth_manager.get_key_by_id(metadata.key_id)
    assert (current.total_requests, current.total_prompt_tokens, current.total_completion_tokens) == (1, 3, 2)


async def _connected():
    return False


@pytest.mark.parametrize("mode", ["body-close", "send-error", "cancel", "engine-error"])
@pytest.mark.parametrize("batch", [False, True])
def test_completion_disconnect_accounts_inflight_batch_child(auth_manager, mode, batch):
    key, metadata = auth_manager.generate_api_key(name="completion", quota=QuotaConfig(max_total_requests=1))
    server = _server(auth_manager)

    class Engine:
        tokenizer = staticmethod(lambda _: {"input_ids": [1, 2, 3]})

        def stream(self, prompt, sampling_params, request_id):
            yield RequestOutput(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=[1, 2, 3],
                outputs=[
                    CompletionOutput(
                        index=0, text=prompt, token_ids=[4, 5], finish_reason="stop" if prompt == "first" else None
                    )
                ],
                finished=prompt == "first",
                accumulated_text=prompt,
                num_generated_tokens=2,
            )
            if prompt != "first":
                raise RuntimeError("engine interrupted")

        def abort_request(self, request_id):
            pass

    server._get_adapter = lambda _: SimpleNamespace(esurge=Engine())

    async def run():
        raw = _raw_request(key)
        raw.is_disconnected = _connected
        response = await server.completions(
            CompletionRequest(
                model="m", prompt=["first", "inflight"] if batch else "inflight", max_tokens=8, stream=True
            ),
            raw,
        )
        if mode == "engine-error":
            body = "".join([event async for event in response.body_iterator])
            assert "engine interrupted" in body
        else:
            await _disconnect(response, mode, "inflight")

    asyncio.run(run())
    current = auth_manager.get_key_by_id(metadata.key_id)
    factor = 2 if batch else 1
    assert (current.total_requests, current.total_prompt_tokens, current.total_completion_tokens) == (
        1,
        3 * factor,
        2 * factor,
    )


@pytest.mark.parametrize("mode", ["body-close", "send-error", "cancel"])
@pytest.mark.parametrize("terminal", [False, True])
def test_responses_disconnect_records_observed_progress(auth_manager, mode, terminal):
    key, metadata = auth_manager.generate_api_key(name="responses", quota=QuotaConfig(max_total_requests=1))
    server = _server(auth_manager)
    server._default_store_responses = False
    server._parse_responses_max_tokens = lambda *_: (8, 8)
    server._responses_payload_to_messages = lambda *_, **__: [{"role": "user", "content": "p"}]
    server._extract_responses_tools = lambda _: (None, None)
    server._create_sampling_params_from_responses = lambda *_, **__: None
    server._apply_extra_stops_to_sampling_params = lambda value: value
    server._responses_reasoning_summary_requested = lambda _: False

    class Engine:
        def iter_responses_stream(self, *, usage_callback, **kwargs):
            usage_callback(3, 2)
            obj = ResponsesResponse(
                id=kwargs["response_id"],
                model="m",
                created_at=123,
                output=[],
                status="completed" if terminal else "in_progress",
                usage=ResponsesUsage(input_tokens=3, output_tokens=2, total_tokens=5) if terminal else None,
            )
            yield StreamEventFrame(
                event="response.completed" if terminal else "response.created",
                payload=ResponseCompletedEvent(response=obj) if terminal else ResponseCreatedEvent(response=obj),
            )
            raise RuntimeError("must not advance after client closes")

        def abort_request(self, request_id):
            pass

    server._get_adapter = lambda _: SimpleNamespace(esurge=Engine())

    async def run():
        raw = _raw_request(key)
        raw.is_disconnected = _connected
        response = await server.responses(ResponsesRequest(model="m", input="p", stream=True, store=False), raw)
        await _disconnect(response, mode, "response.completed" if terminal else "response.created")

    asyncio.run(run())
    current = auth_manager.get_key_by_id(metadata.key_id)
    assert (current.total_requests, current.total_prompt_tokens, current.total_completion_tokens) == (1, 3, 2)


def test_terminal_chat_legacy_engine_without_progress_keyword(auth_manager):
    key, metadata = auth_manager.generate_api_key(name="legacy")
    server = _server(auth_manager)

    class Engine:
        def iter_chat_completion_stream(
            self, *, model, messages, tools, tool_choice, sampling_params, request_id, chat_template_kwargs
        ):
            yield ChatCompletionStreamResponse(
                model=model,
                choices=[ChatCompletionStreamResponseChoice(index=0, delta=DeltaMessage(), finish_reason="stop")],
                usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            )

        def abort_request(self, request_id):
            pass

    async def run():
        raw = _raw_request(key)
        raw.is_disconnected = _connected
        server._authorize_request(raw, model="m", requested_tokens=8)
        response = await server._handle_chat_streaming(
            ChatCompletionRequest(model="m", messages=[]), Engine(), [], "r", raw
        )
        response = server._track_api_key_stream(response, raw)
        event = await anext(response.body_iterator)
        assert json.loads(event.removeprefix("data: "))["usage"]["total_tokens"] == 5
        # Accounting must already be complete, without advancing toward [DONE].
        assert auth_manager.get_key_by_id(metadata.key_id).total_completion_tokens == 2
        await response.body_iterator.aclose()

    asyncio.run(run())
    assert auth_manager.get_key_by_id(metadata.key_id).total_requests == 1
