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

"""Pure-Python server regressions: admin route ordering and multi-prompt /v1/completions."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.inference.openai_api_modules import CompletionRequest
from fastapi import FastAPI, HTTPException, Request

pytestmark = pytest.mark.cpu_ok


def test_admin_keys_stats_route_is_not_shadowed_by_key_id_route() -> None:
    server = object.__new__(eSurgeApiServer)
    app = FastAPI()
    for endpoint in server._endpoints:
        app.add_api_route(path=endpoint.path, endpoint=endpoint.handler, methods=endpoint.methods)

    scope = {"type": "http", "method": "GET", "path": "/v1/admin/keys/stats"}
    from starlette.routing import Match

    first_full_match = next(route for route in app.routes if route.matches(scope)[0] == Match.FULL)
    assert first_full_match.endpoint == server.get_api_key_stats_endpoint

    scope["path"] = "/v1/admin/keys/some-key-id"
    first_full_match = next(route for route in app.routes if route.matches(scope)[0] == Match.FULL)
    assert first_full_match.endpoint == server.get_api_key_endpoint


def _output(request_id: str, text: str, tokens: int) -> RequestOutput:
    return RequestOutput(
        request_id=request_id,
        prompt="p",
        prompt_token_ids=[1],
        outputs=[CompletionOutput(index=0, text=text, token_ids=[1] * tokens, finish_reason="stop")],
        finished=True,
        accumulated_text=text,
        num_generated_tokens=tokens,
        processing_time=1.0,
    )


def _server() -> eSurgeApiServer:
    server = object.__new__(eSurgeApiServer)
    server._prepare_sampling_params = lambda *_: None
    server._record_api_key_usage = lambda *_: None
    server.metrics = SimpleNamespace(total_tokens_generated=0)
    server.thread_pool = ThreadPoolExecutor(max_workers=1)

    @asynccontextmanager
    async def slot(**_kwargs):
        yield

    server._acquire_generation_slot = slot
    return server


def test_completions_with_prompt_list_returns_one_choice_per_prompt() -> None:
    async def run():
        server = _server()
        seen = {}

        def generate(prompts, sampling_params, request_id=None, use_tqdm=False):
            seen["prompts"] = prompts
            seen["request_id"] = request_id
            return [_output(rid, f"out-{p}", 2) for p, rid in zip(prompts, request_id, strict=True)]

        esurge = SimpleNamespace(tokenizer=lambda p: {"input_ids": [1, 2, 3]}, generate=generate)
        raw_request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))
        request = CompletionRequest(model="m", prompt=["a", "b"])
        try:
            response = await server._handle_completion_response(request, esurge, ["a", "b"], "req", raw_request)
        finally:
            server.thread_pool.shutdown()
        return response, seen

    response, seen = asyncio.run(run())
    assert seen["prompts"] == ["a", "b"]
    assert len(set(seen["request_id"])) == 2
    assert [(c.index, c.text) for c in response.choices] == [(0, "out-a"), (1, "out-b")]
    assert response.usage.prompt_tokens == 6
    assert response.usage.completion_tokens == 4


def test_completions_rejects_empty_prompt_entries() -> None:
    async def run(prompt, stream):
        server = _server()
        server._extract_payload_api_keys = lambda *_: []
        server._get_adapter = lambda *_: SimpleNamespace(esurge=SimpleNamespace())
        server._ensure_request_max_tokens = lambda *_: 8
        server._authorize_request = lambda *_, **__: None
        request = CompletionRequest(model="m", prompt=prompt, stream=stream)
        try:
            return await server.completions(request, SimpleNamespace())
        finally:
            server.thread_pool.shutdown()

    with pytest.raises(HTTPException) as empty_error:
        asyncio.run(run(["a", ""], False))
    assert empty_error.value.status_code == 400


@pytest.mark.parametrize("num_choices", [1, 2])
def test_completions_streams_every_prompt_and_aggregates_usage(num_choices) -> None:
    async def run():
        server = _server()
        server._extract_payload_api_keys = lambda *_: []
        server._ensure_request_max_tokens = lambda *_: 8
        server._authorize_request = lambda *_, **__: None
        server._close_stream_queue = lambda *_: None
        seen = []

        def stream(prompt, sampling_params, request_id):
            seen.append((prompt, request_id))
            output = _output(request_id, f"out-{prompt}", 2 * num_choices)
            # Usage comes from the actual engine snapshot, which must agree
            # with this fixture's three-token prompt tokenizer.
            output.prompt_token_ids = [1, 2, 3]
            output.outputs = [
                CompletionOutput(index=i, text=f"out-{prompt}-{i}", token_ids=[1, 2], finish_reason="stop")
                for i in range(num_choices)
            ]
            yield output

        def start_task(generate, abort):
            queue = asyncio.Queue()
            for output in generate():
                queue.put_nowait(("data", output))
            queue.put_nowait(("end", None))
            return queue

        server._start_stream_task = start_task
        esurge = SimpleNamespace(
            tokenizer=lambda p: {"input_ids": [1, 2, 3]}, stream=stream, abort_request=lambda *_: None
        )
        server._get_adapter = lambda *_: SimpleNamespace(esurge=esurge)
        raw_request = Request({"type": "http", "headers": []})
        raw_request.is_disconnected = lambda: asyncio.sleep(0, result=False)
        request = CompletionRequest(model="m", prompt=["a", "b"], stream=True, n=num_choices)
        try:
            response = await server.completions(request, raw_request)
            events = [event async for event in response.body_iterator]
        finally:
            server.thread_pool.shutdown()
        return events, seen, server.metrics.total_tokens_generated

    events, seen, tokens_generated = asyncio.run(run())
    assert [prompt for prompt, _ in seen] == ["a", "b"]
    assert len({request_id for _, request_id in seen}) == 2
    assert events.count("data: [DONE]\n\n") == 1
    chunks = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
    assert len({chunk["id"] for chunk in chunks}) == 1
    texts = [(choice["index"], choice["text"]) for chunk in chunks for choice in chunk["choices"] if choice["text"]]
    assert texts == [
        (prompt_index * num_choices + choice_index, f"out-{prompt}-{choice_index}")
        for prompt_index, prompt in enumerate(("a", "b"))
        for choice_index in range(num_choices)
    ]
    assert chunks[-1]["usage"]["prompt_tokens"] == 6
    assert chunks[-1]["usage"]["completion_tokens"] == 4 * num_choices
    assert tokens_generated == 4 * num_choices


def test_closing_batch_stream_aborts_active_prompt_and_does_not_start_next() -> None:
    async def run():
        server = _server()
        closed_queues = []
        aborted = []
        started = []
        server._close_stream_queue = closed_queues.append

        def stream(prompt, sampling_params, request_id):
            started.append(prompt)
            output = _output(request_id, "partial", 1)
            output.finished = False
            yield output

        def start_task(generate, abort):
            queue = asyncio.Queue()
            for output in generate():
                queue.put_nowait(("data", output))
            return queue

        server._start_stream_task = start_task
        esurge = SimpleNamespace(tokenizer=lambda _: {"input_ids": [1]}, stream=stream, abort_request=aborted.append)
        request = CompletionRequest(model="m", prompt=["a", "b"], stream=True)
        raw_request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))
        try:
            response = await server._handle_completion_batch_streaming(request, esurge, ["a", "b"], "batch", raw_request)
            first_event = await anext(response.body_iterator)
            await response.body_iterator.aclose()
        finally:
            server.thread_pool.shutdown()
        return first_event, started, aborted, closed_queues

    event, started, aborted, closed_queues = asyncio.run(run())
    assert json.loads(event.removeprefix("data: "))["choices"][0]["text"] == "partial"
    assert started == ["a"]
    assert aborted == ["batch-0"]
    assert len(closed_queues) == 1
