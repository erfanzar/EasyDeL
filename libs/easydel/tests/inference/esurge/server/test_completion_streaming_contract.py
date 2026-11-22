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

"""Pure-Python OpenAI text-completion streaming contract tests."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.esurge.server.api_server import eSurgeApiServer


def _snapshot(first: str, second: str, *, finished: bool = False) -> RequestOutput:
    return RequestOutput(
        request_id="req-1",
        prompt="hello",
        prompt_token_ids=[1, 2],
        outputs=[
            CompletionOutput(index=0, text=first, token_ids=[4], finish_reason="length" if finished else None),
            CompletionOutput(index=1, text=second, token_ids=[5], finish_reason="stop" if finished else None),
        ],
        finished=finished,
        accumulated_text=first,
        num_generated_tokens=4 if finished else 2,
    )


def test_completion_stream_emits_text_choices_and_individual_finish_reasons():
    async def run():
        server = object.__new__(eSurgeApiServer)
        server._prepare_sampling_params = lambda *_: None
        server._record_api_key_usage = lambda *_: None
        server.metrics = SimpleNamespace(total_tokens_generated=0)
        snapshots = iter([("data", _snapshot("a", "x")), ("data", _snapshot("ab", "xy", finished=True)), ("end", None)])

        class Queue:
            async def get(self):
                return next(snapshots)

        @asynccontextmanager
        async def slot(**_kwargs):
            yield

        server._acquire_generation_slot = slot
        server._start_stream_task = lambda *_: Queue()
        esurge = SimpleNamespace(tokenizer=lambda _: {"input_ids": [1, 2]})
        raw_request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))
        response = await server._handle_completion_streaming(
            SimpleNamespace(model="model"), esurge, "hello", "req-1", raw_request
        )
        frames = [
            json.loads(chunk.removeprefix("data: ")) for chunk in [part async for part in response.body_iterator][:-1]
        ]
        return frames, server.metrics.total_tokens_generated

    frames, counted = asyncio.run(run())
    assert all(frame["object"] == "text_completion" for frame in frames)
    assert [[choice["text"] for choice in frame["choices"]] for frame in frames] == [
        ["a", "x"],
        ["b", "y"],
        ["", ""],
    ]
    assert all("delta" not in choice for frame in frames for choice in frame["choices"])
    assert [choice["finish_reason"] for choice in frames[-1]["choices"]] == ["length", "stop"]
    assert frames[-1]["usage"]["completion_tokens"] == 4
    assert counted == 4


def test_completion_stream_recovers_each_samples_overlapping_parser_rewrite():
    async def run():
        server = object.__new__(eSurgeApiServer)
        server._prepare_sampling_params = lambda *_: None
        server._record_api_key_usage = lambda *_: None
        server.metrics = SimpleNamespace(total_tokens_generated=0)
        snapshots = iter(
            [
                ("data", _snapshot("a", "hello")),
                ("data", _snapshot("ab", "llo!", finished=True)),
                ("end", None),
            ]
        )

        class Queue:
            async def get(self):
                return next(snapshots)

        @asynccontextmanager
        async def slot(**_kwargs):
            yield

        server._acquire_generation_slot = slot
        server._start_stream_task = lambda *_: Queue()
        esurge = SimpleNamespace(tokenizer=lambda _: {"input_ids": [1, 2]})
        raw_request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))
        response = await server._handle_completion_streaming(
            SimpleNamespace(model="model"), esurge, "hello", "req-1", raw_request
        )
        return [
            json.loads(chunk.removeprefix("data: ")) for chunk in [part async for part in response.body_iterator][:-1]
        ]

    frames = asyncio.run(run())
    assert [[choice["index"], choice["text"]] for frame in frames[:-1] for choice in frame["choices"]] == [
        [0, "a"],
        [1, "hello"],
        [0, "b"],
        [1, "!"],
    ]
    assert [choice["finish_reason"] for choice in frames[-1]["choices"]] == ["length", "stop"]


def test_closing_completion_stream_aborts_unfinished_generation():
    async def run():
        server = object.__new__(eSurgeApiServer)
        server._prepare_sampling_params = lambda *_: None
        aborted = []
        esurge = SimpleNamespace(tokenizer=lambda _: {"input_ids": [1, 2]}, abort_request=aborted.append)

        class Queue:
            async def get(self):
                return "data", _snapshot("a", "x")

        @asynccontextmanager
        async def slot(**_kwargs):
            yield

        server._acquire_generation_slot = slot
        server._start_stream_task = lambda *_: Queue()
        raw_request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))
        response = await server._handle_completion_streaming(
            SimpleNamespace(model="model"), esurge, "hello", "req-1", raw_request
        )
        assert json.loads((await anext(response.body_iterator)).removeprefix("data: "))["choices"][0]["text"] == "a"
        await response.body_iterator.aclose()
        return aborted

    assert asyncio.run(run()) == ["req-1"]


def test_closing_chat_stream_aborts_unfinished_generation():
    async def run():
        server = object.__new__(eSurgeApiServer)
        server._prepare_sampling_params = lambda *_: None
        server.extract_tools = lambda **_kwargs: []
        aborted = []
        esurge = SimpleNamespace(abort_request=aborted.append)

        class Queue:
            async def get(self):
                return "data", SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))],
                    model_dump_json=lambda **_kwargs: '{"choices":[{"delta":{"content":"hello"}}]}',
                )

        @asynccontextmanager
        async def slot(**_kwargs):
            yield

        server._acquire_generation_slot = slot
        server._start_stream_task = lambda *_: Queue()
        raw_request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))
        response = await server._handle_chat_streaming(
            SimpleNamespace(model="model", chat_template_kwargs=None), esurge, [], "req-1", raw_request
        )
        assert "hello" in await anext(response.body_iterator)
        await response.body_iterator.aclose()
        return aborted

    assert asyncio.run(run()) == ["req-1"]
