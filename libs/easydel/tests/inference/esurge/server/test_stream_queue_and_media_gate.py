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

"""Pure-Python contracts: bounded/cancellable stream bridge and the server-local media gate."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from fastapi import HTTPException


def _server(*, maxsize: int = 4, max_workers: int = 2, allow_local_media_paths: bool = False) -> eSurgeApiServer:
    server = object.__new__(eSurgeApiServer)
    server.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    server.stream_queue_maxsize = maxsize
    server._allow_local_media_paths = allow_local_media_paths
    return server


def _close_and_join(server: eSurgeApiServer, queue) -> None:
    """Close an active bridge and wait for its dedicated producer to exit."""
    server._close_stream_queue(queue)
    queue._producer_future.result(timeout=5.0)
    assert queue._producer_future.done()


def test_stream_queue_applies_backpressure_and_close_releases_producer():
    server = _server(maxsize=4)
    produced: list[int] = []
    generator_closed = threading.Event()

    def stream_fn():
        try:
            for i in range(10_000):
                produced.append(i)
                yield i
        finally:
            generator_closed.set()

    async def run():
        queue = server._start_stream_task(stream_fn)
        try:
            first = await queue.get()
            await asyncio.sleep(0.2)  # let the producer run until the bounded queue is full
            return first, len(produced)
        finally:
            _close_and_join(server, queue)

    try:
        first, backlog = asyncio.run(run())
        assert first == ("data", 0)
        # Bounded: the producer stalls at roughly maxsize undelivered items instead of running ahead.
        assert backlog <= 4 + 3
        # close() stops the producer and closes the abandoned engine generator.
        assert generator_closed.is_set()
        assert len(produced) < 10_000
    finally:
        server.thread_pool.shutdown(wait=True)


def test_stream_queue_delivers_all_items_then_end():
    server = _server(maxsize=2)

    async def run():
        queue = server._start_stream_task(lambda: iter(range(20)))
        try:
            items = []
            while True:
                kind, payload = await queue.get()
                if kind == "end":
                    return items
                assert kind == "data"
                items.append(payload)
        finally:
            _close_and_join(server, queue)

    try:
        assert asyncio.run(run()) == list(range(20))
    finally:
        server.thread_pool.shutdown(wait=True)


def test_stream_queue_close_after_normal_completion_does_not_abort():
    # Every streaming endpoint closes its queue in ``finally``; a stream that ended normally
    # must not trigger ``on_abandon`` (an abort of an already-completed request).
    server = _server(maxsize=2)
    aborts: list[str] = []

    async def run():
        queue = server._start_stream_task(lambda: iter(range(3)), lambda: aborts.append("aborted"))
        try:
            items = []
            while True:
                kind, payload = await queue.get()
                if kind == "end":
                    return items
                items.append(payload)
        finally:
            _close_and_join(server, queue)

    try:
        assert asyncio.run(run()) == [0, 1, 2]
        assert aborts == []
    finally:
        server.thread_pool.shutdown(wait=True)


def test_stream_queue_forwards_producer_error():
    server = _server(maxsize=2)

    def stream_fn():
        yield 1
        raise RuntimeError("boom")

    async def run():
        queue = server._start_stream_task(stream_fn)
        try:
            kinds = []
            while True:
                kind, payload = await queue.get()
                kinds.append(kind)
                if kind == "error":
                    assert isinstance(payload, RuntimeError)
                if kind == "end":
                    return kinds
        finally:
            _close_and_join(server, queue)

    try:
        assert asyncio.run(run()) == ["data", "error", "end"]
    finally:
        server.thread_pool.shutdown(wait=True)


def test_stalled_stream_producer_does_not_starve_non_streaming_pool():
    server = _server(maxsize=1, max_workers=1)
    second_output_ready = threading.Event()

    def stream_fn():
        yield "first"
        second_output_ready.set()
        yield "second"
        yield "third"

    queue = server._start_stream_task(stream_fn)
    try:
        assert second_output_ready.wait(timeout=3.0)
        # No consumer drains the one-item queue, so the stream producer stays blocked.
        # The same one-worker pool still has to serve unrelated non-streaming work.
        task = server.thread_pool.submit(lambda: "non-streaming request served")
        assert task.result(timeout=2.0) == "non-streaming request served"
    finally:
        _close_and_join(server, queue)
        server.thread_pool.shutdown(wait=True)


def test_stream_producers_are_bounded_by_the_configured_cap():
    server = _server()
    server._max_stream_producers = 2
    counter_lock = threading.Lock()
    release = threading.Event()
    state = {"running": 0, "peak": 0}

    def stream_fn():
        with counter_lock:
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])
        release.wait(timeout=5.0)
        with counter_lock:
            state["running"] -= 1
        return iter(())

    queues = [server._start_stream_task(stream_fn) for _ in range(5)]
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with counter_lock:
                if state["running"] >= 2:
                    break
            time.sleep(0.01)
        time.sleep(0.2)  # give an unbounded implementation time to start the extra producers
        with counter_lock:
            assert state["running"] == 2
            assert state["peak"] == 2
        release.set()
        for queue in queues:
            queue._producer_future.result(timeout=5.0)
        assert state["peak"] == 2  # queued streams ran later, never concurrently past the cap
    finally:
        release.set()
        for queue in queues:
            server._close_stream_queue(queue)
        server._get_stream_pool().shutdown(wait=True)
        server.thread_pool.shutdown(wait=True)


def test_stream_queue_closed_before_producer_starts_never_calls_stream_fn():
    server = _server()
    server._max_stream_producers = 1
    calls: list[str] = []
    occupied = threading.Event()
    release_pool = threading.Event()

    def hold_pool_thread():
        occupied.set()
        release_pool.wait(timeout=5.0)

    # The single producer thread is busy, so the stream's producer stays queued until released.
    blocker = server._get_stream_pool().submit(hold_pool_thread)
    assert occupied.wait(timeout=3.0)
    queue = server._start_stream_task(
        lambda: calls.append("started") or iter(()), on_abandon=lambda: calls.append("abort")
    )
    try:
        queue.close()
        queue.close()
        assert not queue._producer_future.done()
        release_pool.set()
        blocker.result(timeout=5.0)
        queue._producer_future.result(timeout=5.0)
        assert calls == []
    finally:
        release_pool.set()
        _close_and_join(server, queue)
        server.thread_pool.shutdown(wait=True)


def test_stream_queue_aborts_immediately_and_retries_if_closed_during_stream_start():
    server = _server()
    entered = threading.Event()
    release = threading.Event()
    iterator_closed = threading.Event()
    next_calls: list[int] = []
    aborts: list[str] = []

    class IteratorAfterStart:
        def __next__(self):
            next_calls.append(1)
            return "unexpected output"

        def close(self):
            iterator_closed.set()

    def stream_fn():
        entered.set()
        assert release.wait(timeout=5.0)
        return IteratorAfterStart()

    queue = server._start_stream_task(stream_fn, on_abandon=lambda: aborts.append("aborted"))
    try:
        assert entered.wait(timeout=3.0)
        queue.close()
        queue.close()
        assert aborts == ["aborted"]
        assert not queue._producer_future.done()
        release.set()
        queue._producer_future.result(timeout=5.0)
        assert aborts == ["aborted", "aborted"]
        assert next_calls == []
        assert iterator_closed.is_set()
    finally:
        release.set()
        _close_and_join(server, queue)
        server.thread_pool.shutdown(wait=True)


def test_stream_queue_aborts_after_blocked_first_next_returns():
    server = _server()
    entered = threading.Event()
    release = threading.Event()
    iterator_closed = threading.Event()
    aborts: list[str] = []

    def stream_fn():
        try:
            entered.set()
            assert release.wait(timeout=5.0)
            yield "discard after close"
            raise AssertionError("advanced abandoned iterator")
        finally:
            iterator_closed.set()

    queue = server._start_stream_task(stream_fn, on_abandon=lambda: aborts.append("aborted"))
    try:
        assert entered.wait(timeout=3.0)
        queue.close()
        queue.close()
        assert aborts == ["aborted"]  # before blocked next() is released
        assert not queue._producer_future.done()
        release.set()
        queue._producer_future.result(timeout=5.0)
        assert aborts == ["aborted", "aborted"]
        assert iterator_closed.is_set()
    finally:
        release.set()
        _close_and_join(server, queue)
        server.thread_pool.shutdown(wait=True)


def test_close_stream_queue_tolerates_objects_without_close():
    eSurgeApiServer._close_stream_queue(object())


@pytest.mark.parametrize(
    "item",
    [
        {"type": "image_url", "image_url": {"url": "file:///etc/passwd"}},
        {"type": "image_url", "image_url": "/etc/passwd"},
        {"type": "image", "image": "file:///etc/hostname"},
        {"type": "image", "image": "relative/path.png"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {"type": "video", "video": "/etc/hostname"},
        {"type": "input_video", "video": "file:///tmp/clip.mp4"},
        {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}},
        # r14 S11-1: list-wrapped / nested references reach the HF video loader unchanged.
        {"type": "video", "video": ["/srv/private/clip.mp4"]},
        {"type": "video", "video": [["http://127.0.0.1:8000/private.mp4"]]},
        {"type": "video", "video": ["data:video/mp4;base64,AAAA", "/etc/hostname"]},
        {"type": "image", "image": ("file:///etc/passwd",)},
        {"type": "input_video", "input_video": ["/srv/private/clip.mp4"]},
    ],
)
def test_server_rejects_non_data_image_references_by_default(item):
    server = _server()
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}, item]}]
    with pytest.raises(HTTPException) as err:
        server._reject_local_media_references(messages)
    assert err.value.status_code == 400


def test_server_accepts_inline_data_urls_and_text():
    server = _server()
    messages = [
        {"role": "system", "content": "plain text"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "image", "image": "data:image/png;base64,AAAA"},
            ],
        },
    ]
    server._reject_local_media_references(messages)


def test_server_accepts_inline_lists_numeric_frames_and_url_detail():
    """The recursive gate (r14 S11-1) must still accept non-reference media payloads."""
    server = _server()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": ["data:video/mp4;base64,AAAA", "data:video/mp4;base64,BBBB"]},
                {"type": "video", "video": [[[0.0, 1.0], [2.0, 3.0]]]},  # numeric frames, no strings
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "auto"}},
            ],
        }
    ]
    server._reject_local_media_references(messages)


@pytest.mark.parametrize("allow_local", [False, True])
@pytest.mark.parametrize(
    "item",
    [
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
        {"type": "video_url", "video_url": "file:///tmp/clip.mp4"},
    ],
)
def test_server_rejects_video_url_the_loader_cannot_serve(item, allow_local):
    """The loader rejects every video_url item, so the gate must answer 400, not let it 500 (r15 S11-1)."""
    server = _server(allow_local_media_paths=allow_local)
    messages = [{"role": "user", "content": [item]}]
    with pytest.raises(HTTPException) as err:
        server._reject_local_media_references(messages)
    assert err.value.status_code == 400


def test_loader_decodes_inline_data_url_in_image_field():
    """An ``image`` field holding a data: URL passes the gate, so the loader must decode it (r15 S11-1)."""
    import base64
    import io

    pil_image = pytest.importorskip("PIL.Image")
    from easydel.inference.esurge.multimodal.manager import MultiModalManager

    buffer = io.BytesIO()
    pil_image.new("RGB", (3, 2), (10, 20, 30)).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    messages = [{"role": "user", "content": [{"type": "image", "image": url}]}]

    _server()._reject_local_media_references(messages)  # admitted by the gate
    manager = object.__new__(MultiModalManager)
    images, videos = manager.extract_media_from_messages(messages)
    assert videos == []
    assert len(images) == 1 and images[0].size == (3, 2)
    assert images[0].convert("RGB").getpixel((0, 0)) == (10, 20, 30)


def test_server_allows_local_paths_when_opted_in():
    server = _server(allow_local_media_paths=True)
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file:///tmp/a.png"}}]}]
    server._reject_local_media_references(messages)
