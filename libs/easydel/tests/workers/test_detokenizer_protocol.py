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

"""Exercise the detokenizer wire protocol without model or array computation."""

import threading

import pytest
import zmq
from easydel.workers.esurge.pipeline import worker_main
from easydel.workers.esurge.pipeline.zmq_workers import DetokenizerWorkerClient

pytestmark = pytest.mark.cpu_ok


class _CharacterTokenizer:
    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


@pytest.fixture
def endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(worker_main.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _CharacterTokenizer())
    uri = f"ipc://{tmp_path / 'd.sock'}"
    thread = threading.Thread(target=worker_main._detokenizer_worker, args=(uri, "unused", {}, 8), daemon=True)
    thread.start()
    yield uri
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_drain_then_decode_rebuilds_the_complete_text(endpoint):
    client = DetokenizerWorkerClient(endpoint)
    client._socket.setsockopt(zmq.RCVTIMEO, 5000)
    try:
        kwargs = {"finished": False, "skip_special_tokens": True}
        assert client.decode("r", [97], **kwargs).accumulated_text == "a"
        assert client.decode("r", [97, 98], **kwargs).delta_text == "b"
        client.drain()
        assert client.decode("r", [97, 98, 99], **kwargs).accumulated_text == "abc"
        assert client.decode("r", [97, 98, 99, 100], **kwargs).delta_text == "d"
    finally:
        client.shutdown()


@pytest.mark.parametrize("stream_first", [False, True])
def test_terminal_decode_emits_the_new_suffix(endpoint, stream_first):
    client = DetokenizerWorkerClient(endpoint, request_timeout=5)
    try:
        if stream_first:
            assert client.decode("final", [97], finished=False, skip_special_tokens=True).delta_text == "a"
        result = client.decode("final", [97, 98], finished=True, skip_special_tokens=True)
        assert result.accumulated_text == "ab"
        assert result.delta_text == ("b" if stream_first else "ab")
        assert result.finished
    finally:
        client.shutdown()


def test_unsynchronized_delta_gets_an_error_reply_and_worker_keeps_serving(endpoint):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    socket.connect(endpoint)
    try:
        message = {
            "cmd": "decode",
            "request_id": "lost",
            "tokens_delta": [98],
            "token_offset": 1,
            "finished": False,
            "skip_special_tokens": True,
        }
        socket.send_pyobj(message)
        assert socket.recv_pyobj()["status"] == "error"
        socket.send_pyobj({**message, "token_offset": 0, "tokens_delta": [97, 98], "tokens": [97, 98]})
        reply = socket.recv_pyobj()
        assert reply["status"] == "ok"
        assert reply["result"]["accumulated_text"] == "ab"
    finally:
        socket.send_pyobj({"cmd": "shutdown"})
        assert socket.recv_pyobj()["status"] == "ok"
        socket.close(0)
        context.term()
