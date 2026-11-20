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

"""The dp front under load: streamed parser fields, bounded memory, liveness, several front processes.

Pure Python: loopback sockets only, no array computation.
"""

import queue
import threading
import time
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.distributed import remote_engine, wire
from easydel.inference.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_ok


def _offline_handle(monkeypatch, on_admit=None):
    """A handle with the transport, workers and output pipeline stubbed; admission is real.

    ``on_admit(output)`` runs on each admitted request's output before it is marked finished.
    """

    class Tokenizer:
        eos_token_id = None
        chat_template = None

        def __call__(self, text, **kw):
            return {"input_ids": [ord(c) for c in text]}

        def decode(self, ids, **kw):
            return "".join(chr(t) for t in ids)

        def get_vocab(self):
            return {}

    tokenizer = Tokenizer()

    class Channel:
        alive = True
        queue_stats = wire.QueueStats()
        engine_spec = wire.EngineSpec(
            esurge_name="m", max_model_len=4096, max_num_seqs=4, reserve_tokens=1, tokenizer_source="t", page_size=16
        )

        def __init__(self, **kw):
            pass

        def start(self):
            pass

        def close(self):
            pass

        def set_plane_handler(self, handler):
            self.handler = handler

        def send_to_leader(self, message, payload=None):
            self.handler(wire.AdmitAck(request_id=message.request_id, ok=True), None)
            output = handle._registry.outputs[message.request_id]
            if on_admit is not None:
                on_admit(output)
            output.finished = True
            output.update_seq += 1
            handle._registry.signal(message.request_id)

    class Workers:
        def __init__(self, *a):
            pass

        def start(self, **kw):
            return SimpleNamespace(tokenize=lambda rid, prompt: tokenizer(prompt)["input_ids"]), SimpleNamespace()

        def shutdown(self):
            pass

    class Pipeline:
        def __init__(self, **kw):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(remote_engine, "_ClientChannel", Channel)
    monkeypatch.setattr(remote_engine, "WorkerManager", Workers)
    monkeypatch.setattr(remote_engine, "OutputPipeline", Pipeline)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr("easydel.inference.esurge.engine.admission.jax.process_count", lambda: 1)
    handle = remote_engine.RemoteEngineHandle("unused", 1234, auth_token="test")
    return handle


def test_streamed_snapshots_carry_reasoning_and_tool_calls(monkeypatch):
    call = {"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}

    def parsed(output):
        output.reasoning_content = "plan"
        output.delta_reasoning_content = "plan"
        output.tool_calls = [call]
        output.delta_tool_calls = [call]

    handle = _offline_handle(monkeypatch, on_admit=parsed)
    try:
        snapshots = list(handle.stream("hi", SamplingParams(max_tokens=1), request_id="s1"))
        final = snapshots[-1]
        assert final.finished
        assert final.reasoning_content == "plan"
        assert "".join(s.delta_reasoning_content or "" for s in snapshots) == "plan"
        assert final.tool_calls == [call] and final.delta_tool_calls == [call]
    finally:
        handle.close()


def test_finished_requests_do_not_stay_in_the_handle(monkeypatch):
    handle = _offline_handle(monkeypatch)
    try:
        handle.generate(["a", "b"], SamplingParams(max_tokens=1), request_id=["g1", "g2"])
        list(handle.stream("c", SamplingParams(max_tokens=1), request_id="s1"))
        assert not set(handle._registry.outputs) & {"g1", "g2", "s1"}
        assert not set(handle._registry.events) & {"g1", "g2", "s1"}
    finally:
        handle.close()


def test_queued_owner_heartbeats_keep_a_busy_client_alive():
    """A client whose IO thread was busy longer than the timeout must read queued heartbeats before judging."""
    import zmq

    ports: queue.Queue = queue.Queue()

    def owner():
        sock = zmq.Context.instance().socket(zmq.ROUTER)
        ports.put(sock.bind_to_random_port("tcp://127.0.0.1"))
        ident, _hello = sock.recv_multipart()
        sock.send_multipart([ident, wire.encode_message(wire.HelloOk(rank=0, engine_spec=wire.EngineSpec()))])
        sock.send_multipart([ident, wire.encode_message(wire.AdmitAck(request_id="x", ok=True))])
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            sock.send_multipart([ident, wire.encode_message(wire.Heartbeat(rank=0, ts=time.time()))])
            time.sleep(0.1)
        sock.close(linger=0)

    thread = threading.Thread(target=owner, daemon=True)
    thread.start()
    dead: list[str] = []
    channel = remote_engine._ClientChannel(
        leader_addr="127.0.0.1",
        control_port=ports.get(timeout=5),
        auth_token="t",
        client_id="client-under-load",
        connect_timeout_s=5.0,
        heartbeat_interval_s=0.1,
        heartbeat_timeout_s=0.5,
        on_dead=dead.append,
    )
    channel.set_plane_handler(lambda message, payload: time.sleep(1.0))  # one slow dispatch
    channel.start()
    try:
        time.sleep(1.8)
        assert dead == [] and channel.alive
    finally:
        channel.close()
        thread.join(timeout=5)


def test_several_front_processes_share_the_endpoint_port():
    from easydel.inference.esurge.serving import app
    from easydel.inference.esurge.serving.replicas import ReplicaPlan

    assert app.FRONT_PROCESSES > 1
    first = app.reuse_port_socket("127.0.0.1", 0)
    try:
        port = first.getsockname()[1]
        second = app.reuse_port_socket("127.0.0.1", port)
        second.close()
    finally:
        first.close()
    plan = ReplicaPlan(index=0, rank=0, num_processes=8, coordinator="h:1", leaders=("a",), role="front", front_index=3)
    assert ReplicaPlan.from_json(plan.to_json()).front_index == 3
