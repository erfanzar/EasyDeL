"""Pure-Python request-plane isolation, rollback, abort and heartbeat regressions."""

import logging
import queue
import socket
import threading
import time
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.distributed import request_plane, wire
from easydel.inference.esurge.distributed.remote_engine import RemoteEngineHandle
from easydel.inference.esurge.distributed.request_plane import (
    OriginRequestPlane,
    OwnerRequestPlane,
    RequestPlaneError,
)
from easydel.inference.esurge.distributed.zmq_coordinator import ZmqLeaderCoordinator, ZmqWorkerCoordinator
from easydel.inference.esurge.engine.registry import RequestRecord, RequestRegistry
from easydel.inference.esurge.engine_types import EngineCoreOutput, EngineCoreOutputs, FinishReason
from easydel.inference.sampling_params import SamplingParams


def _request(rid):
    return SimpleNamespace(
        request_id=rid,
        parent_request_id=None,
        sample_index=0,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
    )


class _OwnerChannel:
    world_size = 2

    def __init__(self):
        self.messages = []

    def set_plane_handler(self, handler):
        self.handler = handler

    def send_to_peer(self, identity, message, payload=None):
        self.messages.append((identity, message, payload))


def _owner(channel, scheduled, **kwargs):
    return OwnerRequestPlane(
        coordinator=channel,
        scheduler_submit=scheduled.extend,
        abort_request=lambda rid: None,
        apply_stop_strings=lambda hits: None,
        submit_local_outputs=lambda bundle: None,
        next_arrival_stamp=time.monotonic,
        info=logging.getLogger(__name__).info,
        **kwargs,
    )


@pytest.mark.cpu_ok
def test_crc16_colliding_clients_have_independent_outputs():
    channel = _OwnerChannel()
    scheduled = []
    plane = _owner(channel, scheduled)
    plane.ensure_started()
    identities = [b"client-1623", b"client-8000"]
    try:
        for identity in identities:
            channel.handler(
                identity,
                wire.Admit(request_id="r", origin_rank=-1, origin_counter=1, content_hash="abcdef", n_samples=1),
                wire.encode_payload([_request("r")]),
            )
        plane._queue.join()
        assert len(scheduled) == 2
        assert scheduled[0].request_id != scheduled[1].request_id
        plane.tee_outputs(
            {
                0: EngineCoreOutputs(
                    outputs=[
                        EngineCoreOutput(request_id=req.request_id, new_token_ids=[i], finish_reason=FinishReason.LENGTH)
                        for i, req in enumerate(scheduled)
                    ],
                    timestamp=time.perf_counter(),
                )
            }
        )
        plane._queue.join()
        updates = [
            (identity, wire.decode_payload(payload)[0])
            for identity, msg, payload in channel.messages
            if isinstance(msg, wire.OutputUpdate)
        ]
        assert [(identity, outputs[0].request_id, outputs[0].new_token_ids) for identity, outputs in updates] == [
            (identities[0], "r", [0]),
            (identities[1], "r", [1]),
        ]
        assert not plane.needs_tee
    finally:
        plane.shutdown()


@pytest.mark.cpu_ok
@pytest.mark.parametrize("failure", ["nack", "timeout", "send"])
def test_failed_batch_cancels_all_submitted_parents_and_ignores_late_ack(failure):
    class Channel:
        def __init__(self):
            self.messages = []

        def set_plane_handler(self, handler):
            self.handler = handler

        def send_to_leader(self, message, payload=None):
            self.messages.append(message)
            if isinstance(message, wire.AbortReq):
                return
            if failure == "send" and message.request_id == "b":
                raise OSError("transport failed after enqueue")
            if message.request_id == "b" or failure == "nack":
                self.handler(
                    wire.AdmitAck(
                        request_id=message.request_id,
                        ok=message.request_id != "a",
                        error="no capacity" if message.request_id == "a" else None,
                    ),
                    None,
                )

    channel = Channel()
    rendered = []
    plane = OriginRequestPlane(
        coordinator=channel,
        submit_outputs=rendered.append,
        alive=lambda: True,
        rank=1,
        admit_timeout_s=0.001,
        info=logging.getLogger(__name__).info,
    )
    with pytest.raises((RequestPlaneError, OSError)):
        plane.submit_remote([_request("a"), _request("b")])
    assert [msg.request_id for msg in channel.messages if isinstance(msg, wire.AbortReq)] == ["a", "b"]
    assert not plane.is_remote("a")
    assert not plane.is_remote("b")
    late = EngineCoreOutput(request_id="a", new_token_ids=[4], finish_reason=FinishReason.LENGTH)
    channel.handler(wire.AdmitAck(request_id="a", ok=True), wire.encode_payload(([late], None)))
    assert rendered == []
    assert not plane.is_remote("a")


@pytest.mark.cpu_ok
@pytest.mark.parametrize("abort_parent", [False, True])
def test_remote_child_abort_finishes_parent_and_signals_parent_waiter(abort_parent):
    handle = object.__new__(RemoteEngineHandle)
    registry = RequestRegistry()
    handle._registry = registry
    handle._plane = SimpleNamespace(notify_abort=lambda rid: True)
    resets = []
    handle._detokenizer_client = SimpleNamespace(reset=resets.append)
    output = SimpleNamespace(
        finished=False,
        update_seq=0,
        outputs=[SimpleNamespace(finish_reason=None), SimpleNamespace(finish_reason="length")],
    )
    registry.outputs["p"] = output
    registry.events["p"] = threading.Event()
    registry.records["p-0"] = RequestRecord(parent_request_id="p", sample_index=0)
    registry.records["p-1"] = RequestRecord(parent_request_id="p", sample_index=1)
    handle.abort_request("p" if abort_parent else "p-0")
    assert output.finished
    assert [item.finish_reason for item in output.outputs] == ["abort", "length"]
    assert output.update_seq == 1
    assert registry.events["p"].is_set()
    assert "p-0" not in registry.records
    assert "p-0" in resets
    if not abort_parent:
        assert "p-1" in registry.records


@pytest.mark.cpu_ok
def test_remote_child_abort_keeps_unfinished_sibling_running():
    handle = object.__new__(RemoteEngineHandle)
    handle._registry = RequestRegistry()
    handle._plane = SimpleNamespace(notify_abort=lambda rid: True)
    handle._detokenizer_client = SimpleNamespace(reset=lambda rid: None)
    output = SimpleNamespace(
        finished=False,
        update_seq=0,
        outputs=[SimpleNamespace(finish_reason=None), SimpleNamespace(finish_reason=None)],
    )
    handle._registry.outputs["p"] = output
    handle._registry.records["p-0"] = RequestRecord(parent_request_id="p", sample_index=0)
    handle._registry.records["p-1"] = RequestRecord(parent_request_id="p", sample_index=1)
    handle.abort_request("p-1")
    assert not output.finished
    assert [item.finish_reason for item in output.outputs] == [None, "abort"]
    handle.abort_request("p-0")
    assert output.finished
    assert output.update_seq == 2


@pytest.mark.cpu_ok
def test_retention_expires_during_continuously_nonempty_ingestion(monkeypatch):
    channel = _OwnerChannel()
    scheduled = []
    plane = _owner(channel, scheduled, retention_s=0.5)
    plane.admit_local([_request("req-0000000001")])
    now = [0.0]
    monkeypatch.setattr(request_plane, "time", SimpleNamespace(monotonic=lambda: now[0], perf_counter=time.perf_counter))

    class BusyQueue(queue.Queue):
        def get(self, *args, **kwargs):
            # Never idles; advance time independently of arrival volume.
            now[0] += 0.1
            return super().get(block=False)

    plane._queue = BusyQueue()
    terminal = EngineCoreOutput(request_id=scheduled[0].request_id, new_token_ids=[4], finish_reason=FinishReason.LENGTH)
    plane.tee_outputs({0: EngineCoreOutputs(outputs=[terminal], timestamp=0.0)})
    for _ in range(100):
        plane.tee_outputs({})
    plane._queue.put(request_plane._STOP)
    plane._loop()
    assert plane._queue.unfinished_tasks == 0
    assert not plane.needs_tee


@pytest.mark.cpu_ok
@pytest.mark.parametrize("async_step", [False, True])
def test_long_host_runner_work_uses_step_deadline_not_heartbeat_deadline(async_step):
    # Real loopback transport with non-computing runners: no JAX arrays/devices.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    output = SimpleNamespace(req_ids=["r"], sampled_token_ids=None)

    class Runner:
        def execute_model(self, scheduler_output):
            time.sleep(0.4)
            return output

        def execute_model_async(self, scheduler_output):
            return "handle"

        def wait_for_execution(self, handle):
            time.sleep(0.4)
            return output

    leader = ZmqLeaderCoordinator(
        Runner(),
        world_size=2,
        bind_host="127.0.0.1",
        control_port=port,
        auth_token="test",
        config_fingerprint="test",
        ready_timeout_s=5,
        step_timeout_s=3,
        heartbeat_interval_s=0.01,
        heartbeat_timeout_s=0.15,
        verify_digest_interval=0,
    )
    worker = ZmqWorkerCoordinator(
        Runner(),
        rank=1,
        world_size=2,
        leader_addr="127.0.0.1",
        control_port=port,
        auth_token="test",
        config_fingerprint="test",
        connect_timeout_s=5,
        heartbeat_interval_s=0.01,
        heartbeat_timeout_s=0.15,
    )
    stop = threading.Event()
    failures = []

    def replay():
        try:
            worker.start()
            worker.run_worker_loop(stop)
        except Exception as exc:
            failures.append(exc)
        finally:
            worker.shutdown()

    thread = threading.Thread(target=replay, daemon=True)
    thread.start()
    try:
        leader.start()
        if async_step:
            assert leader.drain(leader.execute_async({})) is output
        else:
            assert leader.execute_sync({}) is output
        time.sleep(0.3)
        leader.check_health()
        assert not failures
        assert thread.is_alive()
    finally:
        leader.shutdown()
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert not failures
