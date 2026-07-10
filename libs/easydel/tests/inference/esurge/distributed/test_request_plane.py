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

"""Real-ZMQ loopback tests for the unified request plane.

An owner plane on the leader coordinator and an origin plane on a worker
coordinator talk over ``tcp://127.0.0.1`` inside one process, with fake
engine callables recording what crosses the boundary. These lock the
plane's core contracts: remote admissions are renamed into the owner id
space and scheduled with owner arrival stamps, output deltas come back
translated to the origin's ids with a locally re-stamped clock, finishes
retire the id maps on both sides, and aborts / stop-string hits forward
upstream. The step-replication stream is untouched throughout.
"""

from __future__ import annotations

import threading
import time

import pytest
from easydel.inference.esurge.distributed.request_plane import (
    OriginRequestPlane,
    OwnerRequestPlane,
    RequestPlaneError,
)
from easydel.inference.esurge.distributed.zmq_coordinator import (
    ZmqLeaderCoordinator,
    ZmqWorkerCoordinator,
)
from easydel.inference.esurge.engine_types import EngineCoreOutput, EngineCoreOutputs, FinishReason

from .test_zmq_coordinator import AUTH, DEADLINE_S, FINGERPRINT, _free_port, _MockRunner, _wait_until


class FakeEngineRequest:
    """Pickle-able stand-in carrying the fields the plane touches."""

    def __init__(self, request_id, parent_request_id=None, prompt_token_ids=(1, 2, 3), sample_index=0):
        self.request_id = request_id
        self.parent_request_id = parent_request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.sampling_params = None
        self.sample_index = sample_index
        self.arrival_time = -1.0


class _OwnerEngine:
    """Thread-safe recorder for the owner plane's engine callables."""

    def __init__(self, *, submit_error: str | None = None):
        self.lock = threading.Lock()
        self.submitted: list[list] = []
        self.aborted: list[str] = []
        self.stop_hits: list[dict[str, str]] = []
        self.arrival = 0
        self.submit_error = submit_error

    def scheduler_submit(self, requests):
        if self.submit_error:
            raise RuntimeError(self.submit_error)
        with self.lock:
            self.submitted.append(list(requests))

    def abort_request(self, request_id):
        with self.lock:
            self.aborted.append(request_id)

    def apply_stop_strings(self, hits):
        with self.lock:
            self.stop_hits.append(dict(hits))

    def next_arrival_stamp(self):
        with self.lock:
            self.arrival += 1
            return float(self.arrival)


class _OriginSink:
    """Records synthesized output bundles fed to the origin's pipeline."""

    def __init__(self):
        self.lock = threading.Lock()
        self.bundles: list[dict] = []

    def submit(self, bundle):
        with self.lock:
            self.bundles.append(bundle)

    def all_outputs(self):
        with self.lock:
            return [out for bundle in self.bundles for co in bundle.values() for out in co.outputs]


class _PlaneHarness:
    """Leader+worker coordinators over loopback with both plane halves attached."""

    def __init__(self, *, submit_error: str | None = None, admit_timeout_s: float = DEADLINE_S):
        port = _free_port()
        self.owner_engine = _OwnerEngine(submit_error=submit_error)
        self.origin_sink = _OriginSink()
        self.leader = ZmqLeaderCoordinator(
            _MockRunner(),
            world_size=2,
            bind_host="127.0.0.1",
            control_port=port,
            auth_token=AUTH,
            config_fingerprint=FINGERPRINT,
            ready_timeout_s=DEADLINE_S,
            step_timeout_s=5.0,
            heartbeat_interval_s=0.2,
            heartbeat_timeout_s=DEADLINE_S,
            max_inflight_steps=4,
        )
        self.worker = ZmqWorkerCoordinator(
            _MockRunner(),
            rank=1,
            world_size=2,
            leader_addr="127.0.0.1",
            control_port=port,
            auth_token=AUTH,
            config_fingerprint=FINGERPRINT,
            connect_timeout_s=DEADLINE_S,
            heartbeat_interval_s=0.2,
            heartbeat_timeout_s=DEADLINE_S,
        )
        self.owner = OwnerRequestPlane(
            coordinator=self.leader,
            scheduler_submit=self.owner_engine.scheduler_submit,
            abort_request=self.owner_engine.abort_request,
            apply_stop_strings=self.owner_engine.apply_stop_strings,
            next_arrival_stamp=self.owner_engine.next_arrival_stamp,
            info=lambda *a, **k: None,
        )
        self.origin = OriginRequestPlane(
            coordinator=self.worker,
            submit_outputs=self.origin_sink.submit,
            alive=lambda: True,
            rank=1,
            admit_timeout_s=admit_timeout_s,
            info=lambda *a, **k: None,
        )
        self.stop_event = threading.Event()
        self.worker_error: list[BaseException] = []

        def _worker_body():
            try:
                self.worker.start()
                self.worker.run_worker_loop(self.stop_event)
            except BaseException as exc:
                self.worker_error.append(exc)

        self.worker_thread = threading.Thread(target=_worker_body, daemon=True)

    def __enter__(self):
        self.worker_thread.start()
        self.leader.start()
        self.owner.ensure_started()
        return self

    def __exit__(self, *exc_info):
        self.leader.shutdown("test teardown")
        self.stop_event.set()
        self.worker_thread.join(timeout=DEADLINE_S)
        self.worker.shutdown()
        self.owner.shutdown()
        assert not self.worker_thread.is_alive(), "worker replay thread failed to exit"


def test_remote_admit_renames_and_schedules():
    with _PlaneHarness() as plane:
        request = FakeEngineRequest("req-0000000001")
        plane.origin.submit_remote([request])

        assert plane.origin.is_remote("req-0000000001")
        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        [scheduled] = plane.owner_engine.submitted[0]
        assert scheduled.request_id.startswith("r1c1-")
        assert scheduled.request_id != "req-0000000001"
        assert scheduled.parent_request_id is None
        assert scheduled.arrival_time == 1.0
        assert scheduled.prompt_token_ids == [1, 2, 3]
        assert plane.owner.has_remote


def test_remote_admit_n2_renames_children_and_parent():
    with _PlaneHarness() as plane:
        requests = [
            FakeEngineRequest("req-0000000001-0", parent_request_id="req-0000000001", sample_index=0),
            FakeEngineRequest("req-0000000001-1", parent_request_id="req-0000000001", sample_index=1),
        ]
        plane.origin.submit_remote(requests)

        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        scheduled = plane.owner_engine.submitted[0]
        owner_parent = scheduled[0].parent_request_id
        assert owner_parent.startswith("r1c1-")
        assert [req.request_id for req in scheduled] == [f"{owner_parent}-0", f"{owner_parent}-1"]
        assert all(req.parent_request_id == owner_parent for req in scheduled)
        # Distinct arrival stamps, both from the owner's counter.
        assert [req.arrival_time for req in scheduled] == [1.0, 2.0]


def test_output_tee_translates_ids_and_restamps_clock():
    with _PlaneHarness() as plane:
        plane.origin.submit_remote([FakeEngineRequest("req-0000000007")])
        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        owner_id = plane.owner_engine.submitted[0][0].request_id

        before = time.perf_counter()
        plane.owner.tee_outputs(
            {
                0: EngineCoreOutputs(
                    outputs=[EngineCoreOutput(request_id=owner_id, new_token_ids=[11, 12])],
                    timestamp=123456.789,  # owner clock: must NOT survive the hop
                )
            }
        )
        _wait_until(lambda: len(plane.origin_sink.all_outputs()) == 1)
        [delta] = plane.origin_sink.all_outputs()
        assert delta.request_id == "req-0000000007"
        assert delta.new_token_ids == [11, 12]
        assert delta.finish_reason is None
        [bundle] = plane.origin_sink.bundles
        stamped = bundle[0].timestamp
        assert before <= stamped <= time.perf_counter(), "origin must re-stamp with its own clock"


def test_finish_delta_retires_both_sides():
    with _PlaneHarness() as plane:
        plane.origin.submit_remote([FakeEngineRequest("req-0000000002")])
        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        owner_id = plane.owner_engine.submitted[0][0].request_id

        plane.owner.tee_outputs(
            {
                0: EngineCoreOutputs(
                    outputs=[
                        EngineCoreOutput(
                            request_id=owner_id,
                            new_token_ids=[9],
                            finish_reason=FinishReason.STOP,
                        )
                    ],
                )
            }
        )
        _wait_until(lambda: not plane.owner.has_remote)
        _wait_until(lambda: not plane.origin.is_remote("req-0000000002"))
        [delta] = plane.origin_sink.all_outputs()
        assert delta.finish_reason == FinishReason.STOP
        # Post-finish bundles for the id are not teed anymore.
        plane.owner.tee_outputs(
            {0: EngineCoreOutputs(outputs=[EngineCoreOutput(request_id=owner_id, new_token_ids=[1])])}
        )
        time.sleep(0.2)
        assert len(plane.origin_sink.all_outputs()) == 1


def test_abort_forwards_to_owner_and_retires():
    with _PlaneHarness() as plane:
        plane.origin.submit_remote([FakeEngineRequest("req-0000000003")])
        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        owner_id = plane.owner_engine.submitted[0][0].request_id

        assert plane.origin.notify_abort("req-0000000003")
        _wait_until(lambda: plane.owner_engine.aborted == [owner_id])
        _wait_until(lambda: not plane.owner.has_remote)
        assert not plane.origin.is_remote("req-0000000003")
        # Unknown ids are not forwarded.
        assert not plane.origin.notify_abort("req-9999999999")


def test_stop_hits_forward_translated():
    with _PlaneHarness() as plane:
        plane.origin.submit_remote([FakeEngineRequest("req-0000000004")])
        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        owner_id = plane.owner_engine.submitted[0][0].request_id

        rest = plane.origin.forward_stop_hits({"req-0000000004": "</s>", "local-req": "x"})
        assert rest == {"local-req": "x"}, "non-remote hits stay with the local path"
        _wait_until(lambda: plane.owner_engine.stop_hits == [{owner_id: "</s>"}])


def test_admit_nack_raises_and_forgets():
    with _PlaneHarness(submit_error="scheduler exploded") as plane:
        with pytest.raises(RequestPlaneError, match="scheduler exploded"):
            plane.origin.submit_remote([FakeEngineRequest("req-0000000005")])
        assert not plane.origin.is_remote("req-0000000005")
        assert not plane.owner.has_remote
