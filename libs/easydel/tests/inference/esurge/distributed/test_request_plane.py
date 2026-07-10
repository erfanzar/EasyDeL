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
plane's core contracts.

Distinct (non-coalescable, explicit-id) requests: remote admissions are
renamed into the owner id space and scheduled with owner arrival stamps,
output deltas come back translated to the origin's ids with a locally
re-stamped clock, finishes retire the id maps on both sides, and aborts /
stop-string hits forward upstream.

Coalesced (auto-id, lockstep) requests: identical cross-rank admissions in
the same per-rank counter slot share ONE scheduled request no matter which
side admitted first; the owner-first path keeps the owner's local ids
(direct rendering, zero synthesis); late attaches receive a catch-up
replay of the group log; a subscriber's abort detaches only that
subscriber until the last one aborts the scheduled rows; and divergent
lockstep admissions fail loudly instead of silently double-scheduling.
"""

from __future__ import annotations

import re
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

# Explicit-style parent ids: never coalesce (the P1 "distinct" path).
DISTINCT_A = "job-alpha"
DISTINCT_B = "job-beta"
# Auto-generated multi-host id shapes: coalescable.
AUTO_1 = "req-0000000001"
AUTO_2 = "req-0000000002"


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


class _Sink:
    """Records synthesized output bundles fed to a rank's pipeline."""

    def __init__(self):
        self.lock = threading.Lock()
        self.bundles: list[dict] = []

    def submit(self, bundle):
        with self.lock:
            self.bundles.append(bundle)

    def all_outputs(self):
        with self.lock:
            return [out for bundle in self.bundles for co in bundle.values() for out in co.outputs]

    def all_finished(self):
        with self.lock:
            return {
                rid for bundle in self.bundles for co in bundle.values() for rid in (co.finished_requests or ())
            }


class _PlaneHarness:
    """Leader+worker coordinators over loopback with both plane halves attached."""

    def __init__(self, *, submit_error: str | None = None, admit_timeout_s: float = DEADLINE_S):
        port = _free_port()
        self.owner_engine = _OwnerEngine(submit_error=submit_error)
        self.origin_sink = _Sink()
        self.local_sink = _Sink()
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
            submit_local_outputs=self.local_sink.submit,
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

    def tee(self, *outputs, finished=None):
        """Feed one synthetic step bundle through the owner tee."""
        self.owner.tee_outputs(
            {0: EngineCoreOutputs(outputs=list(outputs), timestamp=1.0, finished_requests=finished)}
        )

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


# ---------------------------------------------------------------------------
# Distinct (non-coalescable) path
# ---------------------------------------------------------------------------


def test_remote_admit_renames_and_schedules():
    with _PlaneHarness() as plane:
        request = FakeEngineRequest(DISTINCT_A)
        plane.origin.submit_remote([request])

        assert plane.origin.is_remote(DISTINCT_A)
        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        [scheduled] = plane.owner_engine.submitted[0]
        assert re.match(r"^r1i[0-9a-f]{4}c1-", scheduled.request_id)
        assert scheduled.request_id != DISTINCT_A
        assert scheduled.parent_request_id is None
        assert scheduled.arrival_time == 1.0
        assert scheduled.prompt_token_ids == [1, 2, 3]
        assert plane.owner.needs_tee


def test_remote_admit_n2_renames_children_and_parent():
    with _PlaneHarness() as plane:
        requests = [
            FakeEngineRequest(f"{DISTINCT_A}-0", parent_request_id=DISTINCT_A, sample_index=0),
            FakeEngineRequest(f"{DISTINCT_A}-1", parent_request_id=DISTINCT_A, sample_index=1),
        ]
        plane.origin.submit_remote(requests)

        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        scheduled = plane.owner_engine.submitted[0]
        owner_parent = scheduled[0].parent_request_id
        assert re.match(r"^r1i[0-9a-f]{4}c1-", owner_parent)
        assert [req.request_id for req in scheduled] == [f"{owner_parent}-0", f"{owner_parent}-1"]
        assert all(req.parent_request_id == owner_parent for req in scheduled)
        # Distinct arrival stamps, both from the owner's counter.
        assert [req.arrival_time for req in scheduled] == [1.0, 2.0]


def test_output_tee_translates_ids_and_restamps_clock():
    with _PlaneHarness() as plane:
        plane.origin.submit_remote([FakeEngineRequest(DISTINCT_A)])
        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        owner_id = plane.owner_engine.submitted[0][0].request_id

        before = time.perf_counter()
        plane.tee(EngineCoreOutput(request_id=owner_id, new_token_ids=[11, 12]))
        _wait_until(lambda: len(plane.origin_sink.all_outputs()) == 1)
        [delta] = plane.origin_sink.all_outputs()
        assert delta.request_id == DISTINCT_A
        assert delta.new_token_ids == [11, 12]
        assert delta.finish_reason is None
        [bundle] = plane.origin_sink.bundles
        stamped = bundle[0].timestamp
        assert before <= stamped <= time.perf_counter(), "origin must re-stamp with its own clock"


def test_finish_delta_retires_both_sides():
    with _PlaneHarness() as plane:
        plane.origin.submit_remote([FakeEngineRequest(DISTINCT_A)])
        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        owner_id = plane.owner_engine.submitted[0][0].request_id

        plane.tee(EngineCoreOutput(request_id=owner_id, new_token_ids=[9], finish_reason=FinishReason.STOP))
        _wait_until(lambda: not plane.owner.needs_tee)
        _wait_until(lambda: not plane.origin.is_remote(DISTINCT_A))
        [delta] = plane.origin_sink.all_outputs()
        assert delta.finish_reason == FinishReason.STOP
        # Post-finish bundles for the id are not teed anymore.
        plane.tee(EngineCoreOutput(request_id=owner_id, new_token_ids=[1]))
        time.sleep(0.2)
        assert len(plane.origin_sink.all_outputs()) == 1


def test_abort_forwards_to_owner_and_retires():
    with _PlaneHarness() as plane:
        plane.origin.submit_remote([FakeEngineRequest(DISTINCT_A)])
        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        owner_id = plane.owner_engine.submitted[0][0].request_id

        assert plane.origin.notify_abort(DISTINCT_A)
        _wait_until(lambda: plane.owner_engine.aborted == [owner_id])
        _wait_until(lambda: not plane.owner.needs_tee)
        assert not plane.origin.is_remote(DISTINCT_A)
        # Unknown ids are not forwarded.
        assert not plane.origin.notify_abort("job-unknown")


def test_stop_hits_forward_translated():
    with _PlaneHarness() as plane:
        plane.origin.submit_remote([FakeEngineRequest(DISTINCT_A)])
        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        owner_id = plane.owner_engine.submitted[0][0].request_id

        rest = plane.origin.forward_stop_hits({DISTINCT_A: "</s>", "local-req": "x"})
        assert rest == {"local-req": "x"}, "non-remote hits stay with the local path"
        _wait_until(lambda: plane.owner_engine.stop_hits == [{owner_id: "</s>"}])


def test_admit_nack_raises_and_forgets():
    with _PlaneHarness(submit_error="scheduler exploded") as plane:
        with pytest.raises(RequestPlaneError, match="scheduler exploded"):
            plane.origin.submit_remote([FakeEngineRequest(DISTINCT_A)])
        assert not plane.origin.is_remote(DISTINCT_A)
        assert not plane.owner.needs_tee


# ---------------------------------------------------------------------------
# Coalesced (auto-id lockstep) path
# ---------------------------------------------------------------------------


def test_coalesce_owner_first_keeps_local_ids_and_fans_remote():
    with _PlaneHarness() as plane:
        # Owner admits first: scheduled under its OWN ids (direct rendering).
        plane.owner.admit_local([FakeEngineRequest(AUTO_1)])
        [scheduled] = plane.owner_engine.submitted[0]
        assert scheduled.request_id == AUTO_1, "owner-first groups keep the owner's native ids"
        assert plane.owner.needs_tee

        # Tokens flow before the remote twin attaches; the tee logs them.
        plane.tee(EngineCoreOutput(request_id=AUTO_1, new_token_ids=[5, 6]))
        _wait_until(lambda: plane.owner.stats["scheduled_groups"] == 1)

        # The remote twin attaches: no second scheduling, catch-up replay.
        plane.origin.submit_remote([FakeEngineRequest(AUTO_1)])
        assert len(plane.owner_engine.submitted) == 1, "coalesced admission must not schedule again"
        assert plane.owner.stats["coalesced_attaches"] == 1
        _wait_until(lambda: len(plane.origin_sink.all_outputs()) == 1)
        [replayed] = plane.origin_sink.all_outputs()
        assert replayed.request_id == AUTO_1
        assert replayed.new_token_ids == [5, 6]

        # Subsequent deltas fan to the remote subscriber; the local caller
        # renders directly (no synthesized local bundle).
        plane.tee(EngineCoreOutput(request_id=AUTO_1, new_token_ids=[7]))
        _wait_until(lambda: len(plane.origin_sink.all_outputs()) == 2)
        assert plane.origin_sink.all_outputs()[1].new_token_ids == [7]
        assert plane.local_sink.bundles == [], "local_direct subscribers render from their own registry"


def test_coalesce_remote_first_schedules_neutral_ids_and_replays_local():
    with _PlaneHarness() as plane:
        # Remote admits first: scheduled under a neutral q-id.
        plane.origin.submit_remote([FakeEngineRequest(AUTO_1)])
        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        [scheduled] = plane.owner_engine.submitted[0]
        assert scheduled.request_id.startswith("q0000000001-")

        plane.tee(EngineCoreOutput(request_id=scheduled.request_id, new_token_ids=[3, 4]))
        _wait_until(lambda: len(plane.origin_sink.all_outputs()) == 1)
        assert plane.origin_sink.all_outputs()[0].request_id == AUTO_1

        # The owner's own twin arrives late: attach + replay through the
        # owner's local pipeline in the owner's local id space.
        plane.owner.admit_local([FakeEngineRequest(AUTO_1)])
        assert len(plane.owner_engine.submitted) == 1
        assert plane.local_sink.all_outputs()[0].request_id == AUTO_1
        assert plane.local_sink.all_outputs()[0].new_token_ids == [3, 4]

        # Later deltas fan to BOTH subscribers now.
        plane.tee(EngineCoreOutput(request_id=scheduled.request_id, new_token_ids=[8]))
        _wait_until(lambda: len(plane.origin_sink.all_outputs()) == 2)
        _wait_until(lambda: len(plane.local_sink.all_outputs()) == 2)

        # Owner-local stop hits must translate onto the scheduled q-ids.
        translated = plane.owner.translate_local_stop_hits({AUTO_1: "STOP"})
        assert translated == {scheduled.request_id: "STOP"}


def test_coalesce_attach_after_finish_replays_terminal_state():
    with _PlaneHarness() as plane:
        plane.owner.admit_local([FakeEngineRequest(AUTO_1)])
        plane.tee(EngineCoreOutput(request_id=AUTO_1, new_token_ids=[1, 2], finish_reason=FinishReason.LENGTH))
        _wait_until(lambda: plane.owner.stats["scheduled_groups"] == 1)

        plane.origin.submit_remote([FakeEngineRequest(AUTO_1)])
        assert len(plane.owner_engine.submitted) == 1
        _wait_until(lambda: len(plane.origin_sink.all_outputs()) == 1)
        [replayed] = plane.origin_sink.all_outputs()
        assert replayed.new_token_ids == [1, 2]
        assert replayed.finish_reason == FinishReason.LENGTH
        assert not plane.origin.is_remote(AUTO_1), "terminal replay retires the origin-side ids"


def test_same_rank_identical_admissions_never_coalesce():
    with _PlaneHarness() as plane:
        plane.origin.submit_remote([FakeEngineRequest(AUTO_1)])
        plane.origin.submit_remote([FakeEngineRequest(AUTO_2, prompt_token_ids=(1, 2, 3))])
        _wait_until(lambda: len(plane.owner_engine.submitted) == 2)
        first = plane.owner_engine.submitted[0][0].request_id
        second = plane.owner_engine.submitted[1][0].request_id
        assert first.startswith("q0000000001-")
        assert second.startswith("q0000000002-"), "per-rank counters keep identical content distinct"
        assert plane.owner.stats["coalesced_attaches"] == 0


def test_lockstep_divergence_fails_loudly():
    with _PlaneHarness() as plane:
        # Remote coalescable slot 1 carries one prompt...
        plane.origin.submit_remote([FakeEngineRequest(AUTO_1, prompt_token_ids=(1, 2, 3))])
        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        # ...the owner's slot-1 admission carries DIFFERENT content.
        with pytest.raises(RequestPlaneError, match="divergence"):
            plane.owner.admit_local([FakeEngineRequest(AUTO_1, prompt_token_ids=(9, 9, 9))])


def test_subscriber_abort_detaches_until_last_one_aborts_rows():
    with _PlaneHarness() as plane:
        # Remote-first group, then the owner's twin attaches.
        plane.origin.submit_remote([FakeEngineRequest(AUTO_1)])
        _wait_until(lambda: len(plane.owner_engine.submitted) == 1)
        scheduled_id = plane.owner_engine.submitted[0][0].request_id
        plane.owner.admit_local([FakeEngineRequest(AUTO_1)])

        # Remote subscriber aborts: detach only — the local one still streams.
        assert plane.origin.notify_abort(AUTO_1)
        time.sleep(0.2)
        assert plane.owner_engine.aborted == [], "detach must not abort while subscribers remain"
        plane.tee(EngineCoreOutput(request_id=scheduled_id, new_token_ids=[4]))
        _wait_until(lambda: len(plane.local_sink.all_outputs()) == 1)

        # Last subscriber aborts: the scheduled rows go down.
        plane.owner.notify_local_abort(AUTO_1)
        _wait_until(lambda: plane.owner_engine.aborted == [scheduled_id])
        assert not plane.owner.needs_tee


def test_explicit_ids_bypass_coalescing_on_admit_local():
    with _PlaneHarness() as plane:
        plane.owner.admit_local([FakeEngineRequest(DISTINCT_B)])
        [scheduled] = plane.owner_engine.submitted[0]
        assert scheduled.request_id == DISTINCT_B
        assert not plane.owner.needs_tee, "explicit-id owner traffic is untracked (zero hot-path cost)"
        assert plane.owner.stats["scheduled_groups"] == 0
