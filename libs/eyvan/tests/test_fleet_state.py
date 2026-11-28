# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from eyvan._internal.json_types import JsonObject
from eyvan.capacity import CapacityStatus
from eyvan.capacity.fleet_state import (
    ConflictError,
    FleetCellRecord,
    FleetLeaseBusyError,
    FleetStateSchemaError,
    FleetStateStore,
    GcsBackend,
    LocalBackend,
    StaleFenceError,
)


def store(path, owner: str) -> FleetStateStore:
    """Build a store that represents a *distinct* agent.

    The owner must be passed to the constructor, not assigned afterwards: it
    keys the process-wide mutual-exclusion lock, so a store relabelled after
    construction would still share a lock with every default-owned store and
    stop representing a separate agent at all.
    """

    return FleetStateStore(LocalBackend(path), owner=owner)


def test_workload_fence_is_monotonic_and_retains_predecessor(tmp_path) -> None:
    state = store(tmp_path / "fleet.json", "agent-a")
    state.claim_workload_fence("workload", 1, "token-1", remote_id="job-1", cell_id="cell-a")

    current = state.claim_workload_fence("workload", 2, "token-2", remote_id="job-2", cell_id="cell-b")

    assert current["predecessors"] == [
        {
            "generation": 1,
            "fencing_token": "token-1",
            "remote_id": "job-1",
            "cell_id": "cell-a",
        }
    ]
    with pytest.raises(StaleFenceError, match="older"):
        state.claim_workload_fence("workload", 1, "token-1", remote_id="job-1", cell_id="cell-a")
    with pytest.raises(StaleFenceError, match="fencing token"):
        state.claim_workload_fence("workload", 2, "other", remote_id="job-2", cell_id="cell-b")

    state.clear_workload_predecessors("workload", 2, "token-2")
    assert "predecessors" not in state.workload_fence("workload")


def test_fleet_lease_serializes_remote_side_effect_window(tmp_path) -> None:
    path = tmp_path / "fleet.json"
    first = store(path, "agent-a")
    second = store(path, "agent-b")
    entered = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with first.exclusive():
            entered.set()
            assert release.wait(timeout=2)

    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(timeout=2)
    with pytest.raises(FleetLeaseBusyError, match="lease is busy"):
        with second.exclusive():
            pass
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    with second.exclusive():
        assert second.lease_holder() == "agent-b"


def test_fleet_lease_renews_while_a_remote_side_effect_is_running(tmp_path) -> None:
    path = tmp_path / "fleet.json"
    # Renewal runs every ttl/3. A 0.12 s TTL left ~80 ms of scheduling slack,
    # which a loaded CI runner exceeded. The sleep still outlasts the TTL by
    # 2.5x, so the lease can only survive if renewal actually ran.
    first = FleetStateStore(LocalBackend(path), lease_ttl_s=1.0, owner="agent-a")
    second = FleetStateStore(LocalBackend(path), lease_ttl_s=1.0, owner="agent-b")

    with first.exclusive():
        time.sleep(2.5)
        assert first.lease_holder() == "agent-a"
        assert not second.acquire_lease()


def test_local_state_survives_reopen(tmp_path) -> None:
    path = tmp_path / "fleet.json"
    first = store(path, "agent-a")
    first.claim_workload_fence("workload", 4, "token-4", remote_id="job-4", cell_id="cell")

    second = store(path, "agent-b")

    fence = second.workload_fence("workload")
    assert fence is not None and fence["generation"] == 4


def test_foreign_registry_document_is_rejected_instead_of_co_mingled(tmp_path) -> None:
    path = tmp_path / "fleet.json"
    path.write_text(json.dumps({"version": 1, "clusters": {}, "lease": None}))
    state = FleetStateStore(LocalBackend(path))

    with pytest.raises(FleetStateSchemaError, match="not an Eyvan document"):
        state.workload_fence("workload")


def test_unknown_cell_fields_remain_forward_compatible_metadata(tmp_path) -> None:
    path = tmp_path / "fleet.json"
    path.write_text(
        json.dumps(
            {
                "schema": "eyvan.fleet-state",
                "version": 1,
                "cells": {"cell-a": {"name": "cell-a", "future_provider_field": {"enabled": True}}},
                "cell_tombstones": {},
                "workload_fences": {},
                "lease": None,
            }
        )
    )
    state = FleetStateStore(LocalBackend(path))

    def mark_healthy(current: FleetCellRecord) -> None:
        current.state = CapacityStatus.HEALTHY

    record = state.mutate_record("cell-a", mark_healthy)

    assert record.metadata["future_provider_field"] == {"enabled": True}
    persisted = json.loads(path.read_text())
    assert persisted["cells"]["cell-a"]["metadata"]["future_provider_field"] == {"enabled": True}


def test_malformed_persisted_fence_generation_is_not_coerced(tmp_path) -> None:
    path = tmp_path / "fleet.json"
    path.write_text(
        json.dumps(
            {
                "schema": "eyvan.fleet-state",
                "version": 1,
                "cells": {},
                "cell_tombstones": {},
                "workload_fences": {
                    "workload": {
                        "generation": "4",
                        "fencing_token": "token-4",
                        "remote_id": "job-4",
                        "cell_id": "cell-a",
                    }
                },
                "lease": None,
            }
        )
    )
    state = FleetStateStore(LocalBackend(path))

    with pytest.raises(FleetStateSchemaError, match="generation must be an integer"):
        state.claim_workload_fence("workload", 5, "token-5", remote_id="job-5", cell_id="cell-a")


def test_fencing_a_missing_cell_blocks_that_generation_without_touching_a_replacement(tmp_path) -> None:
    state = store(tmp_path / "fleet.json", "agent-a")

    assert state.fence_missing_record("cell-a", 3, "token-3")
    assert state.cell_tombstone("cell-a") == {"generation": 3, "fencing_token": "token-3"}
    assert not state.fence_missing_record("cell-a", 3, "token-3")

    def create(current: FleetCellRecord) -> None:
        current.generation = 4
        current.fencing_token = "token-4"

    replacement = state.mutate_or_create_record("cell-a", create, generation=4)

    assert not state.fence_missing_record("cell-a", 5, "token-5")
    assert state.get("cell-a") == replacement
    with pytest.raises(StaleFenceError, match="not newer"):
        state.mutate_or_create_record("cell-a", create, generation=3)


def test_gcs_state_uses_generation_cas_and_retries_conflicts() -> None:
    class FakeGcs:
        def __init__(self):
            self.document: JsonObject | None = None
            self.generation = 0
            self.conflict_once = True
            self.copy_attempts = 0

        def __call__(self, args):
            command = tuple(args)
            if command[:3] == ("storage", "objects", "describe"):
                if self.document is None:
                    return subprocess.CompletedProcess(command, 1, "", "404 not found")
                return subprocess.CompletedProcess(command, 0, json.dumps({"generation": self.generation}), "")
            if command[:2] == ("storage", "cat"):
                return subprocess.CompletedProcess(command, 0, json.dumps(self.document), "")
            if command[:2] == ("storage", "cp"):
                self.copy_attempts += 1
                if self.conflict_once:
                    self.conflict_once = False
                    return subprocess.CompletedProcess(command, 1, "", "412 precondition failed")
                expected = int(command[-1].split("=", 1)[1])
                assert expected == self.generation
                self.document = json.loads(Path(command[2]).read_text())
                self.generation += 1
                return subprocess.CompletedProcess(command, 0, "", "")
            raise AssertionError(command)

    gcs = FakeGcs()
    first = FleetStateStore(GcsBackend("gs://fleet/eyvan/state.json", runner=gcs))
    first.claim_workload_fence("workload", 3, "token-3", remote_id="job-3", cell_id="cell-a")

    reopened = FleetStateStore(GcsBackend("gs://fleet/eyvan/state.json", runner=gcs))

    fence = reopened.workload_fence("workload")
    assert fence is not None and fence["generation"] == 3
    assert gcs.copy_attempts == 2


def test_gcs_state_rejects_a_malformed_generation_token() -> None:
    def runner(args):
        command = tuple(args)
        if command[:3] == ("storage", "objects", "describe"):
            return subprocess.CompletedProcess(command, 0, json.dumps({"generation": "not-an-integer"}), "")
        raise AssertionError(f"unexpected GCS call after malformed generation: {command}")

    backend = GcsBackend("gs://fleet/eyvan/state.json", runner=runner)

    with pytest.raises(RuntimeError, match="invalid generation"):
        backend.read()


def test_gcs_default_transport_bounds_gcloud_calls(monkeypatch) -> None:
    calls: list[dict] = []

    def run(command, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 1, "", "404 not found")

    monkeypatch.delenv("EYVAN_GCLOUD_STORAGE_TIMEOUT_S", raising=False)
    monkeypatch.setattr(subprocess, "run", run)

    _document, generation = GcsBackend("gs://fleet/eyvan/state.json").read()

    assert generation == 0
    assert calls[0]["timeout"] == 30


def test_local_state_with_corrupt_json_surfaces_a_schema_error(tmp_path) -> None:
    path = tmp_path / "fleet.json"
    path.write_text("{not valid json")
    state = FleetStateStore(LocalBackend(path))

    with pytest.raises(FleetStateSchemaError, match="not valid JSON"):
        state.load()


def test_gcs_state_with_corrupt_document_json_surfaces_a_schema_error() -> None:
    def runner(args):
        command = tuple(args)
        if command[:3] == ("storage", "objects", "describe"):
            return subprocess.CompletedProcess(command, 0, json.dumps({"generation": 7}), "")
        if command[:2] == ("storage", "cat"):
            return subprocess.CompletedProcess(command, 0, "{not valid json", "")
        raise AssertionError(command)

    backend = GcsBackend("gs://fleet/eyvan/state.json", runner=runner)

    with pytest.raises(FleetStateSchemaError, match="not valid JSON"):
        backend.read()


def test_gcs_state_with_corrupt_metadata_json_surfaces_a_schema_error() -> None:
    def runner(args):
        command = tuple(args)
        if command[:3] == ("storage", "objects", "describe"):
            return subprocess.CompletedProcess(command, 0, "{not valid json", "")
        raise AssertionError(f"unexpected GCS call after corrupt metadata: {command}")

    backend = GcsBackend("gs://fleet/eyvan/state.json", runner=runner)

    with pytest.raises(FleetStateSchemaError, match="not valid JSON"):
        backend.read()


def test_from_uri_rejects_unknown_schemes_and_selects_backends(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported fleet-state URI scheme"):
        FleetStateStore.from_uri("s3://fleet/eyvan/state.json")
    with pytest.raises(ValueError, match="unsupported fleet-state URI scheme"):
        FleetStateStore.from_uri("https://example.com/fleet-state.json")
    with pytest.raises(ValueError, match="unsupported fleet-state URI scheme"):
        FleetStateStore.from_uri("file:///fleet/eyvan/state.json")

    local = FleetStateStore.from_uri(str(tmp_path / "fleet.json"))
    local.claim_workload_fence("workload", 1, "token-1", remote_id="job-1", cell_id="cell-a")

    reopened = FleetStateStore.from_uri(str(tmp_path / "fleet.json"))

    fence = reopened.workload_fence("workload")
    assert fence is not None and fence["generation"] == 1
    assert isinstance(FleetStateStore.from_uri("gs://fleet/eyvan/state.json").backend, GcsBackend)


def test_a_write_that_timed_out_after_landing_is_not_reported_as_a_failure() -> None:
    """Read back a timed-out conditional write to distinguish a committed mutation from failure."""

    class TimeoutAfterLandingGcs:
        def __init__(self):
            self.document: JsonObject | None = None
            self.generation = 0

        def __call__(self, args):
            command = tuple(args)
            if command[:3] == ("storage", "objects", "describe"):
                if self.document is None:
                    return subprocess.CompletedProcess(command, 1, "", "404 not found")
                return subprocess.CompletedProcess(command, 0, json.dumps({"generation": self.generation}), "")
            if command[:2] == ("storage", "cat"):
                return subprocess.CompletedProcess(command, 0, json.dumps(self.document), "")
            if command[:2] == ("storage", "cp"):
                # The object accepts the write, then the client gives up waiting.
                self.document = json.loads(Path(command[2]).read_text())
                self.generation += 1
                raise subprocess.TimeoutExpired(command, 30.0)
            raise AssertionError(command)

    gcs = TimeoutAfterLandingGcs()
    store = FleetStateStore(GcsBackend("gs://fleet/eyvan/state.json", runner=gcs))

    store.claim_workload_fence("workload", 3, "token-3", remote_id="job-3", cell_id="cell-a")

    fence = FleetStateStore(GcsBackend("gs://fleet/eyvan/state.json", runner=gcs)).workload_fence("workload")
    assert fence is not None and fence["generation"] == 3


def test_a_write_that_timed_out_without_landing_is_unavailable_not_a_conflict() -> None:
    """Nothing conflicted, so a caller must not treat this as a lost race.

    ``ConflictError`` means another writer won and an immediate retry is correct.
    An unreachable store means the caller learned nothing and must conclude
    nothing about the cells the document describes.
    """

    from eyvan.capacity.fleet_state import FleetStateUnavailableError

    class NeverLandsGcs:
        def __call__(self, args):
            command = tuple(args)
            if command[:3] == ("storage", "objects", "describe"):
                return subprocess.CompletedProcess(command, 1, "", "404 not found")
            if command[:2] == ("storage", "cat"):
                return subprocess.CompletedProcess(command, 1, "", "404 not found")
            if command[:2] == ("storage", "cp"):
                raise subprocess.TimeoutExpired(command, 30.0)
            raise AssertionError(command)

    store = FleetStateStore(GcsBackend("gs://fleet/eyvan/state.json", runner=NeverLandsGcs()))

    with pytest.raises(FleetStateUnavailableError):
        store.claim_workload_fence("workload", 3, "token-3", remote_id="job-3", cell_id="cell-a")
    assert not issubclass(FleetStateUnavailableError, ConflictError)


def test_an_unreadable_store_does_not_claim_the_write_landed() -> None:
    """Guessing "it probably worked" would hand out a fence nobody holds."""

    from eyvan.capacity.fleet_state import FleetStateUnavailableError

    class BlindGcs:
        def __call__(self, args):
            command = tuple(args)
            if command[:2] == ("storage", "cp"):
                raise subprocess.TimeoutExpired(command, 30.0)
            raise subprocess.TimeoutExpired(command, 30.0)

    store = FleetStateStore(GcsBackend("gs://fleet/eyvan/state.json", runner=BlindGcs()))

    with pytest.raises((FleetStateUnavailableError, subprocess.TimeoutExpired)):
        store.claim_workload_fence("workload", 3, "token-3", remote_id="job-3", cell_id="cell-a")


def test_an_unreadable_body_is_not_an_empty_fleet() -> None:
    """Treat an unreadable existing object as unavailable, not an empty fleet that would invalidate ownership."""

    from eyvan.capacity.fleet_state import FleetStateUnavailableError

    class ExistsButUnreadableGcs:
        def __call__(self, args):
            command = tuple(args)
            if command[:3] == ("storage", "objects", "describe"):
                return subprocess.CompletedProcess(command, 0, json.dumps({"generation": 42}), "")
            if command[:2] == ("storage", "cat"):
                return subprocess.CompletedProcess(command, 1, "", "500 backend error")
            raise AssertionError(command)

    with pytest.raises(FleetStateUnavailableError):
        GcsBackend("gs://fleet/eyvan/state.json", runner=ExistsButUnreadableGcs()).read()


def test_a_body_that_reads_back_blank_is_refused() -> None:
    """A live object is never legitimately zero bytes; that is a truncated read."""

    from eyvan.capacity.fleet_state import FleetStateUnavailableError

    class BlankBodyGcs:
        def __call__(self, args):
            command = tuple(args)
            if command[:3] == ("storage", "objects", "describe"):
                return subprocess.CompletedProcess(command, 0, json.dumps({"generation": 42}), "")
            if command[:2] == ("storage", "cat"):
                return subprocess.CompletedProcess(command, 0, "", "")
            raise AssertionError(command)

    with pytest.raises(FleetStateUnavailableError, match="empty fleet"):
        GcsBackend("gs://fleet/eyvan/state.json", runner=BlankBodyGcs()).read()


def test_a_genuinely_absent_object_is_still_an_empty_fleet() -> None:
    """The legitimate bootstrap case must keep working: nothing has been written."""

    class AbsentGcs:
        def __call__(self, args):
            command = tuple(args)
            return subprocess.CompletedProcess(command, 1, "", "404 not found")

    document, generation = GcsBackend("gs://fleet/eyvan/state.json", runner=AbsentGcs()).read()

    assert generation == 0
    assert document.get("cells") == {}


def test_unrelated_fleet_scopes_do_not_wait_behind_one_process_lock(tmp_path) -> None:
    """Long provider I/O for one fleet must not stop an unrelated fleet."""

    first = FleetStateStore(LocalBackend(tmp_path / "fleet-a.json"))
    second = FleetStateStore(LocalBackend(tmp_path / "fleet-b.json"))
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def hold_first() -> None:
        with first.exclusive():
            first_entered.set()
            assert release_first.wait(timeout=2)

    def enter_second() -> None:
        with second.exclusive():
            second_entered.set()

    first_thread = threading.Thread(target=hold_first)
    second_thread = threading.Thread(target=enter_second)
    first_thread.start()
    assert first_entered.wait(timeout=2)
    second_thread.start()

    assert second_entered.wait(timeout=2), "an unrelated durable document shared the first fleet's process lock"
    release_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    assert not first_thread.is_alive() and not second_thread.is_alive()


def test_same_fleet_scope_still_serializes_default_owned_stores(tmp_path) -> None:
    """Same-process stores share a lease identity, so the scoped lock excludes them."""

    path = tmp_path / "fleet.json"
    first = FleetStateStore(LocalBackend(path))
    second = FleetStateStore(LocalBackend(path))
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def hold_first() -> None:
        with first.exclusive():
            first_entered.set()
            assert release_first.wait(timeout=2)

    def enter_second() -> None:
        with second.exclusive():
            second_entered.set()

    first_thread = threading.Thread(target=hold_first)
    second_thread = threading.Thread(target=enter_second)
    first_thread.start()
    assert first_entered.wait(timeout=2)
    second_thread.start()
    assert not second_entered.wait(timeout=0.1)

    release_first.set()
    assert second_entered.wait(timeout=2)
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    assert not first_thread.is_alive() and not second_thread.is_alive()


def test_two_stores_in_one_agent_are_one_writer(tmp_path) -> None:
    """Share lease identity across stores in one agent process while still excluding other agents."""

    path = tmp_path / "fleet.json"
    driver_store = FleetStateStore(LocalBackend(path))
    agent_store = FleetStateStore(LocalBackend(path))

    with driver_store.exclusive():
        assert agent_store.acquire_lease(), "one agent's own stores must not fence each other out"


def test_a_genuinely_separate_agent_is_still_excluded(tmp_path) -> None:
    """The fix must not weaken what the lease is for."""

    path = tmp_path / "fleet.json"
    mine = FleetStateStore(LocalBackend(path))
    other_agent = FleetStateStore(LocalBackend(path), owner="another-host:1234:abc")

    with mine.exclusive():
        assert not other_agent.acquire_lease()


def test_a_lease_held_by_a_dead_process_on_this_host_is_reclaimed(tmp_path) -> None:
    """Otherwise an agent restart freezes every capacity mutation for an hour.

    A dead process cannot be mid-mutation, so its lease is safe to break. PID 1
    stands in for a live holder; an unused high PID stands in for a dead one.
    """

    import os
    import socket

    path = tmp_path / "fleet.json"
    dead_holder = f"{socket.gethostname()}:{2**22}:deadbeef"
    stale = FleetStateStore(LocalBackend(path), owner=dead_holder)
    assert stale.acquire_lease()

    successor = FleetStateStore(LocalBackend(path), owner=f"{socket.gethostname()}:{os.getpid()}:fresh")

    assert successor.acquire_lease()


def test_a_lease_on_another_host_is_never_assumed_dead(tmp_path) -> None:
    """Nothing here can observe a process on a different machine."""

    path = tmp_path / "fleet.json"
    remote = FleetStateStore(LocalBackend(path), owner="some-other-host:999999:abc")
    assert remote.acquire_lease()

    local = FleetStateStore(LocalBackend(path))

    assert not local.acquire_lease()


def test_a_transient_read_failure_is_retried_before_it_is_believed() -> None:
    """Retry transient fleet-state reads so a single storage failure does not break every dependent route."""

    class FlakyThenGoodGcs:
        def __init__(self):
            self.cat_calls = 0

        def __call__(self, args):
            command = tuple(args)
            if command[:3] == ("storage", "objects", "describe"):
                return subprocess.CompletedProcess(command, 0, json.dumps({"generation": 7}), "")
            if command[:2] == ("storage", "cat"):
                self.cat_calls += 1
                if self.cat_calls < 3:
                    return subprocess.CompletedProcess(command, 1, "", "not found: 404")
                return subprocess.CompletedProcess(command, 0, json.dumps(_empty_for_test()), "")
            raise AssertionError(command)

    gcs = FlakyThenGoodGcs()

    document, generation = GcsBackend("gs://fleet/eyvan/state.json", runner=gcs).read()

    assert generation == 7
    assert document.get("cells") == {}
    assert gcs.cat_calls == 3, "the read must actually have been retried"


def test_a_read_that_never_succeeds_still_refuses_to_invent_an_empty_fleet() -> None:
    """Retrying must not become guessing: exhausted attempts are still unknown."""

    from eyvan.capacity.fleet_state import _READ_ATTEMPTS, FleetStateUnavailableError

    class AlwaysFailingGcs:
        def __init__(self):
            self.cat_calls = 0

        def __call__(self, args):
            command = tuple(args)
            if command[:3] == ("storage", "objects", "describe"):
                return subprocess.CompletedProcess(command, 0, json.dumps({"generation": 7}), "")
            self.cat_calls += 1
            return subprocess.CompletedProcess(command, 1, "", "not found: 404")

    gcs = AlwaysFailingGcs()
    with pytest.raises(FleetStateUnavailableError):
        GcsBackend("gs://fleet/eyvan/state.json", runner=gcs).read()

    # Bounded in both directions. Every route that reads fleet state waits on
    # this, so an off-by-one that retried forever would hang them all, and one
    # that retried less would reintroduce the blip this retry exists to absorb.
    assert gcs.cat_calls == _READ_ATTEMPTS


def _empty_for_test() -> dict:
    return {
        "schema": "eyvan.fleet-state",
        "version": 1,
        "cells": {},
        "cell_tombstones": {},
        "workload_fences": {},
        "lease": None,
    }


def test_a_superseded_controller_cannot_drain_a_cell_it_no_longer_owns(tmp_path) -> None:
    """Draining is a mutation, and a stale one must be refused like any other.

    The tombstone check alone was not enough: a cell that has never been drained
    or released has no tombstone, and the fence validator returns immediately
    when there is none. A superseded controller could therefore take healthy
    capacity out of service while every other mutation on that cell was
    correctly refused.
    """

    owner = store(tmp_path / "fleet.json", "agent-a")

    def claim(record: FleetCellRecord) -> None:
        record.generation = 7
        record.fencing_token = "token-7"

    owner.mutate_or_create_record("cell-a", claim)

    with pytest.raises(StaleFenceError, match="older than drained cell"):
        owner.drain_record("cell-a", 6, "token-6", lambda record: None)

    with pytest.raises(StaleFenceError, match="does not own drained cell"):
        owner.drain_record("cell-a", 7, "token-other", lambda record: None)


def test_the_owning_controller_still_drains(tmp_path) -> None:
    """The fence must refuse the stale writer without refusing the real one."""

    owner = store(tmp_path / "fleet.json", "agent-a")

    def claim(record: FleetCellRecord) -> None:
        record.generation = 7
        record.fencing_token = "token-7"

    owner.mutate_or_create_record("cell-a", claim)

    def mutate(record: FleetCellRecord) -> None:
        record.desired_state = "down"
        record.state = CapacityStatus.DRAINING

    drained = owner.drain_record("cell-a", 7, "token-7", mutate)

    assert drained is not None
    assert drained.desired_state == "down"
    assert drained.state is CapacityStatus.DRAINING


def test_a_newer_incarnation_may_drain_an_older_one(tmp_path) -> None:
    """A recreate legitimately supersedes; only a *stale* writer is refused."""

    owner = store(tmp_path / "fleet.json", "agent-a")

    def claim(record: FleetCellRecord) -> None:
        record.generation = 7
        record.fencing_token = "token-7"

    owner.mutate_or_create_record("cell-a", claim)

    assert owner.drain_record("cell-a", 8, "token-8", lambda record: None) is not None


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow
