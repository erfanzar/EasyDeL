# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fail closed before losing gang ownership or interpreting corrupt state as absent."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
from test_native_gang_agent import GangCapacity

from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import ExecutionWorkloadStatus, StaleMutationError
from eyvan.capacity.drivers.local import LocalCapacityDriver
from eyvan.capacity.fleet_state import FleetStateSchemaError, FleetStateStore, GcsBackend, LocalBackend
from eyvan.worker.agent.models import GangWorkloadMember, GangWorkloadRequest, MutationModel, WorkloadRequest
from eyvan.worker.agent.service import CellAgentService


class RecordingExecutors:
    """Keep observable local execution identities without starting processes."""

    def __init__(self) -> None:
        self.executions: dict[str, ExecutionWorkloadStatus] = {}

    def resolve(self, _cell_id, _observation):
        return self

    def resolve_gang(self, _cells):
        return self

    def submit(self, spec, _context=None):
        remote_id = spec["submission_id"]
        self.executions[remote_id] = ExecutionWorkloadStatus.RUNNING
        return remote_id

    def retire(self, remote_id, _context=None) -> None:
        if remote_id in self.executions:
            self.executions[remote_id] = ExecutionWorkloadStatus.STOPPED

    def status(self, remote_id):
        return {
            "remote_id": remote_id,
            "status": self.executions.get(remote_id, ExecutionWorkloadStatus.NOT_FOUND),
            "message": "local execution seam",
        }

    def close(self) -> None:
        pass


def _gang_request() -> GangWorkloadRequest:
    return GangWorkloadRequest(
        members=tuple(
            GangWorkloadMember(
                cell_id=cell_id,
                agent_endpoint="https://agent.example.com",
                mutation=MutationModel(
                    idempotency_key=f"submit-{cell_id}", generation=1, fencing_token=f"old-{cell_id}"
                ),
            )
            for cell_id in ("cell-a", "cell-b")
        ),
        spec={"submission_id": "old-gang", "metadata": {"eyvan_workload_id": "workload"}},
    )


@pytest.mark.parametrize("inherited", [False, True], ids=["current-gang", "inherited-gang"])
@pytest.mark.parametrize("replay", [False, True], ids=["new-generation", "same-generation-replay"])
@pytest.mark.parametrize("old_status", [ExecutionWorkloadStatus.RUNNING, ExecutionWorkloadStatus.INDETERMINATE])
def test_single_submission_refuses_unretired_gang_without_overwriting_ownership(
    tmp_path: Path, inherited: bool, replay: bool, old_status: ExecutionWorkloadStatus
) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    executors = RecordingExecutors()
    service = CellAgentService(
        capacity=GangCapacity(),  # type: ignore[arg-type]
        state=state,
        executors=executors,  # type: ignore[arg-type]
    )
    service.submit_gang("cell-a", _gang_request())
    executors.executions["old-gang"] = old_status
    original = state.workload_fence("workload")
    assert original is not None and original["kind"] == "gang"
    generation = 2
    if inherited:
        predecessor = {
            key: copy.deepcopy(original[key]) for key in ("kind", "remote_id", "coordinator_cell_id", "members")
        }
        intermediary = {
            "generation": 2,
            "fencing_token": "new-token",
            "cell_id": "cell-c",
            "remote_id": "new-single",
            "predecessors": [predecessor],
        }
        state.backend.update(lambda document: document["workload_fences"].update({"workload": intermediary}))
        generation = 2 if replay else 3
    elif replay:
        generation = 1
    expected = copy.deepcopy(state.workload_fence("workload"))
    request = WorkloadRequest(
        mutation=MutationModel(idempotency_key="new-key", generation=generation, fencing_token="new-token"),
        spec={"submission_id": "new-single", "metadata": {"eyvan_workload_id": "workload"}},
    )
    for _ in range(2):
        refusal = None
        try:
            service.submit("cell-c", request)
        except StaleMutationError as exc:
            refusal = exc
        assert "new-single" not in executors.executions, (
            "single successor launched while old gang ownership was unresolved"
        )
        assert state.workload_fence("workload") == expected, "gang identities must remain durable on refusal"
        assert refusal is not None and "gang" in str(refusal)
        assert executors.executions == {"old-gang": old_status}


def test_gang_to_gang_and_single_to_single_remain_supported(tmp_path: Path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    request = _gang_request()
    members = [member.model_dump(mode="json") for member in request.members]
    original = state.claim_gang_workload_fence("gang", remote_id="old", coordinator_cell_id="cell-a", members=members)
    newer = copy.deepcopy(members)
    for member in newer:
        member["mutation"]["generation"] = 2
    successor = state.claim_gang_workload_fence("gang", remote_id="new", coordinator_cell_id="cell-a", members=newer)
    assert successor["predecessors"][0]["members"] == original["members"]
    state.claim_workload_fence("single", 1, "old-token", remote_id="single-old", cell_id="cell-a")
    single = state.claim_workload_fence("single", 2, "new-token", remote_id="single-new", cell_id="cell-c")
    assert single["predecessors"][0]["remote_id"] == "single-old"


class InMemoryGcs:
    """Injected GCS CLI/REST transport; never consult credentials or the network."""

    def __init__(self, document: JsonObject) -> None:
        self.document = copy.deepcopy(document)
        self.generation = 1
        self.writes = 0

    def __call__(self, args):
        command = tuple(args)
        if command[:3] == ("storage", "objects", "describe"):
            body = json.dumps({"generation": self.generation})
        elif command[:2] == ("storage", "cat"):
            body = json.dumps(self.document)
        elif command[:2] == ("storage", "cp"):
            assert command[-1] == f"--if-generation-match={self.generation}"
            self.write(json.loads(Path(command[2]).read_text()), self.generation)
            body = ""
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, body, "")

    def read(self):
        return copy.deepcopy(self.document), self.generation

    def write(self, document, generation) -> None:
        assert generation == self.generation
        self.document = copy.deepcopy(document)
        self.generation += 1
        self.writes += 1


def _document() -> JsonObject:
    return {
        "schema": "eyvan.fleet-state",
        "version": 1,
        "cells": {},
        "cell_tombstones": {},
        "workload_fences": {},
        "lease": None,
    }


def _state(tmp_path: Path, backend_kind: str, document: JsonObject):
    if backend_kind == "local":
        path = tmp_path / "fleet.json"
        path.write_text(json.dumps(document))
        backend = LocalBackend(path)
        return FleetStateStore(backend), lambda: json.loads(path.read_text())
    transport = InMemoryGcs(document)
    backend = GcsBackend(
        "gs://owned-test/fleet.json",
        runner=transport,
        rest=transport if backend_kind == "gcs-rest" else None,  # type: ignore[arg-type]
    )
    return FleetStateStore(backend), lambda: copy.deepcopy(transport.document)


@pytest.mark.parametrize("backend_kind", ["local", "gcs-cli", "gcs-rest"])
@pytest.mark.parametrize("field", ["cells", "cell_tombstones", "workload_fences"])
@pytest.mark.parametrize("invalid", [None, "not-a-record", 1, []], ids=["null", "string", "integer", "list"])
def test_corrupt_registered_entries_never_observe_as_absent(
    tmp_path: Path, backend_kind: str, field: str, invalid: JsonValue
) -> None:
    document = _document()
    document[field] = {"registered": invalid}
    state, persisted = _state(tmp_path, backend_kind, document)
    # Even an unrelated missing-cell observation must not claim conclusive
    # absence from a document with unreadable capacity or ownership state.
    with pytest.raises(FleetStateSchemaError):
        LocalCapacityDriver(state).observe("registered")
    reader = {
        "cells": state.get,
        "cell_tombstones": state.cell_tombstone,
        "workload_fences": state.workload_fence,
    }[field]
    with pytest.raises(FleetStateSchemaError):
        reader("registered")
    assert persisted() == document


@pytest.mark.parametrize("backend_kind", ["local", "gcs-cli", "gcs-rest"])
@pytest.mark.parametrize("invalid", [None, "not-a-record", 1, []], ids=["null", "string", "integer", "list"])
@pytest.mark.parametrize(
    ("field", "operation"),
    [
        ("cells", "create"),
        ("cells", "drain"),
        ("cells", "tombstone"),
        ("cells", "fence-missing"),
        ("cell_tombstones", "create"),
        ("cell_tombstones", "drain"),
        ("cell_tombstones", "tombstone"),
        ("cell_tombstones", "fence-missing"),
        ("workload_fences", "single"),
        ("workload_fences", "gang"),
        ("workload_fences", "lease"),
    ],
)
def test_corrupt_entries_cannot_be_overwritten_as_missing(
    tmp_path: Path, backend_kind: str, invalid: JsonValue, field: str, operation: str
) -> None:
    document = _document()
    document[field] = {"registered": invalid}
    state, persisted = _state(tmp_path, backend_kind, document)
    members = [member.model_dump(mode="json") for member in _gang_request().members]
    actions = {
        "create": lambda: state.mutate_or_create_record("registered", lambda _: None, generation=0),
        "drain": lambda: state.drain_record("registered", 0, "token", lambda _: None),
        "tombstone": lambda: state.tombstone_record("registered", 0, "token", lambda _: True),
        "fence-missing": lambda: state.fence_missing_record("registered", 0, "token"),
        "single": lambda: state.claim_workload_fence("registered", 0, "token", remote_id="new", cell_id="cell-c"),
        "gang": lambda: state.claim_gang_workload_fence(
            "registered", remote_id="new", coordinator_cell_id="cell-a", members=members
        ),
        "lease": state.acquire_lease,
    }
    with pytest.raises(FleetStateSchemaError):
        actions[operation]()
    assert persisted() == document, "corrupt persisted ownership must not be replaced or deleted"


@pytest.mark.parametrize("backend_kind", ["local", "gcs-cli", "gcs-rest"])
def test_missing_keys_bootstrap_and_extra_object_fields_remain_compatible(tmp_path: Path, backend_kind: str) -> None:
    document = _document()
    document["cells"] = {"future-cell": {"name": "future-cell", "future_provider_field": {"enabled": True}}}
    document["cell_tombstones"] = {"retired": {"generation": 1, "fencing_token": "old", "future": [1, 2]}}
    document["workload_fences"] = {
        "old-workload": {
            "generation": 1,
            "fencing_token": "old",
            "cell_id": "cell-a",
            "remote_id": "old",
            "future": True,
        }
    }
    state, _ = _state(tmp_path, backend_kind, document)
    assert LocalCapacityDriver(state).observe("missing").state is CapacityStatus.ABSENT
    assert state.cell_tombstone("missing") is None
    assert state.workload_fence("missing") is None
    assert state.get("future-cell").metadata["future_provider_field"] == {"enabled": True}
    assert state.cell_tombstone("retired")["future"] == [1, 2]
    assert state.workload_fence("old-workload")["future"] is True
    created = state.mutate_or_create_record("missing", lambda _: None, generation=0)
    assert state.get("missing") == created
    state.claim_workload_fence("missing", 0, "token", remote_id="new", cell_id="missing")
    assert state.workload_fence("missing")["remote_id"] == "new"


def test_missing_local_document_bootstraps(tmp_path: Path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "missing.json"))
    assert state.load() == {}
    assert state.workload_fence("missing") is None
    state.claim_workload_fence("missing", 0, "token", remote_id="new", cell_id="cell-a")
    assert state.workload_fence("missing")["remote_id"] == "new"
