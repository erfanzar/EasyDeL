# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cell-agent contracts for native multi-cell workload gangs."""

from __future__ import annotations

import json
import typing as tp

import httpx
import pytest
from fastapi.testclient import TestClient

from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan.capacity.drivers.base import (
    CapacityStatus,
    CellCapabilities,
    CellObservation,
    ExecutionWorkloadStatus,
    MutationContext,
)
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.config import EyvanSettings
from eyvan.worker.agent.app import create_agent_app
from eyvan.worker.agent.client import CellAgentClient
from eyvan.worker.agent.models import (
    GangWorkloadMember,
    GangWorkloadMutationRequest,
    GangWorkloadRequest,
    MutationModel,
    WorkloadMutationRequest,
    WorkloadObservation,
    WorkloadRequest,
)
from eyvan.worker.agent.service import CellAgentService, InvalidWorkloadRequestError
from eyvan.worker.runtime.native_gang import NativeGangWorkloadResult
from eyvan.worker.runtime.registry import NativeGangCell

GangCallValue: tp.TypeAlias = JsonObject | str | None


class GangCapacity:
    def capabilities(self):
        return CellCapabilities(
            provider="fake",
            executor="eyvan-native",
            accelerator_types=("tpu",),
            capacity_classes=("spot",),
        )

    def observe(self, cell_id: str):
        return CellObservation(
            cell_id=cell_id,
            state=CapacityStatus.HEALTHY,
            generation=3,
            execution_endpoint=f"http://{cell_id}:8100",
            metadata={"agent_endpoint": "https://agent.example.com"},
        )

    def reconcile(self, cell_id, _context):
        return self.observe(cell_id)

    def release(self, cell_id, _context):
        return self.observe(cell_id)


class GangExecutor:
    def __init__(self, calls: list[tuple[str, GangCallValue]]):
        self.calls = calls
        self.retired: set[str] = set()

    def submit(self, spec: tp.Mapping[str, JsonValue]) -> str:
        self.calls.append(("submit", dict(spec)))
        submission_id = spec["submission_id"]
        assert isinstance(submission_id, str)
        return submission_id

    def status(self, remote_id: str) -> NativeGangWorkloadResult:
        self.calls.append(("status", remote_id))
        status = ExecutionWorkloadStatus.STOPPED if remote_id in self.retired else ExecutionWorkloadStatus.RUNNING
        return {
            "remote_id": remote_id,
            "status": status,
            "message": "all ranks running" if status is ExecutionWorkloadStatus.RUNNING else "ranks retired",
            "workers": [],
        }

    def stop(self, remote_id: str) -> None:
        self.calls.append(("stop", remote_id))

    def checkpoint(
        self,
        remote_id: str,
        *,
        hook: str,
        output_uri: str | None,
        context: MutationContext,
    ) -> str:
        self.calls.append(("checkpoint", remote_id))
        assert hook.strip()
        return context.idempotency_key

    def retire(self, remote_id: str) -> None:
        self.calls.append(("retire", remote_id))
        self.retired.add(remote_id)

    def logs(self, remote_id: str) -> str:
        self.calls.append(("logs", remote_id))
        return f"logs for {remote_id}"

    def close(self) -> None:
        self.calls.append(("close", None))


class GangResolver:
    def __init__(self) -> None:
        self.cells: list[tuple[NativeGangCell, ...]] = []
        self.calls: list[tuple[str, GangCallValue]] = []

    def resolve_gang(self, cells: tp.Sequence[NativeGangCell]) -> GangExecutor:
        self.cells.append(tuple(cells))
        return GangExecutor(self.calls)


class MismatchedGangExecutor(GangExecutor):
    def submit(self, spec: tp.Mapping[str, JsonValue]) -> str:
        self.calls.append(("submit", dict(spec)))
        return "unexpected-remote-id"


class MismatchedGangResolver(GangResolver):
    def resolve_gang(self, cells: tp.Sequence[NativeGangCell]) -> MismatchedGangExecutor:
        self.cells.append(tuple(cells))
        return MismatchedGangExecutor(self.calls)


def _member(cell_id: str, *, key: str, token: str, endpoint: str = "https://agent.example.com") -> GangWorkloadMember:
    return GangWorkloadMember(
        cell_id=cell_id,
        agent_endpoint=endpoint,
        mutation=MutationModel(idempotency_key=key, generation=7, fencing_token=token),
    )


def _request(*, members: tuple[GangWorkloadMember, ...] | None = None) -> GangWorkloadRequest:
    return GangWorkloadRequest(
        members=members
        or (
            _member("cell-a", key="submit-a", token="fence-a"),
            _member("cell-b", key="submit-b", token="fence-b"),
        ),
        spec={
            "submission_id": "workload-1-a7",
            "metadata": {"eyvan_workload_id": "workload-1"},
            "environment": {},
        },
    )


def _service(tmp_path):
    resolver = GangResolver()
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    service = CellAgentService(
        capacity=GangCapacity(),  # type: ignore[arg-type]
        state=state,
        executors=resolver,  # type: ignore[arg-type]
    )
    return service, state, resolver


def test_native_gang_agent_api_submits_observes_stops_retires_and_logs(tmp_path) -> None:
    service, state, resolver = _service(tmp_path)
    request = _request()
    app = create_agent_app(service, EyvanSettings(agent_shared_secret="agent-secret"))

    with TestClient(app) as client:
        headers = {"Authorization": "Bearer agent-secret"}
        submitted = client.post(
            "/api/v1/cells/cell-a/workload-gangs",
            headers=headers,
            json=request.model_dump(mode="json"),
        )
        assert submitted.status_code == 200
        assert submitted.json() == {
            "remote_id": "workload-1-a7",
            "status": "SUBMITTED",
            "message": "",
            "checkpoint_uri": None,
        }

        status = client.get(
            "/api/v1/cells/cell-a/workload-gangs/workload-1/workload-1-a7",
            headers=headers,
        )
        assert status.status_code == 200
        assert status.json()["status"] == "RUNNING"

        action_members = tuple(
            member.model_copy(
                update={"mutation": member.mutation.model_copy(update={"idempotency_key": f"stop-{member.cell_id}"})}
            )
            for member in request.members
        )
        action = GangWorkloadMutationRequest(members=action_members)
        stopped = client.post(
            "/api/v1/cells/cell-a/workload-gangs/workload-1/workload-1-a7/stop",
            headers=headers,
            json=action.model_dump(mode="json"),
        )
        assert stopped.status_code == 200
        assert stopped.json()["status"] == "STOPPING"

        retired = client.post(
            "/api/v1/cells/cell-a/workload-gangs/workload-1/workload-1-a7/retire",
            headers=headers,
            json=action.model_dump(mode="json"),
        )
        assert retired.status_code == 200
        assert retired.json()["status"] == "STOPPED"

        logs = client.get(
            "/api/v1/cells/cell-a/workload-gangs/workload-1/workload-1-a7/logs",
            headers=headers,
        )
        assert logs.status_code == 200
        assert logs.text == "logs for workload-1-a7"

    fence = state.workload_fence("workload-1")
    assert fence is not None
    assert fence["kind"] == "gang"
    assert fence["coordinator_cell_id"] == "cell-a"
    members = fence["members"]
    assert isinstance(members, list)
    assert all(is_json_object(member) for member in members)
    assert [member["cell_id"] for member in members if is_json_object(member)] == ["cell-a", "cell-b"]
    assert [[cell.cell_id for cell in cells] for cells in resolver.cells] == [
        ["cell-a", "cell-b"],
        ["cell-a", "cell-b"],
        ["cell-a", "cell-b"],
        ["cell-a", "cell-b"],
        ["cell-a", "cell-b"],
    ]
    assert [name for name, _ in resolver.calls] == [
        "submit",
        "close",
        "status",
        "close",
        "stop",
        "close",
        "retire",
        "status",
        "close",
        "logs",
        "close",
    ]
    stop_cells = resolver.cells[2]
    assert [cell.context.idempotency_key for cell in stop_cells] == ["stop-cell-a", "stop-cell-b"]


def test_native_gang_rejects_cross_agent_members_before_submit(tmp_path) -> None:
    service, _state, resolver = _service(tmp_path)
    app = create_agent_app(service, EyvanSettings(), allow_unauthenticated=True)
    payload = _request().model_dump(mode="json")
    payload["members"][1]["agent_endpoint"] = "https://other-agent.example.com"

    with TestClient(app) as client:
        response = client.post("/api/v1/cells/cell-a/workload-gangs", json=payload)

    assert response.status_code == 422
    assert "same agent_endpoint" in response.text
    assert resolver.cells == []


def test_native_gang_rejects_a_stale_member_fence_without_touching_workers(tmp_path) -> None:
    service, _state, resolver = _service(tmp_path)
    request = _request()
    service.submit_gang("cell-a", request)
    stale_members = (
        request.members[0],
        request.members[1].model_copy(
            update={"mutation": request.members[1].mutation.model_copy(update={"fencing_token": "wrong"})}
        ),
    )

    with pytest.raises(Exception, match="does not own every member fence"):
        service.stop_gang(
            "cell-a",
            "workload-1",
            "workload-1-a7",
            GangWorkloadMutationRequest(members=stale_members),
        )

    assert [name for name, _ in resolver.calls] == ["submit", "close"]


def test_native_gang_checkpoint_fans_out_and_names_the_output_uri(tmp_path) -> None:
    """The checkpoint reaches the gang driver, not one cell's single worker."""

    service, _state, resolver = _service(tmp_path)
    request = _request()
    service.submit_gang("cell-a", request)
    members = tuple(
        member.model_copy(update={"mutation": member.mutation.model_copy(update={"idempotency_key": "ckpt-shared"})})
        for member in request.members
    )

    observation = service.checkpoint_gang(
        "cell-a",
        "workload-1",
        "workload-1-a7",
        members,
        hook="python checkpoint.py",
        output_uri="gs://bucket/checkpoints/workload-1",
        context=MutationContext(idempotency_key="ckpt-shared", generation=7, fencing_token="cell:1"),
    )

    assert observation.remote_id == "ckpt-shared"
    assert observation.status is ExecutionWorkloadStatus.CHECKPOINTING
    assert observation.checkpoint_uri == "gs://bucket/checkpoints/workload-1"
    assert [name for name, _ in resolver.calls] == ["submit", "close", "checkpoint", "close"]


def test_native_gang_checkpoint_refuses_a_stale_member_fence(tmp_path) -> None:
    service, _state, resolver = _service(tmp_path)
    request = _request()
    service.submit_gang("cell-a", request)
    stale_members = (
        request.members[0],
        request.members[1].model_copy(
            update={"mutation": request.members[1].mutation.model_copy(update={"fencing_token": "wrong"})}
        ),
    )

    with pytest.raises(Exception, match="does not own every member fence"):
        service.checkpoint_gang(
            "cell-a",
            "workload-1",
            "workload-1-a7",
            stale_members,
            hook="python checkpoint.py",
            output_uri="gs://bucket/checkpoints/workload-1",
            context=MutationContext(idempotency_key="ckpt-shared", generation=7, fencing_token="cell:1"),
        )

    assert [name for name, _ in resolver.calls] == ["submit", "close"]


def test_cell_agent_maps_invalid_workload_submissions_to_422(tmp_path) -> None:
    service, _state, _resolver = _service(tmp_path)
    app = create_agent_app(service, EyvanSettings(agent_shared_secret="agent-secret"))
    mutation = {"idempotency_key": "submit-1", "generation": 1, "fencing_token": "cell:1"}

    with TestClient(app) as client:
        headers = {"Authorization": "Bearer agent-secret"}
        missing_workload_id = client.post(
            "/api/v1/cells/cell-a/workloads",
            headers=headers,
            json={"mutation": mutation, "spec": {"submission_id": "workload-1", "metadata": {}}},
        )
        assert missing_workload_id.status_code == 422
        assert "eyvan_workload_id" in missing_workload_id.json()["detail"]

        bad_submission_id = client.post(
            "/api/v1/cells/cell-a/workloads",
            headers=headers,
            json={
                "mutation": mutation,
                "spec": {"submission_id": 123, "metadata": {"eyvan_workload_id": "workload-1"}},
            },
        )
        assert bad_submission_id.status_code == 422
        assert "submission_id" in bad_submission_id.json()["detail"]

    with pytest.raises(InvalidWorkloadRequestError, match="eyvan_workload_id") as excinfo:
        service.submit(
            "cell-a",
            WorkloadRequest(
                mutation=MutationModel(idempotency_key="submit-1", generation=1, fencing_token="cell:1"),
                spec={"submission_id": "workload-1", "metadata": {}},
            ),
        )
    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.status_code == 422


def test_native_gang_maps_invalid_submissions_to_422(tmp_path) -> None:
    service, _state, resolver = _service(tmp_path)
    app = create_agent_app(service, EyvanSettings(agent_shared_secret="agent-secret"))

    with TestClient(app) as client:
        headers = {"Authorization": "Bearer agent-secret"}
        missing_workload_id = _request().model_dump(mode="json")
        missing_workload_id["spec"]["metadata"] = {}
        response = client.post("/api/v1/cells/cell-a/workload-gangs", headers=headers, json=missing_workload_id)
        assert response.status_code == 422
        assert "eyvan_workload_id" in response.json()["detail"]

        blank_submission_id = _request().model_dump(mode="json")
        blank_submission_id["spec"]["submission_id"] = "  "
        response = client.post("/api/v1/cells/cell-a/workload-gangs", headers=headers, json=blank_submission_id)
        assert response.status_code == 422
        assert "submission_id" in response.json()["detail"]

        response = client.post(
            "/api/v1/cells/cell-b/workload-gangs",
            headers=headers,
            json=_request().model_dump(mode="json"),
        )
        assert response.status_code == 422
        assert "lexicographically first member cell" in response.json()["detail"]

    assert resolver.cells == []


def test_native_gang_retires_a_launched_gang_when_the_remote_id_mismatches(tmp_path) -> None:
    resolver = MismatchedGangResolver()
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    service = CellAgentService(
        capacity=GangCapacity(),  # type: ignore[arg-type]
        state=state,
        executors=resolver,  # type: ignore[arg-type]
    )

    with pytest.raises(
        RuntimeError,
        match=r"unexpected remote ID 'unexpected-remote-id'.*the launched gang was retired",
    ):
        service.submit_gang("cell-a", _request())

    assert [name for name, _ in resolver.calls] == ["submit", "retire", "close"]
    assert resolver.calls[1] == ("retire", "unexpected-remote-id")


def test_cell_agent_client_uses_group_routes() -> None:
    observed: list[tuple[str, str, JsonObject | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        decoded = json.loads(request.content) if request.content else None
        assert decoded is None or is_json_object(decoded)
        payload = decoded if is_json_object(decoded) else None
        observed.append((request.method, request.url.path, payload))
        if request.method == "POST" and request.url.path == "/api/v1/cells/cell-a/workload-gangs":
            return httpx.Response(
                200,
                json={"remote_id": "workload-1-a7", "status": ExecutionWorkloadStatus.SUBMITTED.value},
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"remote_id": "workload-1-a7", "status": ExecutionWorkloadStatus.RUNNING.value},
            )
        if request.method == "POST" and request.url.path.endswith("/stop"):
            return httpx.Response(
                200,
                json={"remote_id": "workload-1-a7", "status": ExecutionWorkloadStatus.STOPPING.value},
            )
        if request.method == "POST" and request.url.path.endswith("/retire"):
            return httpx.Response(
                200,
                json={"remote_id": "workload-1-a7", "status": ExecutionWorkloadStatus.STOPPED.value},
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    http_client = httpx.Client(base_url="http://agent", transport=httpx.MockTransport(handler))
    client = CellAgentClient("http://agent", client=http_client)
    request = _request()
    action = GangWorkloadMutationRequest(members=request.members)
    try:
        assert client.submit_gang("cell-a", request).status is ExecutionWorkloadStatus.SUBMITTED
        assert client.gang_workload("cell-a", "workload-1", "workload-1-a7").status is ExecutionWorkloadStatus.RUNNING
        assert (
            client.stop_gang("cell-a", "workload-1", "workload-1-a7", action).status is ExecutionWorkloadStatus.STOPPING
        )
        assert (
            client.retire_gang("cell-a", "workload-1", "workload-1-a7", action).status is ExecutionWorkloadStatus.STOPPED
        )
    finally:
        client.close()

    assert [path for _method, path, _payload in observed] == [
        "/api/v1/cells/cell-a/workload-gangs",
        "/api/v1/cells/cell-a/workload-gangs/workload-1/workload-1-a7",
        "/api/v1/cells/cell-a/workload-gangs/workload-1/workload-1-a7/stop",
        "/api/v1/cells/cell-a/workload-gangs/workload-1/workload-1-a7/retire",
    ]


class StuckRankGangExecutor(GangExecutor):
    """A rank that keeps reporting RUNNING even after retire was accepted."""

    def retire(self, remote_id: str) -> None:
        self.calls.append(("retire", remote_id))


class _StaticGangResolver:
    def __init__(self, executor: GangExecutor) -> None:
        self.executor = executor

    def resolve_gang(self, cells: tp.Sequence[NativeGangCell]) -> GangExecutor:
        return self.executor


class SingleCellDriver:
    """Minimal execution-driver double for single-cell service tests."""

    def __init__(self, calls: list[tuple[str, tp.Any]]) -> None:
        self.calls = calls
        self.submit_called = False

    def submit(self, spec: tp.Mapping[str, JsonValue], context: MutationContext) -> str:
        self.calls.append(("submit", context.idempotency_key))
        self.submit_called = True
        submission_id = spec.get("submission_id")
        assert isinstance(submission_id, str) and submission_id.strip()
        return submission_id.strip()

    def status(self, remote_id: str) -> dict[str, tp.Any]:
        self.calls.append(("status", remote_id))
        return {"remote_id": remote_id, "status": ExecutionWorkloadStatus.RUNNING, "message": "double"}

    def retire(self, remote_id: str, context: MutationContext) -> None:
        self.calls.append(("retire", context.idempotency_key))


class SingleCellResolver:
    def __init__(self, driver: SingleCellDriver) -> None:
        self.driver = driver

    def resolve(self, _cell_id: str, _observation: tp.Any) -> SingleCellDriver:
        return self.driver


def _single_cell_service(tmp_path, driver: SingleCellDriver) -> CellAgentService:
    return CellAgentService(
        capacity=GangCapacity(),  # type: ignore[arg-type]
        state=FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        executors=SingleCellResolver(driver),  # type: ignore[arg-type]
    )


def test_retire_reports_the_status_the_cell_actually_reaches(tmp_path) -> None:
    calls: list[tuple[str, tp.Any]] = []
    service = _single_cell_service(tmp_path, SingleCellDriver(calls))
    request = WorkloadMutationRequest(
        mutation=MutationModel(idempotency_key="retire-1", generation=3, fencing_token="fence-3")
    )

    observation = service.retire("cell-a", "pred-1", request)

    assert observation.status is ExecutionWorkloadStatus.RUNNING
    assert ("status", "pred-1") in calls


def test_retire_gang_reports_the_status_the_ranks_actually_reach(tmp_path) -> None:
    calls: list[tuple[str, GangCallValue]] = []
    service = CellAgentService(
        capacity=GangCapacity(),  # type: ignore[arg-type]
        state=FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        executors=_StaticGangResolver(StuckRankGangExecutor(calls)),  # type: ignore[arg-type]
    )
    service.submit_gang("cell-a", _request())
    action = GangWorkloadMutationRequest(
        members=(
            _member("cell-a", key="retire-a", token="fence-a"),
            _member("cell-b", key="retire-b", token="fence-b"),
        )
    )

    observation = service.retire_gang("cell-a", "workload-1", "workload-1-a7", action)

    assert observation.status is ExecutionWorkloadStatus.RUNNING


def test_remote_predecessor_retirement_refuses_a_rank_that_is_still_running(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eyvan.worker.agent import client as agent_client_module

    class _Client:
        def __init__(self, _endpoint: str, *, token: str | None = None) -> None:
            pass

        def retire(self, cell_id: str, remote_id: str, request: WorkloadMutationRequest) -> WorkloadObservation:
            return WorkloadObservation(remote_id=remote_id, status=ExecutionWorkloadStatus.RUNNING)

        def close(self) -> None:
            pass

    monkeypatch.setattr(agent_client_module, "CellAgentClient", _Client)
    service, _state, _resolver = _service(tmp_path)
    request = WorkloadMutationRequest(
        mutation=MutationModel(idempotency_key="supersede-1", generation=2, fencing_token="fence-2")
    )

    with pytest.raises(RuntimeError, match="still reports"):
        service._retire_on_source_agent("cell-a", "pred-1", request)


def test_local_predecessor_retirement_refuses_a_rank_that_is_still_running(tmp_path) -> None:
    calls: list[tuple[str, tp.Any]] = []
    service = _single_cell_service(tmp_path, SingleCellDriver(calls))
    context = MutationContext(idempotency_key="submit-new", generation=2, fencing_token="fence-2")
    predecessor = {"cell_id": "cell-a", "remote_id": "pred-1", "generation": 1, "fencing_token": "fence-1"}

    with pytest.raises(RuntimeError, match="still reports"):
        service._retire_predecessor("workload-1", "cell-a", context, predecessor)

    assert ("retire", "supersede:workload-1:2") in calls


def test_submit_retires_a_launched_execution_when_a_later_step_fails(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tp.Any]] = []
    driver = SingleCellDriver(calls)
    service = _single_cell_service(tmp_path, driver)
    state = service.state
    original_require_lease = state.require_lease

    def explode_after_launch() -> None:
        if driver.submit_called:
            raise RuntimeError("lease check exploded after launch")
        original_require_lease()

    monkeypatch.setattr(state, "require_lease", explode_after_launch)
    request = WorkloadRequest(
        mutation=MutationModel(idempotency_key="submit-new", generation=3, fencing_token="fence-3"),
        spec={
            "submission_id": "wl-1",
            "metadata": {"eyvan_workload_id": "workload-1"},
            "environment": {},
        },
    )

    with pytest.raises(RuntimeError, match="exploded"):
        service.submit("cell-a", request)

    rollback_retires = [
        key for name, key in calls if name == "retire" and isinstance(key, str) and key.startswith("rollback:")
    ]
    assert rollback_retires, "a launched execution must be retired when a later submit step fails"


class RetirementEvidenceDriver(SingleCellDriver):
    """Keep execution state separate from a retirement's potentially unknown read-back."""

    def __init__(self, retirement_status: ExecutionWorkloadStatus) -> None:
        super().__init__([])
        self.retirement_status = retirement_status
        self.launches: list[str] = []
        self.live: set[str] = set()
        self.retirement_requested: set[str] = set()

    def submit(self, spec: tp.Mapping[str, JsonValue], context: MutationContext) -> str:
        remote_id = super().submit(spec, context)
        self.launches.append(remote_id)
        self.live.add(remote_id)
        return remote_id

    def retire(self, remote_id: str, context: MutationContext) -> None:
        self.retirement_requested.add(remote_id)
        if self.retirement_status in {
            ExecutionWorkloadStatus.SUCCEEDED,
            ExecutionWorkloadStatus.FAILED,
            ExecutionWorkloadStatus.STOPPED,
            ExecutionWorkloadStatus.LOST,
            ExecutionWorkloadStatus.NOT_FOUND,
        }:
            self.live.discard(remote_id)

    def status(self, remote_id: str) -> dict[str, tp.Any]:
        status = self.retirement_status if remote_id in self.retirement_requested else ExecutionWorkloadStatus.RUNNING
        return {"remote_id": remote_id, "status": status, "message": "retirement read-back"}


def _retirement_request(generation: int) -> WorkloadRequest:
    return WorkloadRequest(
        mutation=MutationModel(
            idempotency_key=f"submit-{generation}", generation=generation, fencing_token=f"fence-{generation}"
        ),
        spec={
            "submission_id": f"workload-1-a{generation}",
            "metadata": {"eyvan_workload_id": "workload-1"},
            "environment": {},
        },
    )


def _retirement_services(tmp_path, monkeypatch, *, remote: bool, status: ExecutionWorkloadStatus):
    """Route remote retirement through the real source-agent API and typed client."""

    from eyvan.worker.agent import client as agent_client_module

    source_driver = RetirementEvidenceDriver(status)
    source = _single_cell_service(tmp_path, source_driver)
    source.submit("cell-a", _retirement_request(1))
    if not remote:
        return source, source_driver, source_driver, "cell-a"

    target_driver = RetirementEvidenceDriver(ExecutionWorkloadStatus.STOPPED)
    target = _single_cell_service(tmp_path, target_driver)
    source_app = create_agent_app(source, EyvanSettings(agent_shared_secret="agent-secret"))
    original_client = CellAgentClient

    def source_client(endpoint: str, *, token: str | None = None) -> CellAgentClient:
        transport_client = TestClient(source_app, headers={"Authorization": "Bearer agent-secret"})
        return original_client(endpoint, client=transport_client)

    monkeypatch.setattr(agent_client_module, "CellAgentClient", source_client)
    return target, source_driver, target_driver, "cell-b"


@pytest.mark.parametrize("remote", [False, True], ids=["local", "source-agent"])
@pytest.mark.parametrize(
    "status",
    [
        ExecutionWorkloadStatus.INDETERMINATE,
        ExecutionWorkloadStatus.PENDING,
        ExecutionWorkloadStatus.SUBMITTED,
        ExecutionWorkloadStatus.RUNNING,
        ExecutionWorkloadStatus.CHECKPOINTING,
        ExecutionWorkloadStatus.STOPPING,
    ],
)
def test_successor_refuses_nonterminal_retirement_and_retains_predecessor(tmp_path, monkeypatch, remote, status) -> None:
    service, source_driver, target_driver, cell_id = _retirement_services(
        tmp_path, monkeypatch, remote=remote, status=status
    )
    errors: list[RuntimeError] = []
    for _ in range(2):
        try:
            service.submit(cell_id, _retirement_request(2))
        except RuntimeError as exc:
            errors.append(exc)

    assert source_driver.live == {"workload-1-a1"}, "an unknown read-back does not prove the predecessor stopped"
    assert target_driver.launches == ([] if remote else ["workload-1-a1"]), "no successor or duplicate may launch"
    persisted = FleetStateStore(LocalBackend(tmp_path / "fleet.json")).workload_fence("workload-1")
    assert persisted is not None
    assert persisted.get("predecessors") == [
        {"cell_id": "cell-a", "remote_id": "workload-1-a1", "generation": 1, "fencing_token": "fence-1"}
    ]
    assert len(errors) == 2
    assert all("still reports" in str(error) and status.value in str(error) for error in errors)

    # Retrying the same successor may proceed only once quiescence can be observed.
    source_driver.retirement_status = ExecutionWorkloadStatus.STOPPED
    result = service.submit(cell_id, _retirement_request(2))
    assert result.remote_id == "workload-1-a2"
    assert target_driver.live == {"workload-1-a2"}
    assert target_driver.launches == (["workload-1-a2"] if remote else ["workload-1-a1", "workload-1-a2"])
    persisted = FleetStateStore(LocalBackend(tmp_path / "fleet.json")).workload_fence("workload-1")
    assert persisted is not None and "predecessors" not in persisted


@pytest.mark.parametrize("remote", [False, True], ids=["local", "source-agent"])
@pytest.mark.parametrize(
    "status",
    [
        ExecutionWorkloadStatus.SUCCEEDED,
        ExecutionWorkloadStatus.FAILED,
        ExecutionWorkloadStatus.STOPPED,
        ExecutionWorkloadStatus.LOST,
        ExecutionWorkloadStatus.NOT_FOUND,
    ],
)
def test_successor_accepts_terminal_or_absent_retirement(tmp_path, monkeypatch, remote, status) -> None:
    service, source_driver, target_driver, cell_id = _retirement_services(
        tmp_path, monkeypatch, remote=remote, status=status
    )

    result = service.submit(cell_id, _retirement_request(2))

    assert result.remote_id == "workload-1-a2"
    assert result.status is ExecutionWorkloadStatus.SUBMITTED
    assert "workload-1-a1" not in source_driver.live
    assert target_driver.live == {"workload-1-a2"}
    assert target_driver.launches == (["workload-1-a2"] if remote else ["workload-1-a1", "workload-1-a2"])
    persisted = FleetStateStore(LocalBackend(tmp_path / "fleet.json")).workload_fence("workload-1")
    assert persisted is not None and "predecessors" not in persisted


@pytest.mark.parametrize(
    "capacity_status",
    [
        CapacityStatus.ABSENT,
        CapacityStatus.RELEASED,
        CapacityStatus.NEEDS_RECREATE,
        CapacityStatus.HALTED_BUDGET,
        CapacityStatus.HALTED_QUOTA,
        CapacityStatus.UNOBSERVED,
        CapacityStatus.UNAVAILABLE,
        CapacityStatus.PROVIDER_ERROR,
        CapacityStatus.HEALTHY,
    ],
)
def test_successor_without_source_endpoint_requires_definitive_capacity_loss(tmp_path, capacity_status) -> None:
    class SourceCapacity(GangCapacity):
        def observe(self, cell_id: str):
            if cell_id == "cell-a":
                return CellObservation(cell_id=cell_id, state=capacity_status, generation=1)
            return super().observe(cell_id)

    driver = RetirementEvidenceDriver(ExecutionWorkloadStatus.STOPPED)
    service = _single_cell_service(tmp_path, driver)
    service.capacity = SourceCapacity()  # type: ignore[assignment]
    service.state.claim_workload_fence("workload-1", 1, "fence-1", remote_id="workload-1-a1", cell_id="cell-a")
    definitive_loss = capacity_status in {
        CapacityStatus.ABSENT,
        CapacityStatus.RELEASED,
        CapacityStatus.NEEDS_RECREATE,
        CapacityStatus.HALTED_BUDGET,
        CapacityStatus.HALTED_QUOTA,
    }
    if definitive_loss:
        result = service.submit("cell-b", _retirement_request(2))
        assert result.remote_id == "workload-1-a2"
        assert driver.live == {"workload-1-a2"}
        assert driver.launches == ["workload-1-a2"]
    else:
        with pytest.raises(RuntimeError, match="no agent endpoint"):
            service.submit("cell-b", _retirement_request(2))
        assert driver.live == set()
        assert driver.launches == []
    persisted = FleetStateStore(LocalBackend(tmp_path / "fleet.json")).workload_fence("workload-1")
    assert persisted is not None
    assert bool(persisted.get("predecessors")) is not definitive_loss
