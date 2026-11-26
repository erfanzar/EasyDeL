# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Retirement requires every rank to be safe, not just an unviable slice."""

from __future__ import annotations

import typing as tp
from contextlib import ExitStack
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_systemd_transient import FakeSystemd

from eyvan._internal.json_types import JsonValue
from eyvan.capacity.drivers.base import (
    CapacityStatus,
    CellCapabilities,
    CellObservation,
    ExecutionWorkloadObservation,
    ExecutionWorkloadStatus,
    MutationContext,
)
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.config import EyvanSettings
from eyvan.worker.agent.app import create_agent_app
from eyvan.worker.agent.client import CellAgentClient
from eyvan.worker.agent.models import MutationModel, WorkloadMutationRequest, WorkloadRequest
from eyvan.worker.agent.service import CellAgentService
from eyvan.worker.runtime.native_slice import NativeSliceExecutor
from eyvan.worker.runtime.native_worker import create_native_worker_app
from eyvan.worker.runtime.systemd_transient import SystemdTransientServiceExecutor

Status = ExecutionWorkloadStatus
_SAFE = {Status.SUCCEEDED, Status.FAILED, Status.STOPPED, Status.LOST, Status.NOT_FOUND}


class RankDriver:
    """Stateful worker-side seam; the slice and both HTTP APIs remain real."""

    def __init__(self, retirement_status: Status) -> None:
        self.retirement_status = retirement_status
        self.statuses: dict[str, Status] = {}
        self.live: set[str] = set()
        self.launches: list[str] = []

    def submit(self, spec: tp.Mapping[str, JsonValue], context: MutationContext) -> str:
        remote_id = spec["submission_id"]
        assert isinstance(remote_id, str)
        self.statuses[remote_id] = Status.RUNNING
        self.live.add(remote_id)
        self.launches.append(remote_id)
        return remote_id

    def retire(self, remote_id: str, context: MutationContext) -> None:
        self.statuses[remote_id] = self.retirement_status
        if self.retirement_status in _SAFE:
            self.live.discard(remote_id)

    def status(self, remote_id: str) -> ExecutionWorkloadObservation:
        return ExecutionWorkloadObservation(
            remote_id=remote_id,
            status=self.statuses.get(remote_id, Status.NOT_FOUND),
            message="worker retirement evidence",
        )


class SliceCapacity:
    def capabilities(self) -> CellCapabilities:
        return CellCapabilities(
            provider="fake", executor="eyvan-native", accelerator_types=("tpu",), capacity_classes=("spot",)
        )

    def observe(self, cell_id: str) -> CellObservation:
        return CellObservation(
            cell_id=cell_id,
            state=CapacityStatus.HEALTHY,
            generation=3,
            execution_endpoint="http://worker-0:8100",
            metadata={"agent_endpoint": "http://source-agent"},
        )


class SliceResolver:
    def __init__(self, executor: NativeSliceExecutor) -> None:
        self.executor = executor

    def resolve(self, cell_id: str, observation: CellObservation) -> NativeSliceExecutor:
        return self.executor


def _request(generation: int) -> WorkloadRequest:
    return WorkloadRequest(
        mutation=MutationModel(
            idempotency_key=f"submit-{generation}", generation=generation, fencing_token=f"fence-{generation}"
        ),
        spec={
            "entrypoint": "python train.py",
            "submission_id": f"workload-1-a{generation}",
            "gang": {"replicas": 1, "per_replica": {"cpu": 1}, "required_topology": {}},
            "metadata": {"eyvan_workload_id": "workload-1"},
            "environment": {},
        },
    )


def _slice(stack: ExitStack, drivers: tp.Sequence[RankDriver | SystemdTransientServiceExecutor]) -> NativeSliceExecutor:
    endpoints = tuple(f"http://worker-{rank}:8100" for rank in range(len(drivers)))
    clients = [
        stack.enter_context(TestClient(create_native_worker_app(driver, secret="worker-secret"), base_url=endpoint))
        for driver, endpoint in zip(drivers, endpoints, strict=True)
    ]
    return NativeSliceExecutor(endpoints, secret="worker-secret", clients=clients)


def _services(tmp_path, monkeypatch, stack: ExitStack, source_slice: NativeSliceExecutor, *, remote: bool):
    state_path = tmp_path / "fleet.json"

    def service(executor: NativeSliceExecutor) -> CellAgentService:
        return CellAgentService(
            capacity=SliceCapacity(),  # type: ignore[arg-type]
            state=FleetStateStore(LocalBackend(state_path)),
            executors=SliceResolver(executor),  # type: ignore[arg-type]
            settings=EyvanSettings(agent_shared_secret="agent-secret"),
        )

    source = service(source_slice)
    source.submit("cell-a", _request(1))
    if not remote:
        return source, source_slice, "cell-a"

    from eyvan.worker.agent import client as agent_client_module

    source_app = create_agent_app(source, source.settings)

    def source_client(endpoint: str, *, token: str | None = None) -> CellAgentClient:
        client = TestClient(source_app, headers={"Authorization": "Bearer agent-secret"})
        stack.callback(client.close)
        return CellAgentClient(endpoint, client=client)

    monkeypatch.setattr(agent_client_module, "CellAgentClient", source_client)
    target_slice = _slice(stack, [RankDriver(Status.STOPPED), RankDriver(Status.STOPPED)])
    return service(target_slice), target_slice, "cell-b"


def _assert_predecessor_retained(tmp_path: Path) -> None:
    persisted = FleetStateStore(LocalBackend(tmp_path / "fleet.json")).workload_fence("workload-1")
    assert persisted is not None
    assert persisted.get("predecessors") == [
        {"cell_id": "cell-a", "remote_id": "workload-1-a1", "generation": 1, "fencing_token": "fence-1"}
    ]


def _assert_successor_absent(executor: NativeSliceExecutor) -> None:
    report = executor.status("workload-1-a2")
    assert [worker["status"] for worker in report["workers"]] == [Status.NOT_FOUND, Status.NOT_FOUND]


def _assert_admitted(service: CellAgentService, cell_id: str, tmp_path: Path) -> None:
    result = service.submit(cell_id, _request(2))
    assert result.remote_id == "workload-1-a2" and result.status is Status.SUBMITTED
    persisted = FleetStateStore(LocalBackend(tmp_path / "fleet.json")).workload_fence("workload-1")
    assert persisted is not None and "predecessors" not in persisted
    assert service.workload(cell_id, "workload-1-a2").status is Status.RUNNING


@pytest.mark.parametrize("remote", [False, True], ids=["local", "source-agent"])
@pytest.mark.parametrize(
    ("statuses", "aggregate"),
    [
        ((Status.STOPPED, Status.RUNNING), Status.LOST),
        ((Status.FAILED, Status.RUNNING), Status.FAILED),
        ((Status.FAILED, Status.INDETERMINATE), Status.FAILED),
        ((Status.STOPPED, Status.PENDING), Status.LOST),
        ((Status.STOPPED, Status.SUBMITTED), Status.LOST),
        ((Status.STOPPED, Status.CHECKPOINTING), Status.LOST),
        ((Status.STOPPED, Status.STOPPING), Status.LOST),
    ],
)
def test_ordinary_successor_requires_every_native_slice_rank_quiescent(
    tmp_path, monkeypatch, remote, statuses, aggregate
) -> None:
    ranks = [RankDriver(status) for status in statuses]
    with ExitStack() as stack:
        source_slice = _slice(stack, ranks)
        service, target_slice, cell_id = _services(tmp_path, monkeypatch, stack, source_slice, remote=remote)
        errors = []
        for _ in range(2):
            try:
                service.submit(cell_id, _request(2))
            except RuntimeError as exc:
                errors.append(exc)

        # Assert observable outcomes before errors so RED demonstrates the unsafe launch.
        _assert_successor_absent(target_slice)
        _assert_predecessor_retained(tmp_path)
        assert ranks[1].live == {"workload-1-a1"}
        assert all(rank.launches == ["workload-1-a1"] for rank in ranks)
        report = source_slice.status("workload-1-a1")
        assert report["status"] is aggregate, "ordinary workload viability aggregation must remain unchanged"
        assert [worker["status"] for worker in report["workers"]] == list(statuses)
        assert len(errors) == 2 and all("still reports" in str(error) for error in errors)

        # The source API must retain rank diagnostics despite its scalar wire schema.
        source = CellAgentService(
            capacity=SliceCapacity(),  # type: ignore[arg-type]
            state=service.state,
            executors=SliceResolver(source_slice),  # type: ignore[arg-type]
            settings=EyvanSettings(agent_shared_secret="agent-secret"),
        )
        with TestClient(create_agent_app(source, source.settings)) as client:
            response = client.post(
                "/api/v1/cells/cell-a/workloads/workload-1-a1/retire",
                headers={"Authorization": "Bearer agent-secret"},
                json=WorkloadMutationRequest(mutation=_request(1).mutation).model_dump(mode="json"),
            )
        assert response.status_code == 200
        evidence = response.json()
        assert set(evidence) == {"remote_id", "status", "message", "checkpoint_uri"}
        assert evidence["status"] == Status.INDETERMINATE.value
        assert f"rank 1={statuses[1].value}" in evidence["message"]
        assert "worker retirement evidence" in evidence["message"]

        ranks[1].retirement_status = Status.NOT_FOUND
        _assert_admitted(service, cell_id, tmp_path)
        assert all("workload-1-a1" not in rank.live for rank in ranks)


@pytest.mark.parametrize("remote", [False, True], ids=["local", "source-agent"])
@pytest.mark.parametrize(
    "statuses",
    [
        (Status.STOPPED, Status.NOT_FOUND),
        (Status.FAILED, Status.SUCCEEDED),
        (Status.LOST, Status.NOT_FOUND),
        (Status.SUCCEEDED, Status.STOPPED),
        (Status.NOT_FOUND, Status.NOT_FOUND),
    ],
)
def test_ordinary_successor_accepts_mixed_safe_native_slice_ranks(tmp_path, monkeypatch, remote, statuses) -> None:
    ranks = [RankDriver(status) for status in statuses]
    with ExitStack() as stack:
        source_slice = _slice(stack, ranks)
        service, _target_slice, cell_id = _services(tmp_path, monkeypatch, stack, source_slice, remote=remote)
        _assert_admitted(service, cell_id, tmp_path)
        assert all("workload-1-a1" not in rank.live for rank in ranks)


@pytest.mark.parametrize("remote", [False, True], ids=["local", "source-agent"])
def test_replayed_supersede_cannot_admit_successor_after_same_id_systemd_rank_relaunch(
    tmp_path, monkeypatch, remote
) -> None:
    managers = [FakeSystemd(), FakeSystemd()]
    ranks = [
        SystemdTransientServiceExecutor(
            tmp_path / f"rank-{rank}", command_runner=manager, boot_id_reader=lambda: "boot-a"
        )
        for rank, manager in enumerate(managers)
    ]
    with ExitStack() as stack:
        source_slice = _slice(stack, ranks)
        service, target_slice, cell_id = _services(tmp_path, monkeypatch, stack, source_slice, remote=remote)
        old = _request(1)
        retire_context = MutationContext(idempotency_key="supersede:workload-1:2", generation=1, fencing_token="fence-1")
        source_slice.retire("workload-1-a1", retire_context)
        # A delayed original submission reaches just rank 1 after the retirement.
        worker = source_slice._workers[1]
        assert (
            worker.submit(old.spec, MutationContext(idempotency_key="submit-1", generation=1, fencing_token="fence-1"))
            == "workload-1-a1"
        )
        before = sum(command[:2] == ("systemctl", "stop") for command in managers[1].commands)
        assert [entry["status"] for entry in source_slice.status("workload-1-a1")["workers"]] == [
            Status.STOPPED,
            Status.RUNNING,
        ]

        errors = []
        for _ in range(2):
            try:
                service.submit(cell_id, _request(2))
            except RuntimeError as exc:
                errors.append(exc)
        _assert_successor_absent(target_slice)
        _assert_predecessor_retained(tmp_path)
        assert ranks[1].status("workload-1-a1").status is Status.RUNNING
        assert sum(command[:2] == ("systemctl", "stop") for command in managers[1].commands) == before
        assert len(errors) == 2

        # A fresh retirement can quiesce the replacement; the successor's replay now succeeds.
        worker.retire(
            "workload-1-a1", MutationContext(idempotency_key="fresh-retire", generation=1, fencing_token="fence-1")
        )
        _assert_admitted(service, cell_id, tmp_path)
