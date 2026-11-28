# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A replayed gang supersession must prove every old rank is quiescent."""

from __future__ import annotations

import copy
import json
import typing as tp
from contextlib import ExitStack
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from test_native_gang_agent import GangCapacity
from test_systemd_transient import FakeSystemd

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.config import EyvanSettings
from eyvan.worker.agent.app import create_agent_app
from eyvan.worker.agent.models import GangWorkloadMember, GangWorkloadRequest, MutationModel
from eyvan.worker.agent.service import CellAgentService
from eyvan.worker.runtime.native_gang import NativeGangExecutor, NativeGangMember
from eyvan.worker.runtime.native_worker import create_native_worker_app
from eyvan.worker.runtime.registry import NativeGangCell
from eyvan.worker.runtime.systemd_transient import SystemdCommandResult, SystemdTransientServiceExecutor

Status = ExecutionWorkloadStatus
_HEADERS = {"Authorization": "Bearer worker-secret"}


class SupersessionSystemd(FakeSystemd):
    """Fail the systemd stop seam without replacing execution or HTTP logic."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_stop = False

    def __call__(self, command: tuple[str, ...]) -> SystemdCommandResult:
        if self.fail_stop and command[:2] == ("systemctl", "stop"):
            self.commands.append(command)
            return SystemdCommandResult(returncode=1, stderr="injected partial supersession stop failure")
        return super().__call__(command)


class RecordingWorkerClient(TestClient):
    """Capture the original ranked submission for its delayed worker-route replay."""

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__(*args, **kwargs)
        self.submissions: list[dict[str, tp.Any]] = []

    def request(self, method: str, url: tp.Any, **kwargs: tp.Any) -> httpx.Response:
        if method == "POST" and str(url) == "/api/v1/workloads":
            self.submissions.append(copy.deepcopy(kwargs["json"]))
        return super().request(method, url, **kwargs)


class RealGangResolver:
    def __init__(self, clients: dict[str, RecordingWorkerClient]) -> None:
        self.clients = clients

    def resolve_gang(self, cells: tp.Sequence[NativeGangCell]) -> NativeGangExecutor:
        return NativeGangExecutor(
            [
                NativeGangMember(
                    cell_id=cell.cell_id,
                    endpoints=(cell.observation.execution_endpoint,),
                    secret="worker-secret",
                    context=cell.context,
                )
                for cell in cells
            ],
            clients=[self.clients[cell.cell_id] for cell in cells],
        )


def _request(generation: int, cell_ids: tuple[str, str]) -> GangWorkloadRequest:
    return GangWorkloadRequest(
        members=tuple(
            GangWorkloadMember(
                cell_id=cell_id,
                agent_endpoint="https://agent.example.com",
                mutation=MutationModel(
                    idempotency_key=f"submit-{generation}-{cell_id}",
                    generation=generation,
                    fencing_token=f"fence-{generation}-{cell_id}",
                ),
            )
            for cell_id in cell_ids
        ),
        spec={
            "entrypoint": "python train.py",
            "submission_id": f"workload-1-a{generation}",
            "gang": {"replicas": 1, "per_replica": {"cpu": 1}, "required_topology": {}},
            "metadata": {"eyvan_workload_id": "workload-1"},
            "environment": {},
        },
    )


def _record(driver: SystemdTransientServiceExecutor, remote_id: str) -> dict[str, tp.Any]:
    return json.loads(driver._record_path(remote_id).read_text(encoding="utf-8"))


def _stop_count(manager: FakeSystemd) -> int:
    return sum(command[:2] == ("systemctl", "stop") for command in manager.commands)


@pytest.mark.parametrize("indeterminate", [False, True], ids=["running-rank", "indeterminate-rank"])
def test_replayed_gang_supersession_requires_relaunched_old_rank_quiescence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, indeterminate: bool
) -> None:
    monkeypatch.setenv("EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", "0.1")
    monkeypatch.setenv("EYVAN_SUPERSEDE_POLL_S", "0.01")
    cell_ids = ("cell-a", "cell-b", "cell-c", "cell-d")
    managers = {cell_id: SupersessionSystemd() for cell_id in cell_ids}
    drivers = {
        cell_id: SystemdTransientServiceExecutor(
            tmp_path / cell_id, command_runner=managers[cell_id], boot_id_reader=lambda: "boot-a"
        )
        for cell_id in cell_ids
    }
    with ExitStack() as stack:
        clients = {
            cell_id: stack.enter_context(
                RecordingWorkerClient(
                    create_native_worker_app(drivers[cell_id], secret="worker-secret"),
                    base_url=f"http://{cell_id}:8100",
                )
            )
            for cell_id in cell_ids
        }
        state_path = tmp_path / "fleet.json"
        service = CellAgentService(
            capacity=GangCapacity(),  # type: ignore[arg-type]
            state=FleetStateStore(LocalBackend(state_path)),
            executors=RealGangResolver(clients),  # type: ignore[arg-type]
            settings=EyvanSettings(agent_shared_secret="agent-secret"),
        )
        agent = stack.enter_context(TestClient(create_agent_app(service, service.settings)))
        original = _request(1, ("cell-a", "cell-b"))
        successor = _request(2, ("cell-c", "cell-d"))
        service.submit_gang("cell-a", original)
        original_payload = copy.deepcopy(clients["cell-a"].submissions[0])
        assert original_payload["mutation"] == original.members[0].mutation.model_dump(mode="json")
        predecessor = {
            "kind": "gang",
            "remote_id": "workload-1-a1",
            "coordinator_cell_id": "cell-a",
            "members": [member.model_dump(mode="json") for member in original.members],
        }

        def assert_predecessor_retained() -> None:
            persisted = FleetStateStore(LocalBackend(state_path)).workload_fence("workload-1")
            assert persisted is not None
            assert persisted["remote_id"] == "workload-1-a2"
            assert persisted["members"] == [member.model_dump(mode="json") for member in successor.members]
            assert persisted.get("predecessors") == [predecessor]

        # The first successor attempt retires rank 0 but fails to stop rank 1.
        managers["cell-b"].fail_stop = True
        with pytest.raises(RuntimeError, match="cannot safely retire"):
            service.submit_gang("cell-c", successor)
        assert drivers["cell-a"].status("workload-1-a1").status is Status.STOPPED
        assert_predecessor_retained()
        assert all(clients[cell_id].submissions == [] for cell_id in ("cell-c", "cell-d"))

        # Agent-level fencing rejects the old generation; the already-delayed
        # original request can still reach its old native-worker host directly.
        old_generation = agent.post(
            "/api/v1/cells/cell-a/workload-gangs",
            headers={"Authorization": "Bearer agent-secret"},
            json=original.model_dump(mode="json"),
        )
        assert old_generation.status_code == 409
        assert "generation 1 is older" in old_generation.text
        assert len(clients["cell-a"].submissions) == 1
        retired_record = _record(drivers["cell-a"], "workload-1-a1")
        relaunched = clients["cell-a"].post("/api/v1/workloads", headers=_HEADERS, json=original_payload)
        assert relaunched.status_code == 200 and relaunched.json() == {"remote_id": "workload-1-a1"}
        replacement_record = _record(drivers["cell-a"], "workload-1-a1")
        assert replacement_record["incarnation"] == retired_record["incarnation"] + 1
        retire_key = "supersede:workload-1:2:cell-a"
        assert replacement_record["retire_keys"][retire_key] == retired_record["incarnation"]
        assert drivers["cell-a"].status("workload-1-a1").status is Status.RUNNING
        before = _stop_count(managers["cell-a"])
        managers["cell-b"].fail_stop = False
        managers["cell-a"].fail_show = indeterminate
        # The 0.1 s retire window exists so the injected stop failure above
        # times out quickly. Here cell-b's stop succeeds and only has to be
        # observed; under coverage on a loaded runner that took longer than
        # 0.1 s, so the retire reported "before its timeout" instead of the
        # quiescence refusal this phase is about. cell-a's old-incarnation
        # replay returns without waiting, so a longer window costs nothing.
        monkeypatch.setenv("EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", "10")

        errors = []
        for _ in range(2):
            try:
                service.submit_gang("cell-c", successor)
            except RuntimeError as exc:
                errors.append(exc)

        # Observable safety outcomes first: RED must demonstrate a real launch
        # on different hosts, not merely a missing status() mock invocation.
        assert all(clients[cell_id].submissions == [] for cell_id in ("cell-c", "cell-d")), (
            "successor ranks launched while a predecessor rank was active or indeterminate"
        )
        assert all(
            drivers[cell_id].status("workload-1-a2").status is Status.NOT_FOUND for cell_id in ("cell-c", "cell-d")
        )
        assert all(not managers[cell_id].units for cell_id in ("cell-c", "cell-d"))
        assert_predecessor_retained()
        assert _stop_count(managers["cell-a"]) == before, "an old-incarnation retire-key replay must remain a no-op"
        assert any(managers["cell-a"].background_children.values()), "the relaunched old cgroup remains alive"
        assert len(errors) == 2 and all("still reports" in str(error) for error in errors)
        evidence_executor = service._gang_executor(original.members)
        try:
            evidence = evidence_executor.status("workload-1-a1")
        finally:
            evidence_executor.close()
        assert evidence["status"] is (Status.INDETERMINATE if indeterminate else Status.LOST)
        assert [rank["status"] for rank in evidence["workers"]] == [
            Status.INDETERMINATE if indeterminate else Status.RUNNING,
            Status.STOPPED,
        ]

        # A new retirement command binds to the replacement incarnation. Only
        # positive read-back then allows replay of the unchanged successor keys.
        managers["cell-a"].fail_show = False
        fresh_mutation = {**original_payload["mutation"], "idempotency_key": "fresh-retire-relaunched-rank"}
        retired = clients["cell-a"].post(
            "/api/v1/workloads/workload-1-a1/retire", headers=_HEADERS, json={"mutation": fresh_mutation}
        )
        assert retired.status_code == 200 and retired.json()["status"] == Status.STOPPED.value
        admitted = service.submit_gang("cell-c", successor)
        assert admitted.remote_id == "workload-1-a2" and admitted.status is Status.SUBMITTED
        persisted = FleetStateStore(LocalBackend(state_path)).workload_fence("workload-1")
        assert persisted is not None and "predecessors" not in persisted
        assert all(len(clients[cell_id].submissions) == 1 for cell_id in ("cell-c", "cell-d"))
        assert all(drivers[cell_id].status("workload-1-a2").status is Status.RUNNING for cell_id in ("cell-c", "cell-d"))
        assert all(not any(managers[cell_id].background_children.values()) for cell_id in ("cell-a", "cell-b"))
