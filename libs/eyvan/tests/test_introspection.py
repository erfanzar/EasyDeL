# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Workload process introspection across executor, worker API, and slice fan-out."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from test_container_executor import FakeDockerSystemd  # reuse the stateful docker seam
from test_systemd_transient import FakeSystemd  # reuse the stateful systemd seam

from eyvan.capacity.drivers.base import MutationContext
from eyvan.records import NativeContainerConfig
from eyvan.worker.runtime.container import DockerContainerExecutor
from eyvan.worker.runtime.native_remote import NativeRemoteExecutor
from eyvan.worker.runtime.native_slice import NativeSliceExecutor
from eyvan.worker.runtime.native_worker import create_native_worker_app
from eyvan.worker.runtime.systemd_transient import SystemdTransientServiceExecutor


def _context(key: str = "submit-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=1, fencing_token="token-1")


def _spec(submission_id: str = "workload-1") -> dict:
    return {
        "entrypoint": "python train.py",
        "submission_id": submission_id,
        "environment": {},
        "gang": {"replicas": 1, "per_replica": {"cpu": 1}, "required_topology": {}},
        "metadata": {"eyvan_workload_id": submission_id},
    }


def _executor(tmp_path, daemon, *, pids=None) -> SystemdTransientServiceExecutor:
    return SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=daemon,
        boot_id_reader=lambda: "boot-1",
        cgroup_pids_reader=(lambda _group: pids),
    )


def test_process_status_reports_unit_state_and_cgroup_pids(tmp_path) -> None:
    daemon = FakeSystemd()
    executor = _executor(tmp_path, daemon, pids=(4242, 4243))
    remote_id = executor.submit(_spec(), _context())
    daemon.units[next(iter(daemon.units))].update({"MainPID": "4242", "ControlGroup": "/eyvan.slice/job"})

    status = executor.process_status(remote_id)

    assert status["observed"] is True
    assert status["main_pid"] == 4242
    assert status["pids"] == [4242, 4243]
    assert status["process_count"] == 2
    assert status["control_group"] == "/eyvan.slice/job"


def test_unreadable_cgroup_reports_none_not_an_empty_process_list(tmp_path) -> None:
    """None and [] must stay distinct: '[]' would read as 'nothing is running'."""

    daemon = FakeSystemd()
    executor = _executor(tmp_path, daemon, pids=None)
    remote_id = executor.submit(_spec(), _context())
    daemon.units[next(iter(daemon.units))].update({"ControlGroup": "/eyvan.slice/job"})

    status = executor.process_status(remote_id)

    assert status["pids"] is None
    assert status["process_count"] is None


def test_process_status_on_an_unknown_workload_is_not_observed(tmp_path) -> None:
    executor = _executor(tmp_path, FakeSystemd())

    status = executor.process_status("never-submitted")

    assert status["observed"] is False
    assert "no durable execution record" in status["detail"]


def test_worker_exposes_process_status_and_requires_auth(tmp_path) -> None:
    daemon = FakeSystemd()
    executor = _executor(tmp_path, daemon, pids=(99,))
    remote_id = executor.submit(_spec(), _context())
    app = create_native_worker_app(executor=executor, secret="s3cret")

    with TestClient(app) as client:
        unauthenticated = client.get(f"/api/v1/workloads/{remote_id}/process-status")
        authenticated = client.get(
            f"/api/v1/workloads/{remote_id}/process-status",
            headers={"Authorization": "Bearer s3cret"},
        )

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 200
    assert authenticated.json()["remote_id"] == remote_id


def test_journal_failure_is_a_typed_503_not_an_unhandled_500(tmp_path) -> None:
    """A log transport failure must stay distinguishable from a silent workload."""

    daemon = FakeSystemd()
    daemon.fail_journal = True
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    app = create_native_worker_app(executor=executor, secret="s3cret")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(
            f"/api/v1/workloads/{remote_id}/logs",
            headers={"Authorization": "Bearer s3cret"},
        )

    assert response.status_code == 503
    assert response.json()["code"] == "logs_unavailable"


def test_slice_reports_every_rank_and_keeps_an_unreachable_one_in_place(tmp_path) -> None:
    """A hung job is where one rank differs, so one bad host must not hide the rest."""

    def healthy(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"remote_id": "r-1", "observed": True, "main_pid": 7})

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("host is gone")

    workers = [
        NativeRemoteExecutor(
            "http://rank-0",
            secret="s",
            client=httpx.Client(transport=httpx.MockTransport(healthy), base_url="http://rank-0"),
        ),
        NativeRemoteExecutor(
            "http://rank-1",
            secret="s",
            client=httpx.Client(transport=httpx.MockTransport(unreachable), base_url="http://rank-1"),
        ),
    ]
    slice_executor = NativeSliceExecutor.__new__(NativeSliceExecutor)
    slice_executor.endpoints = ("http://rank-0", "http://rank-1")
    slice_executor._workers = workers

    status = slice_executor.process_status("r-1")

    ranks = status["ranks"]
    assert [entry["rank"] for entry in ranks] == [0, 1]
    assert ranks[0]["status"]["main_pid"] == 7
    assert ranks[1]["status"]["observed"] is False
    assert "unreachable" in ranks[1]["status"]["detail"]
    for worker in workers:
        worker.close()


@pytest.mark.parametrize("raw", ("", "   ", "not-a-number"))
def test_optional_int_never_invents_a_pid(raw: str) -> None:
    from eyvan.worker.runtime.systemd_transient import _optional_int

    assert _optional_int(raw) is None


def _container_executor(tmp_path, daemon) -> DockerContainerExecutor:
    return DockerContainerExecutor(
        tmp_path,
        container=NativeContainerConfig(image="registry.example/easydel:latest", pull_policy="never"),
        artifact_mounts=(),
        base_environment={},
        command_runner=daemon,
        boot_id_reader=lambda: "boot-1",
    )


def test_a_dead_container_is_not_reported_as_a_live_process(tmp_path) -> None:
    """The transient unit holds the ``docker run`` client, not the trainer.

    Inheriting the systemd implementation reads that unit's cgroup, so a
    containerized trainer that has already exited reports ``observed: true``
    with a running unit and the client's PID -- exactly the misdiagnosis this
    endpoint exists to prevent.
    """

    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _container_executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    # The unit stays active with a PID; only the container died.
    daemon.units[next(iter(daemon.units))].update({"MainPID": "4242", "ControlGroup": "/system.slice/unit"})
    container = next(iter(daemon.containers.values()))
    container["State"].update({"Status": "exited", "ExitCode": 137, "Pid": 0})

    status = executor.process_status(remote_id)

    assert status["runtime"] == "native-container"
    assert status["state"] == "exited"
    assert status["exit_code"] == 137
    # Docker reports 0 for a stopped container; 0 is not a PID, and the docker
    # client's 4242 is not the workload's.
    assert status["main_pid"] is None
    assert "active_state" not in status, "reporting the client unit's state would be the inherited bug"


def test_a_containerized_workload_reports_the_container_pid_not_the_client_pid(tmp_path) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _container_executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    daemon.units[next(iter(daemon.units))].update({"MainPID": "4242", "ControlGroup": "/system.slice/unit"})
    next(iter(daemon.containers.values()))["State"]["Pid"] = 90210

    status = executor.process_status(remote_id)

    assert status["observed"] is True
    assert status["state"] == "running"
    assert status["main_pid"] == 90210
    # The host cgroup belongs to the docker client, so the container's process
    # list is genuinely unknown here -- and unknown must not read as empty.
    assert status["pids"] is None
    assert status["process_count"] is None


def test_a_missing_container_is_unobserved_rather_than_inherited_from_the_unit(tmp_path) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _container_executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    daemon.units[next(iter(daemon.units))].update({"MainPID": "4242", "ControlGroup": "/system.slice/unit"})
    daemon.containers.clear()

    status = executor.process_status(remote_id)

    assert status["runtime"] == "native-container"
    assert status["observed"] is False
    # An absent container yields no process facts at all -- not the unit's.
    assert "main_pid" not in status
