# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Running a diagnostic next to a stuck job.

``exec_in_workload`` was declared on the execution-driver protocol with no
implementation anywhere and no route, and ``supports_exec`` was set by two
capacity drivers and read by nothing. Two halves of a feature, neither
connected to the other.

The shape is deliberately narrow, and each narrowing is a decision rather than
an omission:

* **Beside the workload, not inside its cgroup.** A process in that cgroup
  competes for the memory the trainer is sized against, and its death can take
  the trainer with it. A diagnostic that can kill the job it diagnoses is not
  one.
* **One command, not a PTY.** A session needs a bidirectional long-lived socket
  through five synchronous hops, and ``MutationContext`` has no representation
  for a session -- so a stale controller could hold a shell on a cell whose
  lease was revoked.
* **A list of arguments, never a shell string**, because a string is split by a
  shell on the host.
"""

from __future__ import annotations

import subprocess

import pytest

from eyvan.capacity.drivers.base import UnsupportedCapabilityError
from eyvan.worker.agent.models import AgentCapabilities
from eyvan.worker.agent.service import InvalidWorkloadRequestError


@pytest.fixture(autouse=True)
def _isolated_api_home(monkeypatch, tmp_path) -> None:
    """Keep API startup out of the operator's artifact and credential state."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("EYVAN_ARTIFACT_STORE_URI", str(tmp_path / "managed-artifacts"))


def _executor(tmp_path, runner):
    from eyvan.worker.runtime.systemd_transient import SystemdTransientServiceExecutor

    return SystemdTransientServiceExecutor(tmp_path, command_runner=runner, boot_id_reader=lambda: "boot-a")


def _record(executor, remote_id: str, **fields) -> None:
    import json

    path = executor._record_path(remote_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"unit_name": "eyvan-wl-1.service", **fields}), encoding="utf-8")


# --------------------------------------------------------------------------- the driver


def test_the_systemd_executor_implements_the_exec_capability() -> None:
    """The protocol had no implementer at all."""

    from eyvan.worker.runtime.systemd_transient import SystemdTransientServiceExecutor

    assert hasattr(SystemdTransientServiceExecutor, "exec_in_workload")


def test_exec_runs_a_sibling_unit_rather_than_entering_the_workloads_cgroup(tmp_path) -> None:
    """The narrowing that keeps a diagnostic from killing what it diagnoses."""

    seen: list[tuple[str, ...]] = []

    def runner(command, **_kwargs):
        seen.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, "python 3.11\n", "")

    executor = _executor(tmp_path, runner)
    _record(executor, "wl-1", control_group="/eyvan.slice/eyvan-wl-1.service")

    result = executor.exec_in_workload("wl-1", ["python", "--version"], timeout_s=15)

    assert seen[0][0] == "systemd-run", "exec entered the workload's unit instead of running beside it"
    assert "--" in seen[0] and seen[0][-2:] == ("python", "--version")
    assert result["exit_code"] == 0
    assert result["stdout"] == "python 3.11\n"


def test_the_deadline_is_enforced_by_systemd_not_by_this_process(tmp_path) -> None:
    """A bound this agent enforces dies with this agent.

    A subprocess timeout would leave the unit running with nobody waiting for
    it; ``RuntimeMaxSec`` is enforced by the thing that owns the unit's lifetime.
    """

    seen: list[tuple[str, ...]] = []

    def runner(command, **_kwargs):
        seen.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    executor = _executor(tmp_path, runner)
    _record(executor, "wl-1")

    executor.exec_in_workload("wl-1", ["true"], timeout_s=45)

    assert "--property=RuntimeMaxSec=45" in seen[0]


def test_the_source_job_is_identified_so_a_tool_can_target_it(tmp_path) -> None:
    """``py-spy dump --pid`` needs to know which process to look at.

    ``process_status`` already exposes the control group and the PIDs; this is
    what makes those reachable from a command running outside the unit.
    """

    seen: list[tuple[str, ...]] = []

    def runner(command, **_kwargs):
        seen.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    executor = _executor(tmp_path, runner)
    _record(executor, "wl-1", control_group="/eyvan.slice/eyvan-wl-1.service")

    executor.exec_in_workload("wl-1", ["py-spy", "dump"], timeout_s=30)

    joined = " ".join(seen[0])
    assert "--setenv=EYVAN_SOURCE_JOB_ID=wl-1" in joined
    assert "--setenv=EYVAN_SOURCE_CONTROL_GROUP=/eyvan.slice/eyvan-wl-1.service" in joined


def test_cluster_topology_is_not_inherited(tmp_path) -> None:
    """The source workload is still running and still holds its coordinator port.

    Inheriting it would make the diagnostic try to join a cluster it is only
    meant to look at -- the same filter the checkpoint hook applies, for the
    same reason.
    """

    seen: list[tuple[str, ...]] = []

    def runner(command, **_kwargs):
        seen.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    executor = _executor(tmp_path, runner)
    _record(
        executor,
        "wl-1",
        environment={"JAX_COORDINATOR_ADDRESS": "10.0.0.1:1234", "MY_SETTING": "keep-me"},
    )

    executor.exec_in_workload("wl-1", ["true"], timeout_s=10)

    joined = " ".join(seen[0])
    assert "JAX_COORDINATOR_ADDRESS" not in joined
    assert "--setenv=MY_SETTING=keep-me" in joined


def test_an_empty_command_is_refused(tmp_path) -> None:
    """An empty argument list would run systemd-run with no unit to start."""

    executor = _executor(tmp_path, lambda command, **_k: subprocess.CompletedProcess(command, 0, "", ""))
    _record(executor, "wl-1")

    with pytest.raises(ValueError, match="requires a command"):
        executor.exec_in_workload("wl-1", [], timeout_s=10)


def test_exec_against_an_absent_workload_is_refused(tmp_path) -> None:
    """ "Never ran here" must not read as "ran and produced nothing"."""

    executor = _executor(tmp_path, lambda command, **_k: subprocess.CompletedProcess(command, 0, "", ""))

    with pytest.raises(RuntimeError, match="absent"):
        executor.exec_in_workload("wl-missing", ["true"], timeout_s=10)


# --------------------------------------------------------------------------- the two gates


class _ExecCapable:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], float]] = []

    def exec_in_workload(self, remote_id, command, *, timeout_s):
        self.calls.append((remote_id, list(command), timeout_s))
        return {"exit_code": 0, "stdout": "ok", "stderr": ""}


class _NotExecCapable:
    def logs(self, remote_id: str) -> str:
        return ""


def _capabilities(*, supports_exec: bool = False) -> AgentCapabilities:
    """The minimum a capability document needs, so a test states only what it tests."""

    return AgentCapabilities(
        provider="local",
        executor="eyvan-native",
        accelerator_types=(),
        capacity_classes=(),
        supports_checkpoint=True,
        supports_callables=True,
        supports_exec=supports_exec,
        max_gang_replicas=None,
        resource_enforcement="backend",
    )


def _service(executor, *, permitted: bool):
    from eyvan.config import EyvanSettings
    from eyvan.worker.agent.service import CellAgentService

    class _Capacity:
        def capabilities(self):
            return _capabilities(supports_exec=permitted)

    service = CellAgentService.__new__(CellAgentService)
    service.settings = EyvanSettings()
    service.capacity = _Capacity()
    service.execution_backend = "eyvan-native"
    service.executors = None
    return service


def test_a_fleet_that_does_not_permit_exec_says_so(monkeypatch) -> None:
    """The operator's admission decision, distinct from what a backend can do."""

    executor = _ExecCapable()
    service = _service(executor, permitted=False)
    monkeypatch.setattr(type(service), "_executor", lambda _self, _cell: executor)

    with pytest.raises(UnsupportedCapabilityError, match="does not permit exec"):
        service.exec_in_workload("cell-a", "wl-1", ["true"])


def test_a_backend_that_cannot_exec_says_something_different(monkeypatch) -> None:
    """Permitted-but-incapable and capable-but-forbidden are different fixes.

    One is a line of fleet YAML; the other is choosing a different execution
    backend. Collapsing them into one message sends an operator to the wrong
    place.
    """

    executor = _NotExecCapable()
    service = _service(executor, permitted=True)
    monkeypatch.setattr(type(service), "_executor", lambda _self, _cell: executor)

    with pytest.raises(UnsupportedCapabilityError, match="cannot exec"):
        service.exec_in_workload("cell-a", "wl-1", ["true"])


def test_a_permitted_capable_backend_runs_the_command(monkeypatch) -> None:
    """Both gates open is the only combination that runs anything."""

    executor = _ExecCapable()
    service = _service(executor, permitted=True)
    monkeypatch.setattr(type(service), "_executor", lambda _self, _cell: executor)

    result = service.exec_in_workload("cell-a", "wl-1", ["py-spy", "dump"], timeout_s=12)

    assert executor.calls == [("wl-1", ["py-spy", "dump"], 12)]
    assert result["exit_code"] == 0


def test_an_empty_command_is_refused_at_the_agent(monkeypatch) -> None:
    """422 rather than a unit that starts nothing."""

    executor = _ExecCapable()
    service = _service(executor, permitted=True)
    monkeypatch.setattr(type(service), "_executor", lambda _self, _cell: executor)

    with pytest.raises(InvalidWorkloadRequestError):
        service.exec_in_workload("cell-a", "wl-1", [])


def test_the_admission_flag_is_refused_by_default_and_stated_explicitly() -> None:
    """A default-on capability nothing read is precisely how this became dead.

    Two layers, and they are different on purpose. The *driver* capability
    defaults to refusing, so a backend that never thought about exec does not
    silently permit it. The *wire* model has no default at all, so an agent
    cannot report a capability document that omits its answer -- which is
    stronger than a default, because a missing field and "no" stop looking the
    same on the wire.
    """

    from pydantic import ValidationError

    from eyvan.capacity.drivers.base import CellCapabilities

    assert CellCapabilities.__dataclass_fields__["supports_exec"].default is False

    with pytest.raises(ValidationError):
        AgentCapabilities(
            provider="local",
            executor="eyvan-native",
            accelerator_types=(),
            capacity_classes=(),
            supports_checkpoint=True,
            supports_callables=True,
            max_gang_replicas=None,
            resource_enforcement="backend",
        )


def test_a_command_is_a_list_not_a_shell_string() -> None:
    """A string would be split by a shell on the host.

    That turns a diagnostic into an injection surface the moment any part of it
    is templated from a workload id or a user-supplied name.
    """

    from pydantic import ValidationError

    from eyvan.worker.agent.models import ExecRequest

    assert ExecRequest(command=("py-spy", "dump")).command == ("py-spy", "dump")
    with pytest.raises(ValidationError):
        ExecRequest(command=())


# --------------------------------------------------------------------------- init inside a job


def test_init_does_not_start_a_control_plane_on_a_worker(monkeypatch) -> None:
    """``init()`` inside a job would otherwise try to start a daemon on a TPU host.

    The default base URL is loopback, and loopback plus "no client configured"
    is the local-development signal -- so a callable that calls ``init()`` to get
    a client would have started a second control plane on the worker, fighting
    for the state directory and answering with an empty fleet.
    """

    from eyvan.sdk import execution

    started: list[object] = []
    monkeypatch.setattr(execution, "ensure_loopback_control_plane", lambda endpoint: started.append(endpoint))
    monkeypatch.setattr(execution, "_configured_gcp_runtime_profile", lambda _endpoint: None)

    execution.init(
        environ={
            "EYVAN_WORKLOAD_ID": "wl-1",
            "EYVAN_CHILDREN_URL": "https://eyvan.test/api/v1/workloads/wl-1/children",
            "EYVAN_WORKLOAD_TOKEN": "token-value",
        }
    )

    assert started == [], "init started a control plane inside a running job"


def test_init_still_starts_a_control_plane_on_a_laptop(tmp_path, monkeypatch) -> None:
    """The local-development path must keep working; only the in-job case changes.

    Pinned to a missing context file because ``init()`` probes the active
    fleet recorded in the operator's own context: with a real context this
    test depended on whatever plane happened to be listening on 8080 --
    which on a developer machine was once a plane leaked by an earlier
    test. With no fleet to reconcile, the start attempt is the whole of the
    contract under test.
    """

    from eyvan.sdk import execution

    started: list[object] = []
    monkeypatch.setattr(execution, "ensure_loopback_control_plane", lambda endpoint: started.append(endpoint))
    monkeypatch.setattr(execution, "_configured_gcp_runtime_profile", lambda _endpoint: None)

    execution.init(environ={"EYVAN_CONFIG_FILE": str(tmp_path / "no-context.json")})

    assert started, "init stopped starting the local control plane"


# --------------------------------------------------------------------------- the control plane


class _RankAgent:
    """An agent that records which cell it was asked to exec on."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.asked: list[tuple[str, tuple[str, ...]]] = []
        self.fail_on = fail_on

    def exec_in_workload(self, cell_id, remote_id, command, *, timeout_s=30.0):
        del remote_id, timeout_s
        self.asked.append((cell_id, tuple(command)))
        if cell_id == self.fail_on:
            raise RuntimeError("agent unreachable")
        return {"exit_code": 0, "stdout": f"from {cell_id}", "stderr": ""}


async def _gang(agent):
    """A two-cell gang, which is the shape rank fan-out exists for."""

    from datetime import timedelta

    from eyvan.config import EyvanSettings
    from eyvan.records import (
        CellRecord,
        CellState,
        CommandExecution,
        LeaseRecord,
        LeaseState,
        ResourceGang,
        ResourceVector,
        WorkloadAttemptTopology,
        WorkloadDefinition,
        WorkloadRecord,
        WorkloadState,
        WorkloadTopologyCell,
        utcnow,
    )
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    controller = EyvanController(
        repository,
        ControlPlaneService(repository, scheduler=CoreSchedulingService()),
        settings=EyvanSettings(),
        agent_factory=lambda _cell: agent,
    )
    async with repository.transaction() as transaction:
        for index, cell_id in enumerate(("cell-a", "cell-b")):
            await transaction.put_cell(
                CellRecord(
                    id=cell_id,
                    provider="local",
                    region="local",
                    state=CellState.READY,
                    capacity=ResourceVector(cpu=8),
                    allocatable=ResourceVector(cpu=8),
                    endpoint="http://agent:8090",
                )
            )
            await transaction.put_lease(
                LeaseRecord(
                    id=f"lease-{index}",
                    workload_id="wl-gang",
                    cell_id=cell_id,
                    org_id="acme",
                    project_id="research",
                    state=LeaseState.ACTIVE,
                    resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
                    generation=1,
                    fencing_token=f"token-{index}",
                    expires_at=utcnow() + timedelta(hours=24),
                )
            )
        await transaction.put_workload(
            WorkloadRecord(
                id="wl-gang",
                org_id="acme",
                project_id="research",
                spec=WorkloadDefinition(name="t", execution=CommandExecution(command="python train.py")),
                state=WorkloadState.RUNNING,
                lease_id="lease-0",
                lease_ids=["lease-0", "lease-1"],
                remote_id="remote-1",
                created_by="alice",
            )
        )
        await transaction.put_workload_topology(
            WorkloadAttemptTopology.freeze(
                workload_id="wl-gang",
                attempt=1,
                generation=1,
                cells=[
                    WorkloadTopologyCell(
                        cell_id=cell_id,
                        generation=1,
                        runtime_process_count=1,
                        global_rank_offset=0,
                    )
                    for cell_id in ("cell-a", "cell-b")
                ],
            )
        )
    return controller


async def test_exec_reaches_every_rank_by_default() -> None:
    """One rank cannot distinguish a hung collective from a slow step."""

    agent = _RankAgent()
    controller = await _gang(agent)

    result = await controller.workload_exec("wl-gang", ["py-spy", "dump"])

    assert [cell for cell, _command in agent.asked] == ["cell-a", "cell-b"]
    assert [entry["rank"] for entry in result["results"]] == [0, 1]


async def test_one_unreachable_rank_does_not_lose_the_others() -> None:
    """A result set missing the rank you wanted is still better than an exception.

    The failure is reported beside the ranks that answered rather than replacing
    them.
    """

    agent = _RankAgent(fail_on="cell-b")
    controller = await _gang(agent)

    result = await controller.workload_exec("wl-gang", ["true"])

    assert result["results"][0]["stdout"] == "from cell-a"
    assert "unreachable" in result["results"][1]["error"]


async def test_a_named_rank_narrows_the_fan_out() -> None:
    """Once the answer is known, asking one host is the cheaper question."""

    agent = _RankAgent()
    controller = await _gang(agent)

    result = await controller.workload_exec("wl-gang", ["true"], rank=1)

    assert [cell for cell, _command in agent.asked] == ["cell-b"]
    assert [entry["rank"] for entry in result["results"]] == [1]


async def test_a_rank_that_does_not_exist_is_refused() -> None:
    """An empty result set would read as a workload that answered nothing."""

    from eyvan.server.service.errors import ConflictError

    agent = _RankAgent()
    controller = await _gang(agent)

    with pytest.raises(ConflictError, match="no frozen rank 7"):
        await controller.workload_exec("wl-gang", ["true"], rank=7)
    assert agent.asked == []


async def test_exec_against_a_workload_with_no_execution_is_refused() -> None:
    """Queued work has no host to run a diagnostic on."""

    from eyvan.config import EyvanSettings
    from eyvan.records import CommandExecution, WorkloadDefinition, WorkloadRecord
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.errors import ConflictError
    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    controller = EyvanController(
        repository,
        ControlPlaneService(repository, scheduler=CoreSchedulingService()),
        settings=EyvanSettings(),
    )
    async with repository.transaction() as transaction:
        await transaction.put_workload(
            WorkloadRecord(
                id="wl-pending",
                org_id="acme",
                project_id="research",
                spec=WorkloadDefinition(name="t", execution=CommandExecution(command="echo hi")),
                created_by="alice",
            )
        )

    with pytest.raises(ConflictError, match="not running"):
        await controller.workload_exec("wl-pending", ["true"])


def test_a_viewer_cannot_run_commands_on_the_fleet(tmp_path) -> None:
    """OPERATE, not READ: this starts a process on a host running someone's job.

    It is not a mutation of the workload, and it is not an observation either.
    Treating it as one would let read-only access execute code on the fleet.
    """

    import asyncio

    from fastapi.testclient import TestClient

    from eyvan.config import EyvanSettings
    from eyvan.records import CommandExecution, WorkloadDefinition, WorkloadRecord
    from eyvan.server.api.app import create_app

    app = create_app(settings=EyvanSettings(dev_auth_enabled=True, artifact_token_key="k" * 32))

    async def store():
        async with app.state.service.repository.transaction() as transaction:
            await transaction.put_workload(
                WorkloadRecord(
                    id="wl-1",
                    org_id="acme",
                    project_id="research",
                    spec=WorkloadDefinition(name="t", execution=CommandExecution(command="echo hi")),
                    created_by="alice",
                )
            )

    asyncio.run(store())

    def attempt(role: str):
        return TestClient(app).post(
            "/api/v1/workloads/wl-1/exec",
            json={"command": ["id"]},
            headers={
                "X-Eyvan-User": "bob",
                "X-Eyvan-Roles": role,
                "X-Eyvan-Org": "acme",
                "X-Eyvan-Project": "research",
            },
        )

    viewer, operator = attempt("viewer"), attempt("operator")

    assert viewer.status_code == 403
    assert "operate" in viewer.json()["detail"]
    # The operator gets *past* authorization, which is what makes this a test of
    # the gate rather than of something else refusing first: 409 is the workload
    # having no cell, not a permission answer.
    assert operator.status_code == 409


def test_exec_on_a_workload_whose_lease_is_gone_is_a_conflict_not_a_crash(tmp_path) -> None:
    """A terminal workload keeps its remote id and loses its lease.

    "Has a remote id" is therefore not "is still somewhere", and letting the
    internal staleness error escape turned an ordinary race -- the job finished
    between the read and the exec -- into a 500.
    """

    import asyncio

    from fastapi.testclient import TestClient

    from eyvan.config import EyvanSettings
    from eyvan.records import CommandExecution, WorkloadDefinition, WorkloadRecord, WorkloadState
    from eyvan.server.api.app import create_app

    app = create_app(settings=EyvanSettings(dev_auth_enabled=True, artifact_token_key="k" * 32))

    async def store():
        async with app.state.service.repository.transaction() as transaction:
            await transaction.put_workload(
                WorkloadRecord(
                    id="wl-1",
                    org_id="acme",
                    project_id="research",
                    spec=WorkloadDefinition(name="t", execution=CommandExecution(command="echo hi")),
                    state=WorkloadState.RUNNING,
                    remote_id="remote-1",
                    created_by="alice",
                )
            )

    asyncio.run(store())

    response = TestClient(app).post(
        "/api/v1/workloads/wl-1/exec",
        json={"command": ["id"]},
        headers={
            "X-Eyvan-User": "alice",
            "X-Eyvan-Roles": "operator",
            "X-Eyvan-Org": "acme",
            "X-Eyvan-Project": "research",
        },
    )

    assert response.status_code == 409
    assert "no longer holds a cell" in response.json()["detail"]
