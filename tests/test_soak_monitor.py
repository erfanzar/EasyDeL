# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The external P8 monitor: what it measures, and what it refuses to make up.

The evaluator has always been able to fail closed. What did not exist was a
producer, so every live P8 invocation ended with six ``*_evidence_missing``
codes and no way to clear them. These tests hold the producer to the property
that makes it worth having: it measures a *different* process from outside it,
it reads the ledger and the fleet rather than deciding what they should say,
and the instant one of its six measurements is unavailable it emits nothing at
all rather than a zero.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import typing as tp
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

import eyvan.sdk.client.transport as client_transport
from eyvan.config import EyvanSettings
from eyvan.records import (
    BudgetRecord,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    ResourceGang,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.sdk import cli
from eyvan.sdk.commands.soak import SoakRunInputs, build_soak_receipt, run_soak
from eyvan.sdk.commands.soak_monitor import (
    MAX_CLEANUP_CELLS,
    MAX_INVENTORY_OUTPUT_BYTES,
    MAX_PID_FILE_BYTES,
    measure_local_state_bytes,
    measure_postgres_database_bytes,
    measure_process_fd_count,
    measure_process_rss_bytes,
    measure_provider_inventory,
    read_cleanup_complete,
    read_pid_file,
    read_spend_reconciled,
)
from eyvan.server.api.app import create_app
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.soak_contract import (
    BackendIdentity,
    CapabilityIdentity,
    ExecutionIdentity,
    ExpectedExecution,
    ParticipantBinding,
    ProviderIdentity,
    participant_manifest_sha256,
)
from eyvan.soak_evidence import EXTERNAL_EVIDENCE_MEASURED_FIELDS, build_external_evidence_sample

PROOF_ID = "sha256:" + "d" * 64
OTHER_PROOF_ID = "sha256:" + "e" * 64
CONFIG_SHA256 = "sha256:" + "c" * 64
ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
ADMIN_ARGS = ["--url", "http://127.0.0.1:8080", "--user", "root", "--role", "fleet_admin"]

_CALLABLE_CAPABILITY = CapabilityIdentity(
    execution="callable",
    checkpointable=True,
    coordination="none",
    required_executor="eyvan-native",
)
_CPU_IDENTITY = ExecutionIdentity(
    capability=_CALLABLE_CAPABILITY,
    backend=BackendIdentity(capacity_backend="gcp-compute", executor="eyvan-native"),
    provider=ProviderIdentity(provider="gcp", resource_class="cpu", capacity_class="spot"),
)
_TPU_IDENTITY = ExecutionIdentity(
    capability=_CALLABLE_CAPABILITY,
    backend=BackendIdentity(capacity_backend="gcp-tpu", executor="eyvan-native"),
    provider=ProviderIdentity(provider="gcp", resource_class="tpu", capacity_class="spot"),
)
_CPU_PARTICIPANT = ParticipantBinding(workload_id="wl-cpu", lease_id="lease-cpu", cell_id="cell-cpu")
_TPU_PARTICIPANT = ParticipantBinding(workload_id="wl-tpu", lease_id="lease-tpu", cell_id="cell-tpu")


# --------------------------------------------------------------------------- harness


class _Recorder:
    """A control plane behind the real CLI transport, with its traffic recorded."""

    def __init__(self, app: tp.Any) -> None:
        self.app = app
        self.requests: list[tuple[str, str]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = self

        def factory(*, base_url: str, headers: dict[str, str], timeout: float, trust_env: bool) -> TestClient:
            del timeout, trust_env
            client = TestClient(recorder.app, base_url=base_url, headers=dict(headers))
            client.event_hooks = {"request": [recorder.record]}
            return client

        monkeypatch.setattr(client_transport.httpx, "Client", factory)

    def record(self, request: tp.Any) -> None:
        self.requests.append((request.method, request.url.path))


def _app(tmp_path: Path, repository: InMemoryRepository | None = None) -> tp.Any:
    return create_app(
        repository=repository,
        settings=EyvanSettings(
            dev_auth_enabled=True,
            artifact_token_key="k" * 32,
            artifact_store_uri=str(tmp_path / "artifacts"),
        ),
    )


def _state_document(tmp_path: Path, size: int = 4_096) -> Path:
    path = tmp_path / "control-plane.json"
    path.write_bytes(b"{}" + b" " * (size - 2))
    return path


def _monitor_args(
    tmp_path: Path,
    *,
    proof_id: str = PROOF_ID,
    pid_file: Path | None = None,
    state_path: Path | None = None,
    inventory: str = "/bin/echo 0",
) -> list[str]:
    args = [
        *ADMIN_ARGS,
        "soak-monitor",
        "--proof-id",
        proof_id,
        "--provider-inventory-command",
        inventory,
        "--samples",
        "1",
        "--interval-s",
        "0",
    ]
    if pid_file is not None:
        args += ["--control-plane-pid-file", str(pid_file)]
    else:
        args += ["--control-plane-pid", str(os.getpid())]
    args += ["--state-path", str(_state_document(tmp_path) if state_path is None else state_path)]
    return args


def _lines(output: str) -> list[dict[str, tp.Any]]:
    return [json.loads(line) for line in output.splitlines() if line.startswith("{")]


def _published_evidence(app: tp.Any, proof_id: str = PROOF_ID) -> object:
    response = TestClient(app).post(
        "/api/v1/system/invariants/soak",
        json={
            "proof_id": proof_id,
            "config_sha256": CONFIG_SHA256,
            "participants": [participant.as_request_dict() for participant in (_CPU_PARTICIPANT, _TPU_PARTICIPANT)],
        },
        headers=ADMIN,
    )
    assert response.status_code == 200, response.text
    return response.json()["snapshot"]["p8_evidence"]


# --------------------------------------------------------------------------- measuring another process


def test_the_monitor_measures_a_different_live_process_from_outside_it() -> None:
    """A process reporting its own RSS proves nothing about one that is leaking.

    So the probes are pointed at a real child process here, not at the test
    interpreter: the number has to come from the operating system's view of
    another pid, and it has to disappear when that pid does.
    """

    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10.0
        rss = measure_process_rss_bytes(child.pid, timeout_s=10.0)
        while rss is None and time.monotonic() < deadline:
            rss = measure_process_rss_bytes(child.pid, timeout_s=10.0)
        descriptors = measure_process_fd_count(child.pid, timeout_s=30.0)

        assert rss is not None and rss > 0
        assert descriptors is not None and descriptors >= 3
    finally:
        if child.stdin is not None:
            child.stdin.close()
        child.wait(timeout=30)

    assert measure_process_rss_bytes(child.pid, timeout_s=10.0) is None
    assert measure_process_fd_count(child.pid, timeout_s=30.0) is None


def test_a_state_document_that_cannot_be_read_is_unknown_and_not_zero_bytes(tmp_path: Path) -> None:
    """A mistyped path and an empty database are the same observation from here.

    Only one of them is evidence, so neither is reported as a measurement.
    """

    assert measure_local_state_bytes(tmp_path / "absent.json") is None
    assert measure_local_state_bytes(_state_document(tmp_path, size=1_024)) == 1_024


def test_an_unreachable_database_is_unknown_rather_than_zero_bytes() -> None:
    """A refused connection says nothing about how large the database is."""

    unreachable = "postgresql+asyncpg://eyvan:eyvan@127.0.0.1:1/eyvan-does-not-exist"

    assert measure_postgres_database_bytes(unreachable, timeout_s=5.0) is None


@pytest.mark.parametrize(
    "command",
    [
        "/bin/echo not-a-number",
        "/bin/echo -1",
        "/usr/bin/false",
        "/nonexistent/inventory-probe",
        "",
    ],
)
def test_a_provider_inventory_probe_that_does_not_answer_is_unknown(command: str) -> None:
    """ "The provider did not answer" and "the provider has nothing" differ.

    Conflating them is how a soak reports an empty final inventory while the
    account is still being billed.
    """

    assert measure_provider_inventory(command, timeout_s=10.0) is None


def test_a_provider_inventory_probe_reports_exactly_what_it_printed() -> None:
    assert measure_provider_inventory("/bin/echo 0", timeout_s=10.0) == 0
    assert measure_provider_inventory("/bin/echo 17", timeout_s=10.0) == 17


def test_a_provider_inventory_probe_that_floods_stdout_is_unknown(tmp_path: Path) -> None:
    """A count buried in megabytes of other output is not a count.

    Bounded before parsing rather than after, so a command that streams is
    refused instead of scanned.
    """

    script = tmp_path / "flood.sh"
    script.write_text(f"#!/bin/sh\nprintf '7'\nprintf '%{MAX_INVENTORY_OUTPUT_BYTES + 16}s' ''\n")
    script.chmod(0o755)

    assert measure_provider_inventory(str(script), timeout_s=10.0) is None


def test_a_cell_listing_past_its_bound_is_unknown_rather_than_judged() -> None:
    """A response this long is not a fleet inventory, so it settles nothing."""

    oversized = [{"id": f"cell-{index}", "state": CellState.DELETED.value} for index in range(MAX_CLEANUP_CELLS + 1)]

    assert read_cleanup_complete(_StubClient({"/api/v1/cells": oversized})) is None


def test_a_pid_file_is_re_read_so_a_restarted_control_plane_is_still_measured(tmp_path: Path) -> None:
    """A P8 run restarts the control plane deliberately.

    A pid captured once names a dead process for the rest of the soak, which
    would silently suppress the resource half of every remaining sample.
    """

    pid_file = tmp_path / "control-plane.pid"

    assert read_pid_file(pid_file) is None

    pid_file.write_text("4321\n")
    assert read_pid_file(pid_file) == 4321

    pid_file.write_text("9999\n")
    assert read_pid_file(pid_file) == 9999

    pid_file.write_text("not-a-pid")
    assert read_pid_file(pid_file) is None

    pid_file.write_text("0")
    assert read_pid_file(pid_file) is None

    pid_file.write_text("1" * (MAX_PID_FILE_BYTES + 1))
    assert read_pid_file(pid_file) is None


# --------------------------------------------------------------------------- reading the ledger and the fleet


class _StubClient:
    """Returns canned documents, or raises, for one path at a time."""

    def __init__(self, documents: dict[str, object]) -> None:
        self.documents = documents

    def request(self, method: str, path: str, **_kwargs: object) -> object:
        assert method == "GET"
        if path not in self.documents:
            raise RuntimeError("the control plane is unavailable")
        return self.documents[path]


def _ledger(
    *,
    violations: list[dict[str, object]] | None = None,
    truncated: bool = False,
    unpriced: int = 0,
) -> dict[str, object]:
    return {
        "/api/v1/system/invariants": {
            "checked": [f"I{number}" for number in range(1, 10)],
            "violations": violations or [],
            "violations_truncated": truncated,
        },
        "/api/v1/capacity/billing": {"cells": 4, "unpriced_cells": unpriced},
    }


def test_a_reconciled_ledger_with_priced_cells_reads_as_reconciled() -> None:
    assert read_spend_reconciled(_StubClient(_ledger())) is True


def test_an_i5_violation_reads_as_unreconciled() -> None:
    """I5 is exactly the claim "spend equals what the settled leases cost"."""

    violation = {"invariant": "I5", "subject": "acme/research", "detail": "spend disagrees"}

    assert read_spend_reconciled(_StubClient(_ledger(violations=[violation]))) is False


def test_a_truncated_violation_list_is_unknown_rather_than_reconciled() -> None:
    """The I5 finding may be one of the ones the server dropped."""

    assert read_spend_reconciled(_StubClient(_ledger(truncated=True))) is None


def test_a_plane_that_does_not_check_i5_leaves_reconciliation_unknown() -> None:
    """An older plane that never evaluated I5 has not reported that it holds."""

    documents = _ledger()
    documents["/api/v1/system/invariants"]["checked"] = ["I1", "I2"]  # type: ignore[index]

    assert read_spend_reconciled(_StubClient(documents)) is None


def test_unpriced_cells_are_not_a_reconciled_ledger() -> None:
    """The terminated P8 accrued $3.68 with 42 of 46 cells unpriced.

    That number is a floor, not a total, and a receipt calling it reconciled
    would be asserting a cost nobody measured.
    """

    assert read_spend_reconciled(_StubClient(_ledger(unpriced=42))) is False


def test_an_unreachable_control_plane_leaves_the_ledger_and_the_fleet_unknown() -> None:
    assert read_spend_reconciled(_StubClient({})) is None
    assert read_cleanup_complete(_StubClient({})) is None


def test_cleanup_is_complete_only_when_every_cell_has_been_released() -> None:
    released = [{"id": "cell-a", "state": CellState.DELETED.value}]
    draining = [{"id": "cell-a", "state": CellState.DELETED.value}, {"id": "cell-b", "state": CellState.DRAINING.value}]
    deleting = [{"id": "cell-b", "state": CellState.DELETING.value}]

    assert read_cleanup_complete(_StubClient({"/api/v1/cells": []})) is True
    assert read_cleanup_complete(_StubClient({"/api/v1/cells": released})) is True
    assert read_cleanup_complete(_StubClient({"/api/v1/cells": draining})) is False
    # DELETING is a provider call still in flight, which is capacity still owned.
    assert read_cleanup_complete(_StubClient({"/api/v1/cells": deleting})) is False


def test_a_cell_state_this_build_does_not_recognise_is_unknown_not_cleaned_up() -> None:
    """A rolling upgrade must not turn a state it cannot read into "done"."""

    future = [{"id": "cell-a", "state": "quiescing"}]

    assert read_cleanup_complete(_StubClient({"/api/v1/cells": future})) is None


# --------------------------------------------------------------------------- a partial sample is not a sample


@pytest.mark.parametrize("field", EXTERNAL_EVIDENCE_MEASURED_FIELDS)
def test_any_single_unmeasured_field_suppresses_the_whole_sample(field: str) -> None:
    """There is no encoding for "unknown", so the answer is silence.

    Substituting a zero would be a measurement nobody took; carrying the
    previous value forward would republish an old observation as a new one.
    """

    measured: dict[str, tp.Any] = {
        "rss_bytes": 1,
        "fd_count": 2,
        "database_bytes": 3,
        "spend_reconciled": True,
        "provider_inventory_count": 4,
        "cleanup_complete": True,
    }
    measured[field] = None

    sample, unmeasurable = build_external_evidence_sample(PROOF_ID, **measured)

    assert sample is None
    assert unmeasurable == (field,)


def test_a_complete_set_of_measurements_becomes_exactly_one_sample() -> None:
    sample, unmeasurable = build_external_evidence_sample(
        PROOF_ID,
        rss_bytes=1,
        fd_count=2,
        database_bytes=3,
        spend_reconciled=False,
        provider_inventory_count=4,
        cleanup_complete=True,
    )

    assert unmeasurable == ()
    assert sample is not None
    assert sample.as_dict()["rss_bytes"] == 1


@pytest.mark.parametrize(
    ("broken", "expected"),
    [
        ("inventory", ["provider_inventory_count"]),
        ("state", ["database_bytes"]),
        ("pid", ["fd_count", "rss_bytes"]),
    ],
)
def test_the_command_posts_nothing_when_a_field_cannot_be_measured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    broken: str,
    expected: list[str],
) -> None:
    """Nothing reaches the control plane, and the evidence stays absent.

    The evaluator then reports its own ``*_evidence_missing`` codes, which is
    the honest outcome for a measurement that was never taken.
    """

    app = _app(tmp_path)
    recorder = _Recorder(app)
    recorder.install(monkeypatch)
    overrides: dict[str, tp.Any] = {}
    if broken == "inventory":
        overrides["inventory"] = "/usr/bin/false"
    elif broken == "state":
        overrides["state_path"] = tmp_path / "never-written.json"
    else:
        overrides["pid_file"] = tmp_path / "absent.pid"

    result = CliRunner().invoke(cli.main, _monitor_args(tmp_path, **overrides))

    assert result.exit_code == 1, result.output
    assert _lines(result.output) == [{"outcome": "incomplete", "unmeasurable": expected}]
    assert ("POST", "/api/v1/system/soak-evidence") not in recorder.requests
    assert _published_evidence(app) is None


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], "exactly one of --control-plane-pid or --control-plane-pid-file"),
        (["--control-plane-pid", "1", "--control-plane-pid-file", "/tmp/x.pid"], "exactly one of --control-plane-pid"),
        (["--control-plane-pid", "1"], "exactly one of --state-path or --database-url"),
        (
            ["--control-plane-pid", "1", "--state-path", "/tmp/s.json", "--database-url", "postgresql://x/y"],
            "exactly one of --state-path or --database-url",
        ),
        (["--control-plane-pid", "0", "--state-path", "/tmp/s.json"], "positive process identifier"),
    ],
)
def test_an_ambiguous_or_absent_measurement_source_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    expected: str,
) -> None:
    """Two database sources is not "pick one"; no pid source is not "measure me".

    Both are the operator saying something the monitor cannot honour, and
    guessing would produce a sample about the wrong process or the wrong store.
    """

    reached: list[str] = []
    monkeypatch.setattr(client_transport.httpx, "Client", lambda **_kwargs: reached.append("built"))

    result = CliRunner().invoke(
        cli.main,
        [
            *ADMIN_ARGS,
            "soak-monitor",
            "--proof-id",
            PROOF_ID,
            "--provider-inventory-command",
            "/bin/echo 0",
            "--samples",
            "1",
            *args,
        ],
    )

    assert result.exit_code == 2, result.output
    assert expected in result.output


def test_a_proof_id_that_is_not_an_opaque_identifier_is_a_usage_error(tmp_path: Path) -> None:
    """An unbound sample cannot be evidence for any run, so it is never taken."""

    args = _monitor_args(tmp_path, proof_id="run-7")

    result = CliRunner().invoke(cli.main, args)

    assert result.exit_code == 2, result.output
    assert "--proof-id must be a lowercase sha256 opaque identifier" in result.output


# --------------------------------------------------------------------------- the whole round trip


def test_a_well_formed_sample_travels_from_the_monitor_to_a_passing_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Producer to control plane to evaluator, with nothing synthesised between.

    The sample the evaluator scores here is the one the real command measured
    and the real route republished; no fixture stands in for either half. That
    closes the evidence gap and nothing else: the receipt still owes a mixed
    CPU+TPU fleet, 72 unbroken hours, and provider evidence of its own.
    """

    app = _app(tmp_path)
    _Recorder(app).install(monkeypatch)

    result = CliRunner().invoke(cli.main, _monitor_args(tmp_path))
    assert result.exit_code == 0, result.output
    emitted = _lines(result.output)
    assert [line["outcome"] for line in emitted] == ["emitted"]

    republished = _published_evidence(app)
    assert republished == emitted[0]["sample"]

    receipt = _evaluate(tp.cast("dict[str, tp.Any]", republished))

    assert receipt["verdict"] == "pass"
    assert receipt["exit_code"] == 0
    assert receipt["reasons"] == []


def test_evidence_bound_to_another_run_leaves_this_run_without_any(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A monitor pointed at the wrong proof cannot accidentally qualify this one."""

    app = _app(tmp_path)
    _Recorder(app).install(monkeypatch)

    result = CliRunner().invoke(cli.main, _monitor_args(tmp_path, proof_id=OTHER_PROOF_ID))
    assert result.exit_code == 0, result.output

    assert _published_evidence(app, proof_id=PROOF_ID) is None

    receipt = _evaluate(None)

    assert receipt["verdict"] == "inconclusive"
    assert receipt["exit_code"] == 2
    assert set(tp.cast("list[str]", receipt["reasons"])) >= {
        "cleanup_evidence_missing",
        "database_growth_evidence_missing",
        "fd_growth_evidence_missing",
        "p8_evidence_binding_mismatch",
        "p8_evidence_missing",
        "provider_inventory_evidence_missing",
        "rss_growth_evidence_missing",
        "spend_reconciliation_evidence_missing",
    }


def test_a_control_plane_that_refuses_the_sample_is_reported_and_not_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plane that is restarting is not evidence about the fleet.

    The tick is reported as undelivered, the next one is still taken, and the
    evaluator sees the gap as missing evidence -- which is what it is. A monitor
    that exited here would turn one unavailable moment into no evidence at all
    for the rest of a 72-hour run.
    """

    served: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        served.append(f"{request.method} {request.url.path}")
        if request.url.path == "/api/v1/system/invariants":
            return httpx.Response(
                200,
                json={
                    "checked": [f"I{number}" for number in range(1, 10)],
                    "violations": [],
                    "violations_truncated": False,
                },
            )
        if request.url.path == "/api/v1/capacity/billing":
            return httpx.Response(200, json={"cells": 0, "unpriced_cells": 0})
        if request.url.path == "/api/v1/cells":
            return httpx.Response(200, json=[])
        return httpx.Response(503, json={"detail": "the control plane is restarting"})

    # Bound before patching: the factory below would otherwise call itself.
    real_client = httpx.Client

    def factory(*, base_url: str, headers: dict[str, str], timeout: float, trust_env: bool) -> httpx.Client:
        del timeout, trust_env
        return real_client(base_url=base_url, headers=dict(headers), transport=httpx.MockTransport(handler))

    monkeypatch.setattr(client_transport.httpx, "Client", factory)
    args = _monitor_args(tmp_path)
    args[args.index("--samples") + 1] = "2"

    result = CliRunner().invoke(cli.main, args)

    assert result.exit_code == 1, result.output
    assert _lines(result.output) == [
        {"outcome": "not_delivered", "unmeasurable": []},
        {"outcome": "not_delivered", "unmeasurable": []},
    ]
    assert served.count("POST /api/v1/system/soak-evidence") >= 2


# --------------------------------------------------------------------------- read-only


async def test_the_monitor_changes_no_fleet_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Its only write is the evidence slot, which is not fleet state.

    A monitor that drained a cell, settled a lease, or appended an audit event
    while measuring would be observing a fleet it had itself disturbed.
    """

    repository = await _seeded_fleet()
    before = await _durable_state(repository)
    app = _app(tmp_path, repository)
    recorder = _Recorder(app)
    recorder.install(monkeypatch)

    result = CliRunner().invoke(cli.main, _monitor_args(tmp_path))

    assert result.exit_code == 0, result.output
    assert await _durable_state(repository) == before
    assert recorder.requests == [
        ("GET", "/api/v1/system/invariants"),
        ("GET", "/api/v1/capacity/billing"),
        ("GET", "/api/v1/cells"),
        ("POST", "/api/v1/system/soak-evidence"),
    ]


async def test_a_fleet_that_still_owns_capacity_is_not_reported_as_cleaned_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two halves have to disagree when the fleet and the provider do.

    ``cleanup_complete`` is the control plane's own view, and a live cell means
    the fleet still owns something whatever the inventory command reports.
    """

    repository = await _seeded_fleet()
    app = _app(tmp_path, repository)
    _Recorder(app).install(monkeypatch)

    result = CliRunner().invoke(cli.main, _monitor_args(tmp_path))

    assert result.exit_code == 0, result.output
    sample = tp.cast("dict[str, tp.Any]", _lines(result.output)[0]["sample"])
    assert sample["cleanup_complete"] is False
    assert sample["provider_inventory_count"] == 0


# --------------------------------------------------------------------------- fixtures


async def _seeded_fleet() -> InMemoryRepository:
    """One live cell, one workload, one lease and one budget."""

    repository = InMemoryRepository()
    async with repository.transaction() as tx:
        await tx.put_capacity_pool(
            CapacityPoolRecord(
                id="pool-a",
                fleet_name="research",
                node_type="cpu",
                provider="local",
                region="local",
                capacity=ResourceVector(cpu=8, memory_gib=64),
                allocatable=ResourceVector(cpu=8, memory_gib=64),
                agent_endpoint="http://agent:8090",
            )
        )
        await tx.put_cell(
            CellRecord(
                id="cell-cpu",
                provider="local",
                region="local",
                state=CellState.READY,
                pool_id="pool-a",
                capacity=ResourceVector(cpu=8, memory_gib=64),
                allocatable=ResourceVector(cpu=8, memory_gib=64),
                endpoint="http://agent:8090",
            )
        )
        await tx.put_workload(
            WorkloadRecord(
                id="wl-cpu",
                org_id="acme",
                project_id="research",
                spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
                state=WorkloadState.RUNNING,
                created_by="alice",
            )
        )
        await tx.put_lease(
            LeaseRecord(
                id="lease-cpu",
                workload_id="wl-cpu",
                cell_id="cell-cpu",
                org_id="acme",
                project_id="research",
                state=LeaseState.ACTIVE,
                fencing_token="cell-cpu:1",
                resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
                expires_at=utcnow() + timedelta(hours=1),
                run_started_at=utcnow(),
            )
        )
        await tx.put_budget(
            BudgetRecord(org_id="acme", project_id="research", limit=1000.0, spent=0.0, revision=1, updated_by="root")
        )
    return repository


async def _durable_state(repository: InMemoryRepository) -> str:
    """A comparable rendering of everything the monitor could have disturbed."""

    async with repository.transaction() as tx:
        return json.dumps(
            {
                "budgets": [budget.model_dump(mode="json") for budget in await tx.list_budgets()],
                "cells": [cell.model_dump(mode="json") for cell in await tx.list_cells()],
                "events": [event.model_dump(mode="json") for event in await tx.list_events(limit=1_000)],
                "leases": [lease.model_dump(mode="json") for lease in await tx.list_leases()],
                "pools": [pool.model_dump(mode="json") for pool in await tx.list_capacity_pools()],
                "workloads": [item.model_dump(mode="json") for item in await tx.list_workloads()],
            },
            sort_keys=True,
        )


def _coverage(*, exercised: bool) -> dict[str, object]:
    participants = (_CPU_PARTICIPANT, _TPU_PARTICIPANT)
    executions: list[dict[str, object]] = []
    if exercised:
        for participant, identity in ((_CPU_PARTICIPANT, _CPU_IDENTITY), (_TPU_PARTICIPANT, _TPU_IDENTITY)):
            executions.append(
                {
                    "participant_id": participant.opaque_id(PROOF_ID),
                    **identity.as_dict(),
                    "execution_count": 1,
                }
            )
    return {
        "proof_id": PROOF_ID,
        "manifest_sha256": participant_manifest_sha256(PROOF_ID, participants),
        "executions": executions,
        "truncated": False,
    }


def _observation(*, artifact_bytes: int, exercised: bool, evidence: dict[str, tp.Any] | None) -> dict[str, object]:
    return {
        "checked": [f"I{number}" for number in range(1, 10)],
        "violations": [],
        "violations_truncated": False,
        "snapshot": {
            "observed_at": "2026-01-01T00:00:00+00:00",
            "budget_spend": [],
            "budget_spend_truncated": False,
            "artifact_bytes": artifact_bytes,
            "workload_count": 2,
            "cell_count": 2,
            "coverage": _coverage(exercised=exercised),
            "p8_evidence": evidence,
        },
    }


def _evaluate(evidence: dict[str, tp.Any] | None) -> dict[str, object]:
    """Score three observations carrying ``evidence`` through the real evaluator.

    Everything except the external sample is a qualifying run: an exercised
    CPU and TPU join, an artifact created and collected, no violations. The
    sample is therefore the only thing the verdict can turn on.
    """

    payloads = [
        _observation(artifact_bytes=0, exercised=False, evidence=evidence),
        _observation(artifact_bytes=100, exercised=True, evidence=evidence),
        _observation(artifact_bytes=0, exercised=True, evidence=evidence),
    ]
    state = run_soak(lambda: payloads.pop(0), ticks=3, sleep=_no_sleep, interval_s=0)
    inputs = SoakRunInputs.capture(
        candidate_sha="a" * 40,
        source_sha="b" * 40,
        config_sha256=CONFIG_SHA256,
        config_file_count=1,
        config_snapshot_materialized=True,
        proof_id=PROOF_ID,
        requested_duration_s=3,
        interval_s=1,
        expected_executions=[
            ExpectedExecution(participant=_CPU_PARTICIPANT, identity=_CPU_IDENTITY),
            ExpectedExecution(participant=_TPU_PARTICIPANT, identity=_TPU_IDENTITY),
        ],
        max_rss_growth_bytes=0,
        max_fd_growth=0,
        max_database_growth_bytes=0,
        max_artifact_net_growth_bytes=0,
    )
    started = datetime(2026, 1, 1, tzinfo=UTC)
    return build_soak_receipt(
        state,
        inputs,
        started_at=started,
        ended_at=started + timedelta(seconds=3),
        actual_duration_s=3,
        status="completed",
    )


def _no_sleep(_seconds: float) -> None:
    return None
