# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Several workloads can share one TPU cell's chips; nothing changes for anyone who does not ask.

Chip sharing is opt-in per pool (``chip_sharing``) and per workload
(``TpuConfig(chips=...)`` / ``chips_indexes=[...]``). Every new field is omitted
from wire documents while unused, so existing payloads, fingerprints and stored
records stay byte-identical.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from eyvan.records import (
    CapacityPoolRecord,
    CommandExecution,
    LeaseRecord,
    PlacementMode,
    ResourceGang,
    ResourceVector,
    TpuChipRequest,
    WorkloadDefinition,
)

_T0 = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)


def _pool(**extra) -> CapacityPoolRecord:
    fields: dict = {
        "id": "tpu-train-a-1a2b3c4d-v5p-8",
        "fleet_name": "tpu-train-a-1a2b3c4d",
        "node_type": "v5p-8",
        "provider": "gcp",
        "capacity_backend": "gcp-attach",
        "provider_project": "project-a",
        "region": "us-central1",
        "zone": "us-central1-a",
        "capacity": {"cpu": 208, "memory_gib": 448, "tpu": 4},
        "allocatable": {"cpu": 208, "memory_gib": 448, "tpu": 4},
        "accelerator_type": "v5p-8",
        "capacity_class": "on_demand",
        "executor": "eyvan-native",
        "agent_endpoint": "http://agent:8090",
        "labels": {"eyvan_resource_kind": "tpu"},
        "max_cells": 1,
    }
    return CapacityPoolRecord(**{**fields, **extra})


def _spec(*, tpu: int = 4, chips: TpuChipRequest | None = None, **extra) -> WorkloadDefinition:
    return WorkloadDefinition(
        name="job",
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(per_replica=ResourceVector(cpu=8, tpu=tpu)),
        tpu_chips=chips,
        **extra,
    )


def _lease(**extra) -> LeaseRecord:
    return LeaseRecord(
        id="lease-1",
        workload_id="wl-1",
        cell_id="cell-1",
        org_id="acme",
        project_id="research",
        resources=ResourceGang(per_replica=ResourceVector(tpu=1)),
        fencing_token="fence",
        expires_at=_T0 + timedelta(hours=1),
        created_at=_T0,
        updated_at=_T0,
        **extra,
    )


def test_unused_chip_fields_never_reach_a_wire_document() -> None:
    pool = _pool()
    cell = pool.make_cell("cell-1")
    for document in (
        _spec().model_dump(mode="json"),
        _lease().model_dump(mode="json"),
        pool.model_dump(mode="json"),
        cell.model_dump(mode="json"),
    ):
        assert "tpu_chips" not in document
        assert "chip_sharing" not in document


def test_opting_in_does_not_change_the_pool_template_fingerprint() -> None:
    # The fingerprint pins running workloads to a template; sharing chips is a
    # scheduling policy, not a different machine.
    assert _pool().template_fingerprint == _pool(chip_sharing=True).template_fingerprint


def test_a_sharing_pool_mints_sharing_cells_and_says_so_on_the_wire() -> None:
    pool = _pool(chip_sharing=True)
    cell = pool.make_cell("cell-1")
    assert cell.chip_sharing is True
    assert pool.model_dump(mode="json")["chip_sharing"] is True
    assert cell.model_dump(mode="json")["chip_sharing"] is True
    assert type(cell).model_validate(cell.model_dump(mode="json")) == cell


def test_chip_request_sorts_indexes_and_round_trips() -> None:
    request = TpuChipRequest(count=2, indexes=[2, 1])
    assert request.indexes == (1, 2)
    spec = _spec(tpu=2, chips=request)
    document = spec.model_dump(mode="json")
    assert document["tpu_chips"] == {"count": 2, "indexes": [1, 2]}
    assert WorkloadDefinition.model_validate(document) == spec

    lease = _lease(tpu_chips=(5, 6))
    assert lease.model_dump(mode="json")["tpu_chips"] == [5, 6]
    assert LeaseRecord.model_validate(lease.model_dump(mode="json")) == lease


@pytest.mark.parametrize(
    "fields",
    [
        {"count": 0},
        {"count": 2, "indexes": [1]},
        {"count": 2, "indexes": [1, 1]},
        {"count": 1, "indexes": [-1]},
        {"count": 1, "indexes": [True]},
        {"count": True},
    ],
)
def test_malformed_chip_requests_are_refused(fields: dict) -> None:
    with pytest.raises(ValidationError):
        TpuChipRequest(**fields)


def test_a_chip_workload_reserves_exactly_its_chips_on_one_host() -> None:
    with pytest.raises(ValidationError, match="exactly its chip count"):
        _spec(tpu=4, chips=TpuChipRequest(count=1))
    with pytest.raises(ValidationError, match="one host of one cell"):
        _spec(tpu=1, chips=TpuChipRequest(count=1), cell_group_count=2)
    with pytest.raises(ValidationError, match="atomic placement"):
        WorkloadDefinition(
            name="job",
            execution=CommandExecution(command="true"),
            gang=ResourceGang(per_replica=ResourceVector(tpu=1)),
            tpu_chips=TpuChipRequest(count=1),
            placement=PlacementMode.ELASTIC,
            worker_resource_scope="per_worker",
        )
    assert _spec(tpu=1, chips=TpuChipRequest(count=1)).tpu_chips == TpuChipRequest(count=1)


# --- pure chip picking -------------------------------------------------------


def test_pick_chips_prefers_aligned_blocks_and_honours_pins() -> None:
    from eyvan.domain.tpu_chips import allowed_chip_counts, chip_request_error, pick_chips

    assert allowed_chip_counts(4) == (1, 2, 4)
    everything = frozenset(range(8))
    assert pick_chips(everything, 1, None, total_chips=8, chips_per_host=4) == (0,)
    # Chip 0 taken: a pair goes to the aligned {2,3}, not the unaligned {1,2}.
    assert pick_chips(everything - {0}, 2, None, total_chips=8, chips_per_host=4) == (2, 3)
    # Host 0 too fragmented for a whole host: host 1 is used.
    assert pick_chips(everything - {0}, 4, None, total_chips=8, chips_per_host=4) == (4, 5, 6, 7)
    assert pick_chips(everything, 2, (1, 2), total_chips=8, chips_per_host=4) == (1, 2)
    assert pick_chips(everything - {2}, 2, (1, 2), total_chips=8, chips_per_host=4) is None
    assert chip_request_error(3, None, total_chips=8, chips_per_host=4) is not None
    assert "one host" in (chip_request_error(2, (3, 4), total_chips=8, chips_per_host=4) or "")
    assert "outside" in (chip_request_error(1, (8,), total_chips=8, chips_per_host=4) or "")


def test_pairs_follow_the_measured_v5p_ring_and_diagonals_are_refused() -> None:
    from eyvan.domain.tpu_chips import chip_request_error, pick_chips

    # Measured on a v5p host: {0,2} and {1,3} abort libtpu under every bounds value.
    for diagonal in ((0, 2), (1, 3), (4, 6)):
        assert "neighbours" in (chip_request_error(2, diagonal, total_chips=8, chips_per_host=4) or "")
    for ring in ((0, 1), (1, 2), (2, 3), (0, 3), (5, 6)):
        assert chip_request_error(2, ring, total_chips=8, chips_per_host=4) is None
    # Any-free never hands out a diagonal: with only {0,2} free on host 0 it moves on.
    assert pick_chips(frozenset({0, 2, 5, 6}), 2, None, total_chips=8, chips_per_host=4) == (5, 6)
    assert pick_chips(frozenset({0, 2}), 2, None, total_chips=4, chips_per_host=4) is None
    assert pick_chips(frozenset({1, 2}), 2, None, total_chips=4, chips_per_host=4) == (1, 2)
    # A host width nothing has measured offers single chips only.
    assert chip_request_error(2, None, total_chips=8, chips_per_host=8) is not None
    assert chip_request_error(1, None, total_chips=8, chips_per_host=8) is None


# --- scheduler ---------------------------------------------------------------


def _core_cell(*, sharing: bool = True, cell_id: str = "cell"):
    from eyvan.domain import CapacityCell, CapacityClass
    from eyvan.domain import ResourceVector as Vector

    return CapacityCell(
        cell_id,
        Vector(cpu=8, tpu=4),
        CapacityClass.ON_DEMAND,
        exclusive_workloads=True,
        chip_sharing=sharing,
        chips_per_host=4 if sharing else None,
    )


def _core_workload(
    workload_id: str,
    *,
    chips: int | None = None,
    indexes: tuple[int, ...] | None = None,
    queue_position: int = 0,
):
    from eyvan.domain import ResourceGang as Gang
    from eyvan.domain import ResourceVector as Vector
    from eyvan.domain import WorkloadIdentity, WorkloadSpec

    tpu = 4 if chips is None else chips
    return WorkloadSpec(
        workload_id,
        WorkloadIdentity("tenant", "project", "user"),
        (Gang("workers", 1, Vector(cpu=2 * tpu, tpu=tpu)),),
        queue_position=queue_position,
        tpu_chip_count=chips,
        tpu_chip_indexes=indexes,
    )


def _schedule(workloads, cells, leases=()):
    from eyvan.domain import Scheduler

    return Scheduler().schedule(workloads, cells, leases=leases)


def test_chip_workloads_share_one_cell_with_disjoint_chips() -> None:
    result = _schedule(
        [_core_workload("a", chips=1), _core_workload("b", chips=2), _core_workload("c", chips=1)],
        [_core_cell()],
    )
    granted = {decision.workload_id: decision.lease.tpu_chips for decision in result.placements}
    assert set(granted) == {"a", "b", "c"}, result.infeasibilities
    chips = [chip for held in granted.values() for chip in held]
    assert sorted(chips) == [0, 1, 2, 3]
    assert len(granted["b"]) == 2


def test_a_pinned_request_gets_its_chips_and_a_conflicting_pin_waits() -> None:
    result = _schedule(
        [
            _core_workload("pinned", chips=2, indexes=(1, 2)),
            _core_workload("clash", chips=1, indexes=(2,), queue_position=1),
        ],
        [_core_cell()],
    )
    assert [(p.workload_id, p.lease.tpu_chips) for p in result.placements] == [("pinned", (1, 2))]
    assert [i.workload_id for i in result.infeasibilities] == ["clash"]
    assert "[2]" in str(result.infeasibilities[0])
    assert not result.preemptions, "a chip share waits; it never preempts"


def test_whole_cell_and_chip_workloads_never_mix_on_one_cell() -> None:
    whole_first = _schedule([_core_workload("whole"), _core_workload("chip", chips=1, queue_position=1)], [_core_cell()])
    assert [p.workload_id for p in whole_first.placements] == ["whole"]
    assert [i.workload_id for i in whole_first.infeasibilities] == ["chip"]

    chip_lease = _schedule([_core_workload("chip", chips=1)], [_core_cell()]).placements[0].lease
    blocked = _schedule([_core_workload("whole")], [_core_cell()], leases=[chip_lease])
    assert not blocked.placements and blocked.infeasibilities


def test_a_cell_that_did_not_opt_in_keeps_one_workload_and_refuses_chip_requests() -> None:
    refused = _schedule([_core_workload("chip", chips=1)], [_core_cell(sharing=False)])
    assert not refused.placements
    assert "chip sharing" in str(refused.infeasibilities[0])

    parity = _schedule([_core_workload("one"), _core_workload("two")], [_core_cell(sharing=False)])
    assert [p.workload_id for p in parity.placements] == ["one"]
    assert parity.placements[0].lease.tpu_chips == ()


# --- dispatch ----------------------------------------------------------------


def test_chip_environment_confines_one_process_to_its_chips() -> None:
    from eyvan.worker.distributed_environment import chip_host_of, chip_submission_id, tpu_chip_environment

    pair = tpu_chip_environment([2, 1])
    assert pair == {
        "TPU_VISIBLE_CHIPS": "1,2",
        "TPU_CHIPS_PER_PROCESS_BOUNDS": "2,1,1",
        "TPU_PROCESS_BOUNDS": "1,1,1",
        "TPU_PROCESS_PORT": "8477",
        "TPU_PROCESS_ADDRESSES": "localhost:8477",
        "CLOUD_TPU_TASK_ID": "0",
        "ALLOW_MULTIPLE_LIBTPU_LOAD": "1",
    }
    # Neighbours on one host never share a libtpu port.
    ports = {tpu_chip_environment([chip])["TPU_PROCESS_PORT"] for chip in range(4)}
    assert len(ports) == 4
    # Each pair's orientation is the one the v5p host accepted; the other aborted libtpu.
    bounds = {
        pair: tpu_chip_environment(pair)["TPU_CHIPS_PER_PROCESS_BOUNDS"] for pair in ((0, 1), (2, 3), (1, 2), (0, 3))
    }
    assert bounds == {(0, 1): "1,2,1", (2, 3): "1,2,1", (1, 2): "2,1,1", (0, 3): "2,1,1"}
    assert tpu_chip_environment([0, 1, 2, 3])["TPU_CHIPS_PER_PROCESS_BOUNDS"] == "2,2,1"
    for impossible in ([0, 1, 2], [0, 2], [1, 3]):
        with pytest.raises(ValueError):
            tpu_chip_environment(impossible)

    routed = chip_submission_id("wl_1-a1-g2", 3)
    assert routed == "wl_1-a1-g2-tpuhost3" and chip_host_of(routed) == 3
    assert chip_host_of("wl_1-a1-g2") is None, "a whole-slice ID is never mistaken for a chip share"


def test_the_controller_routes_a_chip_lease_to_the_host_that_owns_its_chips() -> None:
    from eyvan.server.controller.workload_commands import PermanentDispatchRefusal, _chip_share_dispatch

    cell = _pool(
        accelerator_type="v5p-16",
        node_type="v5p-16",
        capacity={"cpu": 416, "memory_gib": 896, "tpu": 8},
        allocatable={"cpu": 416, "memory_gib": 896, "tpu": 8},
        id="pool-v5p-16",
    ).make_cell("cell-16")
    submission_id, environment = _chip_share_dispatch("wl_1-a1-g1", (5, 6), cell)
    assert submission_id == "wl_1-a1-g1-tpuhost1"
    assert environment["TPU_VISIBLE_CHIPS"] == "1,2", "host-local indexes, not global ones"

    unknown = cell.model_copy(update={"accelerator_type": "made-up-1"})
    with pytest.raises(PermanentDispatchRefusal, match="chips per host"):
        _chip_share_dispatch("wl_1-a1-g1", (0,), unknown)


def test_a_slice_sends_every_operation_of_a_chip_share_to_its_one_host_only() -> None:
    import json

    import httpx

    from eyvan.capacity.drivers.base import MutationContext
    from eyvan.worker.runtime._multihost_site import GangInitializationPolicy
    from eyvan.worker.runtime.native_slice import NativeSliceExecutor

    remote_id = "wl_1-a1-g1-tpuhost1"
    seen: list[tuple[str, str, str]] = []
    submitted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.method, request.url.path))
        path = request.url.path
        if request.method == "POST" and path == "/api/v1/workloads":
            submitted.append(json.loads(request.content)["spec"])
            return httpx.Response(200, json={"remote_id": remote_id})
        if request.method == "GET" and path == f"/api/v1/workloads/{remote_id}":
            return httpx.Response(200, json={"remote_id": remote_id, "status": "RUNNING", "message": "ok"})
        if request.method == "GET" and path.endswith("/logs"):
            return httpx.Response(200, text="chip share log\n")
        return httpx.Response(200, json={})

    endpoints = ("http://worker-a:8100", "http://worker-b:8100")
    clients = [httpx.Client(base_url=endpoint, transport=httpx.MockTransport(handler)) for endpoint in endpoints]
    environment = {"USER_FLAG": "kept", "TPU_VISIBLE_CHIPS": "1,2"}
    spec = {
        "entrypoint": "python share.py",
        "submission_id": remote_id,
        "environment": environment,
        "gang": {"replicas": 1, "per_replica": {"tpu": 2}},
        "metadata": {"eyvan_workload_id": "wl_1"},
    }
    context = MutationContext(idempotency_key="submit-1", generation=1, fencing_token="token-1")
    try:
        executor = NativeSliceExecutor(
            endpoints, secret="worker-secret", clients=clients, initialization=GangInitializationPolicy(fatal=False)
        )
        assert executor.submit(spec, context) == remote_id
        report = executor.status(remote_id)
        logs = executor.logs(remote_id)
        executor.stop(remote_id, context)
        executor.retire(remote_id, context)
    finally:
        for client in clients:
            client.close()

    assert {host for host, _, _ in seen} == {"worker-b"}, seen
    assert submitted[0]["environment"] == environment, "no slice-wide JAX world over a chip share"
    assert report["status"].value == "RUNNING" and [w["endpoint"] for w in report["workers"]] == [endpoints[1]]
    assert "chip share log" in logs and endpoints[0] not in logs


# --- SDK / CLI / fleet API ---------------------------------------------------


def test_tpu_config_accepts_a_count_or_exact_indexes_and_refuses_contradictions() -> None:
    from eyvan.sdk.client.jobs import TpuConfig

    whole = TpuConfig("v5p-128")
    assert whole.chips is None and whole.chip_request is None

    any_chip = TpuConfig("v5p-128", chips=1)
    assert any_chip.chip_request == TpuChipRequest(count=1)

    pinned = TpuConfig("v5p-128", chips_indexes=[2, 1])
    assert pinned.chips == 2 and pinned.chips_indexes == (1, 2)
    assert pinned.chip_request == TpuChipRequest(count=2, indexes=(1, 2))

    for kwargs in (
        {"chips": 0},
        {"chips": True},
        {"chips": 2, "chips_indexes": [1]},
        {"chips_indexes": []},
        {"chips_indexes": [1, 1]},
        {"chips_indexes": [-1]},
        {"chips": 1, "pods": 2},
        {"chips": 1, "atomic": True},
    ):
        with pytest.raises(ValueError):
            TpuConfig("v5p-128", **kwargs)


def _execution():
    from eyvan.records import CallableExecution

    return CallableExecution(artifact_uri="gs://bucket/callable.pkl", sha256="a" * 64, python="3.11")


def test_requests_without_chips_are_byte_identical_on_the_wire() -> None:
    from eyvan.records import CapacityClass
    from eyvan.sdk.client.requests import pool_callable_request, pool_request_json, pool_workload_request

    common = {"org_id": "acme", "project_id": "research", "name": None, "environment": None, "metadata": None}
    callable_plain = pool_request_json(
        pool_callable_request(
            **common,
            accelerator_type="v5p-8",
            pods=1,
            capacity_class=CapacityClass.SPOT,
            execution=_execution(),
            max_retries=3,
        )
    )
    assert "tpu_chips" not in callable_plain
    callable_chips = pool_request_json(
        pool_callable_request(
            **common,
            accelerator_type="v5p-8",
            pods=1,
            capacity_class=CapacityClass.SPOT,
            execution=_execution(),
            max_retries=3,
            tpu_chips=TpuChipRequest(count=2, indexes=(1, 2)),
        )
    )
    assert callable_chips["tpu_chips"] == {"count": 2, "indexes": [1, 2]}
    assert {k: v for k, v in callable_chips.items() if k != "tpu_chips"} == callable_plain

    from eyvan._internal.json_types import model_json

    command_plain = model_json(pool_workload_request(**common, command="true", max_retries=3))
    assert "tpu_chips" not in command_plain
    command_chips = model_json(
        pool_workload_request(**common, command="true", max_retries=3, tpu_chips=TpuChipRequest(count=1))
    )
    assert command_chips["tpu_chips"] == {"count": 1, "indexes": None}


def test_cli_chip_flags_build_the_same_request_as_the_sdk() -> None:
    from eyvan.sdk.commands.workloads import _chip_request

    assert _chip_request(None, ()) is None
    assert _chip_request(1, ()) == TpuChipRequest(count=1)
    assert _chip_request(None, (2, 1)) == TpuChipRequest(count=2, indexes=(1, 2))
    with pytest.raises(ValidationError):
        _chip_request(1, (1, 2))


def test_fleet_yaml_chip_sharing_reaches_the_pool_only_when_enabled() -> None:
    from eyvan.capacity.fleet_config import EyvanFleetConfig
    from tests.test_fleet_config import _mixed_fleet

    def compiled(**extra):
        raw = _mixed_fleet()
        raw["executor"] = "eyvan-native"
        raw["node_types"]["tpu"] = {"kind": "tpu", "accelerator_type": "v5p-8", "resources": {"tpu": 4}, **extra}
        for node in raw["node_types"].values():
            node["provider_options"] = {"bootstrap_egress": "private-google-access"}
        pools = EyvanFleetConfig.model_validate(raw).pool_payloads()
        return next(pool for pool in pools if pool.get("accelerator_type") == "v5p-8")

    assert "chip_sharing" not in compiled()
    sharing = compiled(chip_sharing=True)
    assert sharing["chip_sharing"] is True
    assert CapacityPoolRecord.model_validate(sharing).chip_sharing is True

    # The path `eyvan up` actually takes: the fleet-apply request must carry it to the record.
    from eyvan._internal.json_types import model_json
    from eyvan.server.api.schemas import FleetApplyRequest

    request = FleetApplyRequest.model_validate({"pools": [sharing]})
    assert request.to_records()[0].chip_sharing is True
    assert model_json(request)["pools"][0]["chip_sharing"] is True
    plain = FleetApplyRequest.model_validate({"pools": [compiled()]})
    assert "chip_sharing" not in model_json(plain)["pools"][0], "older planes forbid unknown fields"
    assert plain.to_records()[0].chip_sharing is False


# --- service + controller, end to end on an in-memory plane -----------------


def _admin():
    from eyvan.records import Principal, Role, RoleGrant

    return Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _operator():
    from eyvan.records import Principal, Role, RoleGrant

    return Principal(
        subject="operator", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),)
    )


async def _plane(pool: CapacityPoolRecord):
    from eyvan.server.persistence import InMemoryRepository
    from eyvan.server.service import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    await service.apply_fleet(_admin(), pool.fleet_name, (pool,), idempotency_key="fleet-v1")
    return repository, service


async def _submit(service, key: str, chips: TpuChipRequest | None):
    workload, _ = await service.submit_pool_workload(
        _operator(),
        fleet_name="tpu-train-a-1a2b3c4d",
        node_type="v5p-8",
        org_id="acme",
        project_id="research",
        command="python share.py",
        tpu_chips=chips,
        idempotency_key=key,
    )
    return workload


@pytest.mark.asyncio
async def test_a_pool_that_did_not_opt_in_refuses_a_chip_request_naming_the_setting() -> None:
    from eyvan.server.service.errors import ConflictError

    _repository, service = await _plane(_pool())
    with pytest.raises(ConflictError, match="chip_sharing: true"):
        await _submit(service, "chip-1", TpuChipRequest(count=1))
    whole = await _submit(service, "whole-1", None)
    assert whole.spec.tpu_chips is None and whole.spec.gang.per_replica.tpu == 4
    assert "tpu_chips" not in whole.spec.model_dump(mode="json")


@pytest.mark.asyncio
async def test_a_sharing_pool_validates_shapes_and_reserves_a_proportional_share() -> None:
    from eyvan.server.service.errors import InputError

    _repository, service = await _plane(_pool(chip_sharing=True))
    for bad in (TpuChipRequest(count=3), TpuChipRequest(count=1, indexes=(4,))):
        with pytest.raises(InputError):
            await _submit(service, f"bad-{bad.count}-{bad.indexes}", bad)
    share = await _submit(service, "pair", TpuChipRequest(count=2, indexes=(1, 2)))
    assert share.spec.tpu_chips == TpuChipRequest(count=2, indexes=(1, 2))
    assert share.spec.gang.per_replica.tpu == 2
    assert share.spec.gang.per_replica.cpu == 104 and share.spec.gang.per_replica.memory_gib == 224
    assert "eyvan_pool_alternates" not in share.spec.required_labels


@pytest.mark.asyncio
async def test_turning_chip_sharing_on_reaches_live_cells_without_replacing_them() -> None:
    from eyvan.records import CellState

    pool = _pool()
    repository, service = await _plane(pool)
    async with repository.transaction() as tx:
        stored = (await tx.list_capacity_pools(fleet_name=pool.fleet_name))[0]
        await tx.put_cell(stored.make_cell(f"{pool.id}-0001").model_copy(update={"state": CellState.READY}))

    await service.apply_fleet(_admin(), pool.fleet_name, (_pool(chip_sharing=True),), idempotency_key="fleet-v2")
    async with repository.transaction() as tx:
        after = (await tx.list_capacity_pools(fleet_name=pool.fleet_name))[0]
        cell = (await tx.list_cells())[0]
    assert after.chip_sharing is True
    assert after.generation == stored.generation, "a policy flag must not fence and replace the cell"
    assert cell.chip_sharing is True and cell.state is CellState.READY and cell.generation == 1

    await service.apply_fleet(_admin(), pool.fleet_name, (_pool(),), idempotency_key="fleet-v3")
    async with repository.transaction() as tx:
        assert (await tx.list_cells())[0].chip_sharing is False


@pytest.mark.asyncio
async def test_the_controller_leases_disjoint_chips_to_concurrent_shares_of_one_cell() -> None:
    from eyvan.config import EyvanSettings
    from eyvan.records import CellState, LeaseState, WorkloadState
    from eyvan.server.controller import EyvanController

    pool = _pool(chip_sharing=True)
    repository, service = await _plane(pool)
    async with repository.transaction() as tx:
        await tx.put_cell(pool.make_cell(f"{pool.id}-0001").model_copy(update={"state": CellState.READY}))
    shares = [
        await _submit(service, "any-1", TpuChipRequest(count=1)),
        await _submit(service, "pinned-pair", TpuChipRequest(count=2, indexes=(1, 2))),
        await _submit(service, "clash", TpuChipRequest(count=1, indexes=(2,))),
    ]
    controller = EyvanController(repository, service, settings=EyvanSettings())
    await controller.reconcile_once()

    async with repository.transaction() as tx:
        leases = [lease for lease in await tx.list_leases() if lease.state is not LeaseState.RELEASED]
        workloads = {workload.id: workload for workload in await tx.list_workloads()}
    held = {lease.workload_id: lease.tpu_chips for lease in leases}
    granted = [chip for chips in held.values() for chip in chips or ()]
    assert len(granted) == len(set(granted)), f"chips granted twice: {held}"
    assert all(chips for chips in held.values()), "every share lease names its chips"
    assert len(held) == 2, held
    pending = [shares[i].id for i in range(3) if workloads[shares[i].id].state is WorkloadState.PENDING]
    assert len(pending) == 1, "exactly one of the two requests wanting chip 2 waits"
    if shares[1].id in held:
        assert held[shares[1].id] == (1, 2)
