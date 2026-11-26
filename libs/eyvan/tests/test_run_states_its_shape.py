# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Asking for hardware by shape, width and zone, on the command path.

The decorator could ask for two slices; a command could not. It also could not
say which zone, so a job pinned to data in one region was placed wherever the
fleet felt like. Both were expressible only through the advanced submission
API, which is a JSON document, not something anyone types.
"""

from __future__ import annotations

import pytest

from eyvan.capacity.planning import POOL_ALTERNATES_LABEL
from eyvan.records import (
    WORKER_COUNT_LABEL,
    CapacityPoolRecord,
    Principal,
    ResourceVector,
    Role,
    RoleGrant,
)
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import ConflictError, NotFoundError

FLEET = "research-fleet"


def _pool(
    *, pool_id: str, zone: str, node_type: str = "v5p-workers", executor: str = "eyvan-native"
) -> CapacityPoolRecord:
    resources = ResourceVector(cpu=0, memory_gib=0, tpu=8)
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name=FLEET,
        node_type=node_type,
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="gcp-project-a",
        region=zone.rsplit("-", 1)[0],
        zone=zone,
        capacity=resources,
        allocatable=resources,
        accelerator_type="v5p-8",
        capacity_class="spot",
        executor=executor,
        agent_endpoint="http://agent:8090",
        topology={"fabric": "ici"},
        min_cells=0,
        initial_cells=0,
        max_cells=4,
        enabled=True,
    )


async def _service(*pools: CapacityPoolRecord) -> ControlPlaneService:
    service = ControlPlaneService(InMemoryRepository())
    fleet_admin = Principal(subject="system", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    for index, pool in enumerate(pools):
        await service.upsert_capacity_pool(fleet_admin, pool, idempotency_key=f"pool-{index}")
    return service


def _operator() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


async def _submit(service: ControlPlaneService, *, node_type: str = "v5p-workers", key: str = "run-1", **kwargs):
    workload, _replayed = await service.submit_pool_workload(
        _operator(),
        fleet_name=FLEET,
        node_type=node_type,
        org_id="acme",
        project_id="research",
        command="python train.py",
        idempotency_key=key,
        **kwargs,
    )
    return workload


# --------------------------------------------------------------------------- --pods


@pytest.mark.asyncio
async def test_pods_asks_for_several_slices_allocated_together() -> None:
    """A multislice command, which previously had to be a Python callable."""

    service = await _service(_pool(pool_id="pool-a", zone="us-east5-a"))

    workload = await _submit(service, pods=2)

    assert workload.spec.worker_count == 2
    assert workload.spec.required_labels[WORKER_COUNT_LABEL] == "2"


@pytest.mark.asyncio
async def test_one_pod_is_the_shape_a_plain_command_always_had() -> None:
    """Additive: a submission that says nothing about width is unchanged."""

    service = await _service(_pool(pool_id="pool-a", zone="us-east5-a"))

    workload = await _submit(service)

    assert workload.spec.worker_count == 1
    assert workload.spec.required_labels[WORKER_COUNT_LABEL] == "1"


@pytest.mark.asyncio
async def test_a_replay_cannot_return_a_different_width_than_was_asked_for() -> None:
    """The idempotency key names a request, and the width is part of it.

    Without this a retried submission that widened the request would be answered
    with the narrow job the first call created, and the caller would wait for
    slices nothing had asked for.
    """

    service = await _service(_pool(pool_id="pool-a", zone="us-east5-a"))
    await _submit(service, pods=1, key="same-key")

    with pytest.raises(Exception) as raised:
        await _submit(service, pods=4, key="same-key")

    assert "idempotency" in str(raised.value).lower() or "fingerprint" in str(raised.value).lower()


# --------------------------------------------------------------------------- --zone


@pytest.mark.asyncio
async def test_a_pinned_zone_selects_the_pool_in_it() -> None:
    """Two interchangeable pools, and the caller says which one."""

    service = await _service(
        _pool(pool_id="pool-a", zone="us-east5-a"),
        _pool(pool_id="pool-b", zone="us-east5-b"),
    )

    workload = await _submit(service, zone="us-east5-b")

    assert workload.spec.metadata["eyvan_pool"] == "pool-b"


@pytest.mark.asyncio
async def test_a_pinned_zone_is_not_quietly_widened_into_alternates() -> None:
    """The whole point of pinning. An alternate in another zone defeats it.

    Unpinned, these two pools are cross-zone capacity for one job. Pinned, the
    other zone must not be offered at all -- a job placed there would be running
    away from the data or the reservation it was pinned to.
    """

    service = await _service(
        _pool(pool_id="pool-a", zone="us-east5-a"),
        _pool(pool_id="pool-b", zone="us-east5-b"),
    )

    pinned = await _submit(service, zone="us-east5-a", key="pinned")
    free = await _submit(service, key="free")

    assert POOL_ALTERNATES_LABEL not in pinned.spec.metadata
    assert free.spec.metadata.get(POOL_ALTERNATES_LABEL) == "pool-b"


@pytest.mark.asyncio
async def test_a_zone_with_no_pool_of_that_shape_is_refused_by_name() -> None:
    """Refused before anything is provisioned, and told where it could run."""

    service = await _service(
        _pool(pool_id="pool-a", zone="us-east5-a"),
        _pool(pool_id="pool-b", zone="us-east5-b"),
    )

    with pytest.raises(NotFoundError) as raised:
        await _submit(service, zone="europe-west4-a")

    message = str(raised.value)
    assert "europe-west4-a" in message
    assert "us-east5-a" in message and "us-east5-b" in message


@pytest.mark.asyncio
async def test_a_pool_that_cannot_run_multislice_says_so_before_provisioning() -> None:
    """Several slices at once is a TPU multislice shape, not a general option.

    The record's own validator already refuses the spec, but it raises while the
    spec is being built -- so the caller was told their request was malformed
    when the truth is the pool they named cannot host that shape.
    """

    service = await _service(_pool(pool_id="cpu-pool", zone="us-east5-a", executor="native-local"))

    with pytest.raises(ConflictError) as raised:
        await _submit(service, pods=2)

    assert "eyvan-native" in str(raised.value)


@pytest.mark.asyncio
async def test_pinning_and_widening_compose() -> None:
    """`--tpu v5p-8 --pods 2 --zone us-east5-b` is one coherent request."""

    service = await _service(
        _pool(pool_id="pool-a", zone="us-east5-a"),
        _pool(pool_id="pool-b", zone="us-east5-b"),
    )

    workload = await _submit(service, node_type="v5p-8", pods=2, zone="us-east5-b")

    assert workload.spec.metadata["eyvan_pool"] == "pool-b"
    assert workload.spec.worker_count == 2
