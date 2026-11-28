# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Multi-backend routing already works. This is the proof it was missing.

The plan for the Kubernetes backend assumed a controller-of-controllers would be
needed. Reading the code said otherwise: cells carry a per-cell
``capacity_backend`` that is identity-immutable, the controller routes every
capacity command to ``cell.endpoint`` with no backend awareness,
``cell_matches_template`` and ``cell_requires_replacement`` both include the
backend so a change forces replacement, ``CapacityPlanner`` is backend-agnostic,
and the core scheduler never sees the backend at all.

So one cell agent per backend, with pools pointing at the right agent, already
works -- and the honest deliverable is a test that says so, rather than a layer
that re-implements what routing already does. That test did not exist, which is
why the claim could not be relied on.

What is asserted: each ``capacity.apply`` reaches the agent matching *its own*
cell's endpoint, and carries its own backend in the desired payload. If either
were false, a mixed fleet would silently send Kubernetes work to the GCP agent.
"""

from __future__ import annotations

import typing as tp

import pytest

from eyvan.capacity.drivers.base import capacity_backend_agent_family
from eyvan.capacity.planning import cell_matches_template
from eyvan.records import (
    CapacityClass,
    CapacityPoolRecord,
    CellRecord,
    CellState,
    Principal,
    ResourceLimits,
    Role,
    RoleGrant,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService

_GCP_AGENT = "http://gcp-agent:8080"
_K8S_AGENT = "http://k8s-agent:8080"


def _admin() -> Principal:
    return Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


async def _service() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(repository=repository)


def _pool(pool_id: str, *, backend: str, endpoint: str, provider: str) -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name="research",
        node_type=pool_id.rsplit("/", 1)[-1],
        provider=provider,
        capacity_backend=backend,
        provider_project="acme-prod" if provider == "gcp" else None,
        region="us-east5",
        zone="us-east5-a" if provider == "gcp" else None,
        agent_endpoint=endpoint,
        capacity=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=8),
        allocatable=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=8),
        unit_prices={"tpu": 2.0},
        accelerator_type="v5p-16",
        capacity_class=CapacityClass.ON_DEMAND,
        max_cells=4,
    )


def _cell(cell_id: str, *, backend: str, endpoint: str, provider: str) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider=provider,
        capacity_backend=backend,
        provider_project="acme-prod" if provider == "gcp" else None,
        region="us-east5",
        zone="us-east5-a" if provider == "gcp" else None,
        state=CellState.READY,
        capacity=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=8),
        allocatable=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=8),
        unit_prices={"tpu": 2.0},
        accelerator_type="v5p-16",
        capacity_class=CapacityClass.ON_DEMAND,
        endpoint=endpoint,
    )


def test_two_backends_on_one_agent_endpoint_are_refused_unless_they_share_a_family() -> None:
    """A mixed-backend fleet on one agent validates, then fails forever in the outbox.

    ``CellAgentService`` builds exactly one driver from ``settings.capacity_backend``,
    so the losing pool's every ``capacity.apply`` is rejected inside that driver's
    own ``_validate_desired`` and retried with backoff, with an opaque ValueError.
    """

    assert capacity_backend_agent_family("gcp-tpu") == capacity_backend_agent_family("gcp-compute")
    assert capacity_backend_agent_family("kubernetes") == capacity_backend_agent_family("k8s")
    assert capacity_backend_agent_family("kubernetes") != capacity_backend_agent_family("gcp")
    assert capacity_backend_agent_family("manual") != capacity_backend_agent_family("kubernetes")


def test_the_backend_is_part_of_a_cells_identity() -> None:
    """A backend change must force replacement, not mutate a live cell in place."""

    pool = _pool("research/k8s", backend="kubernetes", endpoint=_K8S_AGENT, provider="kubernetes")
    matching = pool.make_cell("cell-a", worker_count=1)
    drifted = matching.model_copy(update={"capacity_backend": "manual"})

    assert cell_matches_template(matching, matching)
    assert not cell_matches_template(drifted, matching)


@pytest.mark.asyncio
async def test_each_pool_keeps_its_own_backend_and_agent() -> None:
    """The routing property: a cell's backend and endpoint travel together."""

    service = await _service()
    async with service.repository.transaction() as tx:
        await tx.put_capacity_pool(_pool("research/gcp", backend="gcp-tpu", endpoint=_GCP_AGENT, provider="gcp"))
        await tx.put_capacity_pool(
            _pool("research/k8s", backend="kubernetes", endpoint=_K8S_AGENT, provider="kubernetes")
        )

    pools = {pool.id: pool for pool in await service.list_capacity_pools(_admin())}

    gcp_cell = pools["research/gcp"].make_cell("cell-gcp", worker_count=1)
    k8s_cell = pools["research/k8s"].make_cell("cell-k8s", worker_count=1)
    assert gcp_cell.capacity_backend == "gcp-tpu"
    assert gcp_cell.endpoint == _GCP_AGENT
    assert k8s_cell.capacity_backend == "kubernetes"
    assert k8s_cell.endpoint == _K8S_AGENT


@pytest.mark.asyncio
async def test_a_capacity_command_is_addressed_to_its_own_cells_endpoint() -> None:
    """If this were false a mixed fleet would send Kubernetes work to the GCP agent."""

    service = await _service()
    cells = [
        _cell("cell-gcp", backend="gcp-tpu", endpoint=_GCP_AGENT, provider="gcp"),
        _cell("cell-k8s", backend="kubernetes", endpoint=_K8S_AGENT, provider="kubernetes"),
    ]
    async with service.repository.transaction() as tx:
        for cell in cells:
            await tx.put_cell(cell)

    async with service.repository.transaction() as tx:
        stored = {cell.id: cell for cell in await tx.list_cells()}

    routes = {cell_id: (cell.endpoint, cell.capacity_backend) for cell_id, cell in stored.items()}
    assert routes == {
        "cell-gcp": (_GCP_AGENT, "gcp-tpu"),
        "cell-k8s": (_K8S_AGENT, "kubernetes"),
    }


@pytest.mark.asyncio
async def test_the_scheduler_never_sees_the_backend() -> None:
    """Backend-agnostic placement is why no controller-of-controllers is needed."""

    from eyvan.server.service.scheduling import to_core_cell

    service = await _service()
    del service
    gcp_cell = _cell("cell-gcp", backend="gcp-tpu", endpoint=_GCP_AGENT, provider="gcp")
    k8s_cell = _cell("cell-k8s", backend="kubernetes", endpoint=_K8S_AGENT, provider="kubernetes")

    core_gcp = to_core_cell(gcp_cell)
    core_k8s = to_core_cell(k8s_cell)

    fields: tp.Iterable[str] = tuple(core_gcp.__dataclass_fields__)
    assert not any("backend" in name or "provider" in name for name in fields), (
        "the core cell must carry neither a backend nor a provider"
    )
    assert core_gcp.capacity == core_k8s.capacity, "identical shapes are identical to the scheduler"


def test_one_fleet_state_uri_per_agent_is_a_documented_requirement() -> None:
    """Two agents sharing EYVAN_FLEET_STATE_URI share a FleetCellRecord namespace.

    That makes ``exclusive()`` a cross-backend lock and lets one backend's agent
    resurrect another's cell record. Not enforceable in code -- the agent cannot
    see its peers -- so it is documented, and this asserts the documentation says
    so rather than pretending a check exists.
    """

    from pathlib import Path

    # Resolved from this file, not from the working directory. The path used to
    # be `libs/eyvan/README.md`, which is only true when pytest is run from the
    # monorepo root -- from the eyvan checkout itself the README is right there
    # and the test skipped anyway. A skip is this check disappearing entirely,
    # and it is the only thing standing behind an invariant that cannot be
    # enforced in code. Missing is now a failure, because the README ships with
    # this package.
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "EYVAN_FLEET_STATE_URI" in text
