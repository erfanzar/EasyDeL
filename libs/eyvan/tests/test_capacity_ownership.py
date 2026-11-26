# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""An organization administering the capacity it pays for, and nothing else.

Every cell and pool mutation was ``MANAGE_FLEET``-only, and ``authz`` returned
``permission != MANAGE_FLEET`` for ``ORG_ADMIN`` -- so multi-tenant capacity
self-service was not merely unbuilt, it was refused. Resizing a pool you paid
for meant asking someone who could also delete every other tenant's cells and
change the provider configuration underneath them.

``MANAGE_OWN_CAPACITY`` and ``CapacityPoolRecord.owner_org_id`` make the middle
ground expressible. The tests below are mostly about its edges, because the
middle is easy and the edges are where a capacity permission becomes a
cross-tenant one: taking over a pool by declaring yourself its owner, handing
one away, or reaching shared capacity nobody assigned.
"""

from __future__ import annotations

import pytest

from eyvan.records import (
    CapacityPoolRecord,
    Principal,
    ResourceVector,
    Role,
    RoleGrant,
)
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import AuthorizationError
from eyvan.server.service.scheduling import CoreSchedulingService

FLEET_ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _org_admin(org: str) -> Principal:
    return Principal(subject=f"admin@{org}", grants=(RoleGrant(role=Role.ORG_ADMIN, org_id=org),))


def _service() -> ControlPlaneService:
    return ControlPlaneService(InMemoryRepository(), scheduler=CoreSchedulingService())


def _pool(
    pool_id: str = "v5p-pool",
    *,
    owner: str | None = None,
    max_cells: int = 1,
    initial_cells: int | None = None,
) -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name="research-fleet",
        node_type="v5p-workers",
        owner_org_id=owner,
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        capacity=ResourceVector(tpu=8),
        allocatable=ResourceVector(tpu=8),
        accelerator_type="v5p-8",
        capacity_class="spot",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        initial_cells=initial_cells,
        min_cells=0,
        max_cells=max_cells,
    )


async def _owned(service: ControlPlaneService, org: str, pool_id: str = "v5p-pool") -> CapacityPoolRecord:
    stored, _ = await service.upsert_capacity_pool(
        FLEET_ADMIN, _pool(pool_id, owner=org), idempotency_key=f"seed-{pool_id}"
    )
    return stored


# --------------------------------------------------------------------------- the point


async def test_an_org_admin_resizes_a_pool_its_organization_owns() -> None:
    """The capability that did not exist: capacity self-service inside a tenant."""

    service = _service()
    await _owned(service, "acme")

    stored, _ = await service.upsert_capacity_pool(
        _org_admin("acme"), _pool(owner="acme", max_cells=8), idempotency_key="resize"
    )

    assert stored.max_cells == 8


async def test_an_org_admin_disables_a_pool_its_organization_owns() -> None:
    """Scaling to zero is as much a part of owning capacity as scaling up."""

    service = _service()
    await _owned(service, "acme")

    stored, _ = await service.disable_capacity_pool(_org_admin("acme"), "v5p-pool", idempotency_key="disable")

    assert (stored.enabled, stored.max_cells) == (False, 0)


async def test_disabling_a_pool_declared_with_a_floor_keeps_its_record_revalidatable() -> None:
    """``min <= initial <= max`` has to survive the disable.

    ``apply_fleet`` and ``disable_fleet`` zero ``initial_cells`` beside the floor
    and the cap for exactly this reason; the single-pool path zeroed only
    min/max, so a pool declared with ``initial_cells=1`` stored initial 1 > max 0
    and any later re-validation of the stored record refused it.
    """

    service = _service()
    await service.upsert_capacity_pool(
        FLEET_ADMIN, _pool("floored-pool", initial_cells=1), idempotency_key="seed-floored"
    )

    stored, _ = await service.disable_capacity_pool(FLEET_ADMIN, "floored-pool", idempotency_key="disable-floored")

    assert (stored.min_cells, stored.max_cells, stored.initial_cells) == (0, 0, 0)
    # The failure this pins was a 422 naming a field the operator never set.
    assert CapacityPoolRecord.model_validate(stored.model_dump(mode="json")) == stored


# --------------------------------------------------------------------------- the edges


async def test_a_pool_cannot_be_taken_over_by_declaring_yourself_its_owner() -> None:
    """Ownership is read from storage, not from the request that wants to change it.

    Trusting the declared owner would make ``owner_org_id`` a self-service
    field: anyone with capacity rights in *any* organization could claim any
    pool by naming their own.
    """

    service = _service()
    await _owned(service, "rival")

    with pytest.raises(AuthorizationError, match="owned by another organization"):
        await service.upsert_capacity_pool(
            _org_admin("acme"), _pool(owner="acme", max_cells=99), idempotency_key="takeover"
        )


async def test_a_pool_cannot_be_handed_to_another_organization() -> None:
    """The mirror of takeover, and it loses capacity rather than stealing it."""

    service = _service()
    await _owned(service, "acme")

    with pytest.raises(AuthorizationError, match="would be owned by another organization"):
        await service.upsert_capacity_pool(_org_admin("acme"), _pool(owner="rival"), idempotency_key="donate")


async def test_a_pool_cannot_be_pushed_into_shared_ownership() -> None:
    """Disowning is not a way out either.

    An org admin who could set ``owner_org_id`` to ``None`` would move capacity
    into the fleet-administered pool nobody asked the operator about -- and
    would then be unable to reach it themselves, which is a support ticket
    rather than a permission.
    """

    service = _service()
    await _owned(service, "acme")

    with pytest.raises(AuthorizationError, match="shared ownership"):
        await service.upsert_capacity_pool(_org_admin("acme"), _pool(owner=None), idempotency_key="disown")


async def test_shared_capacity_stays_fleet_administered() -> None:
    """Every pool that existed before ownership did keeps the authority it had.

    ``owner_org_id`` defaults to ``None``, so this is not an edge case -- it is
    every pool in every deployment upgrading to this build.
    """

    service = _service()
    await service.upsert_capacity_pool(FLEET_ADMIN, _pool(owner=None), idempotency_key="seed")

    with pytest.raises(AuthorizationError, match="unowned capacity"):
        await service.upsert_capacity_pool(_org_admin("acme"), _pool(owner=None, max_cells=8), idempotency_key="grab")

    with pytest.raises(AuthorizationError):
        await service.disable_capacity_pool(_org_admin("acme"), "v5p-pool", idempotency_key="kill")


async def test_an_org_admin_cannot_disable_another_organizations_pool() -> None:
    """The destructive direction, which needs no request body to get wrong."""

    service = _service()
    await _owned(service, "rival")

    with pytest.raises(AuthorizationError, match="owned by another organization"):
        await service.disable_capacity_pool(_org_admin("acme"), "v5p-pool", idempotency_key="kill")


async def test_creating_a_new_owned_pool_still_needs_fleet_administration() -> None:
    """Otherwise an org admin allocates itself capacity nobody approved.

    Sizing a pool the fleet operator gave you is self-service; conjuring a new
    one that will go request accelerators from a provider is not.
    """

    service = _service()

    with pytest.raises(AuthorizationError):
        await service.upsert_capacity_pool(_org_admin("acme"), _pool(owner="acme"), idempotency_key="create")


async def test_reassigning_an_owner_requires_fleet_administration() -> None:
    """Handing a pool between tenants changes who may administer it from then on.

    Deliberate rather than a side effect of an update that happened to also
    change ``max_cells``.
    """

    service = _service()
    await _owned(service, "acme")

    reassigned, _ = await service.upsert_capacity_pool(FLEET_ADMIN, _pool(owner="rival"), idempotency_key="reassign")

    assert reassigned.owner_org_id == "rival"
    with pytest.raises(AuthorizationError):
        await service.upsert_capacity_pool(_org_admin("acme"), _pool(owner="acme"), idempotency_key="take-back")


async def test_an_admin_of_two_organizations_still_cannot_move_capacity_between_them() -> None:
    """The case that makes the reassignment gate load-bearing.

    Holding ``ORG_ADMIN`` in both organizations satisfies the ownership check on
    both sides -- existing and declared -- so without a separate gate a single
    person administering two tenants could move a pool between them. That is a
    fleet allocation decision: it changes which tenant's budget the capacity
    lands on, and it is not visible to the fleet operator who assigned it.
    """

    service = _service()
    await _owned(service, "acme")
    both = Principal(
        subject="admin@both",
        grants=(
            RoleGrant(role=Role.ORG_ADMIN, org_id="acme"),
            RoleGrant(role=Role.ORG_ADMIN, org_id="rival"),
        ),
    )

    # Sizing either organization's own pool is still theirs to do.
    resized, _ = await service.upsert_capacity_pool(both, _pool(owner="acme", max_cells=4), idempotency_key="resize")
    assert resized.max_cells == 4

    with pytest.raises(AuthorizationError):
        await service.upsert_capacity_pool(both, _pool(owner="rival"), idempotency_key="move")


async def test_a_declarative_fleet_apply_stays_fleet_administered() -> None:
    """It disables every pool absent from the request, including ones you do not own.

    An organization admin running it would take down the rest of the fleet, so
    the ownership-scoped surface is per-pool upsert and disable rather than this.
    """

    service = _service()
    await _owned(service, "acme")

    with pytest.raises(AuthorizationError):
        await service.apply_fleet(_org_admin("acme"), "research-fleet", [_pool(owner="acme")], idempotency_key="apply")


async def test_capacity_administration_is_not_a_project_level_permission() -> None:
    """A pool serves every project in its organization.

    Letting one project's admin resize it would let them take capacity from a
    sibling project, which is why the permission sits on the organization role.
    """

    service = _service()
    await _owned(service, "acme")
    project_admin = Principal(
        subject="p",
        grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="research"),),
    )

    with pytest.raises(AuthorizationError):
        await service.upsert_capacity_pool(project_admin, _pool(owner="acme", max_cells=8), idempotency_key="resize")
