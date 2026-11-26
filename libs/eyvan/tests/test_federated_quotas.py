# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Quota grants: the cross-plane hole budgets closed, in the other currency.

``QuotaRecord.limits`` is one durable vector in one database, so a tenant with
projects on three planes could run three times its accelerator ceiling with every
plane locally compliant and every dashboard green. Money and chips are separate
limits and a fleet can hold one while breaching the other -- a tenant on a small
budget can still exhaust a TPU pool everyone else is queued behind, because what
is scarce is chips.

Two properties carry the whole design, and both are asserted against the
*scheduler's* view rather than against the records, because the failure this
package keeps producing is a mechanism that is recorded and consulted by nothing:

* headroom holds in **every** dimension at once -- a slice with TPU to spare and
  no CPU left is refused; and
* a grantee's effective ceiling is the **tighter** of grant and local quota, so a
  grant can never raise a limit a local operator set.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.records import Principal, QuotaRecord, ResourceLimits, Role, RoleGrant, utcnow
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import InputError, NotFoundError
from eyvan.server.service.federation import (
    GrantState,
    PlanePeerRecord,
    QuotaGrantRecord,
    exceeds_quota_headroom,
    outstanding_quota,
    quota_headroom,
)
from eyvan.server.service.scheduling import federated_quotas

_HOME = "us-east5"
_PEER = "europe-west4"
NOW = utcnow()


def _admin() -> Principal:
    return Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


async def _service(*, plane_id: str = _HOME) -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(repository=repository, plane_id=plane_id)


def _quota(**overrides: object) -> QuotaRecord:
    defaults: dict[str, object] = {
        "org_id": "acme",
        "project_id": "research",
        "limits": ResourceLimits(cpu=64, tpu=16),
        "updated_by": "root",
    }
    return QuotaRecord(**{**defaults, **overrides})


def _grant(**overrides: object) -> QuotaGrantRecord:
    defaults: dict[str, object] = {
        "id": "qgr-1",
        "org_id": "acme",
        "project_id": "research",
        "home_plane_id": _HOME,
        "grantee_plane_id": _PEER,
        "limits": ResourceLimits(cpu=16, tpu=8),
        "issued_at": NOW,
        "expires_at": NOW + timedelta(days=7),
    }
    return QuotaGrantRecord(**{**defaults, **overrides})


async def _seed(service: ControlPlaneService, *, limits: ResourceLimits | None = None) -> None:
    async with service.repository.transaction() as tx:
        await tx.put_quota(_quota(limits=limits or ResourceLimits(cpu=64, tpu=16)))
        await tx.put_plane_peer(PlanePeerRecord(plane_id=_PEER, endpoint="https://eu.example.test"))


# --------------------------------------------------------------------------- the vector math


def test_headroom_is_per_dimension() -> None:
    """A scalar answer would hide the dimension that is actually exhausted."""

    headroom = quota_headroom(ResourceLimits(cpu=64, tpu=16), [_grant()], now=NOW)

    assert headroom == {"cpu": 48.0, "memory_gib": 0.0, "gpu": 0.0, "tpu": 8.0}


def test_an_expired_slice_retains_its_headroom() -> None:
    """Expiry is not proof that the peer stopped its existing work."""

    lapsed = _grant(expires_at=NOW - timedelta(seconds=1), issued_at=NOW - timedelta(days=1))

    assert not lapsed.active_at(NOW)
    assert quota_headroom(ResourceLimits(tpu=16), [lapsed], now=NOW)["tpu"] == 8.0
    assert outstanding_quota([lapsed], now=NOW)["tpu"] == 8.0


def test_replacing_a_grant_preserves_its_retained_capacity() -> None:
    """Replacement can use only headroom not already delegated to a peer."""

    headroom = quota_headroom(ResourceLimits(tpu=16), [_grant()], now=NOW, replacing="qgr-1")

    assert headroom["tpu"] == 8.0


def test_every_overrun_dimension_is_named() -> None:
    """One dimension at a time is four round trips and one abandoned operator."""

    overrun = exceeds_quota_headroom(ResourceLimits(cpu=100, tpu=100), {"cpu": 48.0, "tpu": 8.0})

    assert overrun == ("cpu", "tpu")


def test_a_dimension_the_quota_does_not_constrain_has_no_headroom() -> None:
    """Absent is zero, not unlimited.

    A grant that silently authorized every resource its tenant's quota forgot to
    mention would be the opposite of a ceiling.
    """

    assert exceeds_quota_headroom(ResourceLimits(gpu=1), {"cpu": 48.0}) == ("gpu",)


def test_outstanding_quota_sums_live_and_retired_slices() -> None:
    second = _grant(id="qgr-2", grantee_plane_id="ap-1", limits=ResourceLimits(tpu=4))
    revoked = _grant(id="qgr-3", grantee_plane_id="ap-2", state=GrantState.REVOKED)

    assert outstanding_quota([_grant(), second, revoked], now=NOW)["tpu"] == 20.0


# --------------------------------------------------------------------------- the record


def test_a_grant_must_carry_something() -> None:
    """An all-zero slice is a typo that reads as a grant in every listing."""

    with pytest.raises(ValueError, match="positive quantity"):
        _grant(limits=ResourceLimits())


def test_a_plane_does_not_grant_to_itself() -> None:
    with pytest.raises(ValueError, match="does not grant quota to itself"):
        _grant(grantee_plane_id=_HOME)


def test_a_grant_must_expire() -> None:
    with pytest.raises(ValueError, match="must expire after it is issued"):
        _grant(expires_at=NOW - timedelta(seconds=1))


# --------------------------------------------------------------------------- issuance


@pytest.mark.asyncio
async def test_the_home_plane_refuses_to_over_issue() -> None:
    """The entire cross-plane guarantee: slices are bounded when they are cut."""

    service = await _service()
    await _seed(service)

    with pytest.raises(InputError, match=r"tpu 32 > 16(?:\.0)?$"):
        await service.issue_quota_grant(
            _admin(),
            org_id="acme",
            project_id="research",
            grantee_plane_id=_PEER,
            limits=ResourceLimits(tpu=32),
            expires_at=NOW + timedelta(days=1),
            idempotency_key="k1",
        )


@pytest.mark.asyncio
async def test_a_second_grant_is_measured_against_what_the_first_took() -> None:
    """Two peers each inside the limit but jointly over it is the whole failure."""

    service = await _service()
    await _seed(service)
    async with service.repository.transaction() as tx:
        await tx.put_plane_peer(PlanePeerRecord(plane_id="ap-1", endpoint="https://ap.example.test"))
    await service.issue_quota_grant(
        _admin(),
        org_id="acme",
        project_id="research",
        grantee_plane_id=_PEER,
        limits=ResourceLimits(tpu=12),
        expires_at=NOW + timedelta(days=1),
        idempotency_key="k1",
    )

    with pytest.raises(InputError, match="tpu"):
        await service.issue_quota_grant(
            _admin(),
            org_id="acme",
            project_id="research",
            grantee_plane_id="ap-1",
            limits=ResourceLimits(tpu=8),
            expires_at=NOW + timedelta(days=1),
            idempotency_key="k2",
        )


@pytest.mark.asyncio
async def test_re_issuing_replaces_admission_but_retains_both_capacity_slices() -> None:
    """Retiring the old grant does not prove its capacity has been released."""

    service = await _service()
    await _seed(service)
    first = await service.issue_quota_grant(
        _admin(),
        org_id="acme",
        project_id="research",
        grantee_plane_id=_PEER,
        limits=ResourceLimits(tpu=8),
        expires_at=NOW + timedelta(days=1),
        idempotency_key="k1",
    )
    with pytest.raises(InputError, match="tpu"):
        await service.issue_quota_grant(
            _admin(),
            org_id="acme",
            project_id="research",
            grantee_plane_id=_PEER,
            limits=ResourceLimits(tpu=12),
            expires_at=NOW + timedelta(days=1),
            idempotency_key="k2",
        )
    assert [grant.id for grant in await service.list_quota_grants(_admin())] == [first.id]
    second = await service.issue_quota_grant(
        _admin(),
        org_id="acme",
        project_id="research",
        grantee_plane_id=_PEER,
        limits=ResourceLimits(tpu=8),
        expires_at=NOW + timedelta(days=1),
        idempotency_key="k2",
    )

    live = await service.list_quota_grants(_admin())
    retained = await service.list_quota_grants(_admin(), include_inactive=True)

    assert [grant.id for grant in live] == [second.id]
    assert second.id != first.id
    assert next(grant for grant in retained if grant.id == first.id).state is GrantState.REVOKED
    assert outstanding_quota(retained, now=NOW)["tpu"] == 16.0
    assert quota_headroom(ResourceLimits(tpu=16), retained, now=NOW)["tpu"] == 0.0


@pytest.mark.asyncio
async def test_a_plane_with_no_identity_cannot_issue() -> None:
    service = await _service(plane_id="")
    await _seed(service)

    with pytest.raises(InputError, match="no plane_id"):
        await service.issue_quota_grant(
            _admin(),
            org_id="acme",
            project_id="research",
            grantee_plane_id=_PEER,
            limits=ResourceLimits(tpu=1),
            expires_at=NOW + timedelta(days=1),
            idempotency_key="k1",
        )


@pytest.mark.asyncio
async def test_an_unregistered_peer_is_refused() -> None:
    """Peers are written down, never discovered."""

    service = await _service()
    async with service.repository.transaction() as tx:
        await tx.put_quota(_quota())

    with pytest.raises(NotFoundError, match="not registered"):
        await service.issue_quota_grant(
            _admin(),
            org_id="acme",
            project_id="research",
            grantee_plane_id="stranger",
            limits=ResourceLimits(tpu=1),
            expires_at=NOW + timedelta(days=1),
            idempotency_key="k1",
        )


@pytest.mark.asyncio
async def test_a_tenant_with_no_quota_cannot_have_one_sliced() -> None:
    """An unlimited tenant has no ceiling to cut a slice out of."""

    service = await _service()
    async with service.repository.transaction() as tx:
        await tx.put_plane_peer(PlanePeerRecord(plane_id=_PEER, endpoint="https://eu.example.test"))

    with pytest.raises(NotFoundError, match="quota for acme/research"):
        await service.issue_quota_grant(
            _admin(),
            org_id="acme",
            project_id="research",
            grantee_plane_id=_PEER,
            limits=ResourceLimits(tpu=1),
            expires_at=NOW + timedelta(days=1),
            idempotency_key="k1",
        )


@pytest.mark.asyncio
async def test_the_expiry_sweep_covers_quota_slices_too() -> None:
    """One sweep, not two.

    A second leader phase is one more thing to forget to register, which is
    exactly how the budget sweep spent its first life.
    """

    service = await _service()
    async with service.repository.transaction() as tx:
        await tx.put_quota_grant(_grant(issued_at=NOW - timedelta(days=2), expires_at=NOW - timedelta(days=1)))

    assert await service.expire_budget_grants_once() == 1
    async with service.repository.transaction() as tx:
        stored = await tx.get_quota_grant("qgr-1")
    assert stored is not None and stored.state is GrantState.EXPIRED


# --------------------------------------------------------------------------- admission


def test_a_plane_holding_no_grants_is_unchanged() -> None:
    """The home plane, and every non-federated deployment."""

    quotas = federated_quotas([_quota()], [], plane_id=_PEER, now=NOW)

    assert quotas[0].limit.as_dict()["tpu"] == 16


def test_a_grantee_is_bounded_by_the_tighter_of_the_two() -> None:
    """A grant may not raise a ceiling a local operator set, and vice versa."""

    quotas = federated_quotas([_quota()], [_grant()], plane_id=_PEER, now=NOW)
    limit = quotas[0].limit.as_dict()

    assert limit["tpu"] == 8  # the grant is tighter
    assert limit["cpu"] == 16  # the grant is tighter here too


def test_a_local_quota_tighter_than_the_grant_still_wins() -> None:
    """The direction that proves this is a minimum rather than an assignment."""

    quotas = federated_quotas([_quota(limits=ResourceLimits(tpu=2))], [_grant()], plane_id=_PEER, now=NOW)

    assert quotas[0].limit.as_dict()["tpu"] == 2


def test_a_grantee_with_no_local_quota_still_gets_a_ceiling() -> None:
    """Otherwise the plane with the least configuration has no limit at all."""

    quotas = federated_quotas([], [_grant()], plane_id=_PEER, now=NOW)

    assert len(quotas) == 1
    assert quotas[0].limit.as_dict()["tpu"] == 8


def test_a_grant_addressed_elsewhere_does_not_bind_here() -> None:
    """A slice cut for Europe must not constrain Asia."""

    quotas = federated_quotas([_quota()], [_grant()], plane_id="ap-1", now=NOW)

    assert quotas[0].limit.as_dict()["tpu"] == 16


def test_an_expired_grant_stops_licensing_work() -> None:
    """An unreachable home plane has to fail closed."""

    lapsed = _grant(issued_at=NOW - timedelta(days=2), expires_at=NOW - timedelta(days=1))

    quotas = federated_quotas([], [lapsed], plane_id=_PEER, now=NOW)

    assert len(quotas) == 1
    assert quotas[0].limit.as_dict() == {"cpu": 0.0, "memory_gib": 0.0, "gpu": 0.0, "tpu": 0.0}


def test_principal_scoped_quotas_are_left_alone() -> None:
    """A cross-plane grant has nothing to say about how a tenant splits its own slice.

    Federating them would let one plane's grant silently rewrite a project's
    internal fair-share between its own people.
    """

    # Deliberately *above* the grant's 8. A personal quota below it would be
    # min()-ed to the same number either way, and the test would pass whether or
    # not the exemption existed -- which is how the first version of this passed
    # the very mutation it was written to catch.
    personal = _quota(principal_id="alice", limits=ResourceLimits(tpu=12))

    quotas = federated_quotas([personal], [_grant()], plane_id=_PEER, now=NOW)

    assert quotas[0].limit.as_dict()["tpu"] == 12


async def _plane_with(grant: QuotaGrantRecord | None, *, tpu: float) -> object:
    """A grantee plane holding one cell, one quota, and a workload wanting ``tpu``."""

    from test_controller import principal

    from eyvan.records import CellRecord, CellState, CommandExecution, ResourceGang, ResourceVector, WorkloadDefinition
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService(), plane_id=_PEER)
    async with repository.transaction() as tx:
        await tx.put_quota(_quota(constrained_resources=("tpu",)))
        if grant is not None:
            await tx.put_quota_grant(grant)
        await tx.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=64, memory_gib=512, tpu=16),
                allocatable=ResourceVector(cpu=64, memory_gib=512, tpu=16),
                endpoint="http://agent:8090",
            )
        )
    await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1, tpu=tpu)),
        ),
        idempotency_key="train",
    )
    return await service.schedule_once(principal(Role.FLEET_ADMIN))


@pytest.mark.asyncio
async def test_admission_refuses_work_beyond_the_granted_slice() -> None:
    """The seam, and the only assertion here that could have caught the real bug.

    ``federated_quotas`` being correct proves nothing about whether the control
    plane passes it anything -- which is precisely how budget grants were issued,
    reported on and expired for a month while licensing nothing.

    The local quota of 16 TPU would admit this. The grant of 8 does not.
    """

    result = await _plane_with(_grant(limits=ResourceLimits(cpu=16, memory_gib=1, tpu=8)), tpu=12)

    assert not result.placements


@pytest.mark.asyncio
async def test_the_same_plane_admits_the_work_within_its_slice() -> None:
    """The gate permits work within every explicitly delegated dimension.

    The workload needs CPU and memory as well as TPU; omitted grant dimensions
    carry zero authority even when the local quota constrains only TPU.
    """

    result = await _plane_with(_grant(limits=ResourceLimits(cpu=16, memory_gib=1, tpu=8)), tpu=8)

    assert [placement.workload_id for placement in result.placements]


@pytest.mark.asyncio
async def test_without_a_grant_the_local_quota_alone_applies() -> None:
    """A non-federated plane must be untouched by any of this."""

    result = await _plane_with(None, tpu=12)

    assert [placement.workload_id for placement in result.placements]
