# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One tenant, several control planes, and one budget that actually binds.

``BudgetRecord.spent`` is a single durable counter in one database. A tenant
with projects on three planes therefore had three independent budgets, each
enforcing its own limit and each reporting full compliance while the tenant
spent three times what its operator set.

Grants were built to fix exactly this -- issuance, expiry, reporting, a
migration, twenty-five tests -- and nothing consulted them where it mattered.
``list_budget_grants`` was called from issuance, from the status view and from
the expiry sweep, and never from admission. A grant was recorded, and licensed
nothing.

These assert the licence. The distinguishing case is the last one: a plane whose
*local* budget would admit the work and whose grant would not.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from eyvan.records import BudgetRecord, utcnow
from eyvan.server.service.federation import BudgetGrantRecord, GrantState
from eyvan.server.service.scheduling import federated_budgets

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def _budget(limit: float = 1000.0, *, spent: float = 0.0, hard: bool = True) -> BudgetRecord:
    return BudgetRecord(
        org_id="acme",
        project_id="research",
        limit=limit,
        spent=spent,
        hard_limit=hard,
        updated_by="root",
    )


def _grant(
    amount: float = 100.0,
    *,
    grantee: str = "plane-b",
    state: GrantState = GrantState.ACTIVE,
    hours: int = 24,
    anchor: datetime = NOW,
):
    """Build one grant relative to ``anchor``.

    ``hours`` is measured from the anchor, and issuance is always earlier still,
    so an already-lapsed grant stays constructible -- a record whose expiry
    precedes its issuance is refused by the model, and rightly.
    """

    return BudgetGrantRecord(
        id=f"grant-{grantee}-{amount:g}-{hours}",
        org_id="acme",
        project_id="research",
        home_plane_id="plane-a",
        grantee_plane_id=grantee,
        amount=amount,
        state=state,
        issued_at=anchor - timedelta(days=7),
        expires_at=anchor + timedelta(hours=hours),
    )


def _limit(budgets) -> float:
    assert len(budgets) == 1
    return float(budgets[0].limit)


# --------------------------------------------------------------------------- unchanged


def test_a_plane_holding_no_grants_is_unaffected() -> None:
    """The home plane, and every non-federated deployment.

    Which is all of them by default, so this is the case that must not move.
    """

    budgets = federated_budgets([_budget(1000.0)], [], plane_id="plane-a", now=NOW)

    assert _limit(budgets) == 1000.0


def test_a_plane_with_no_identity_ignores_grants_entirely() -> None:
    """An empty ``plane_id`` means "not federated".

    A plane that cannot be a grantee must not accidentally match one, or a
    misaddressed grant would silently retune a limit nobody meant to change.
    """

    budgets = federated_budgets([_budget(1000.0)], [_grant(100.0)], plane_id="", now=NOW)

    assert _limit(budgets) == 1000.0


def test_a_grant_cannot_be_addressed_to_an_unidentified_plane() -> None:
    """What actually protects a non-federated plane, asserted where it lives.

    The ``if plane_id:`` short-circuit in the adapter is redundant given this,
    and a test that only exercised the short-circuit would pass with it removed.
    The invariant is the model's: a grantee is never the empty string, so a
    plane with no identity matches nothing by construction.
    """

    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _grant(100.0, grantee="")


def test_a_grant_addressed_to_another_plane_is_ignored() -> None:
    """A grantee's authority is what was issued *to it*, not what exists."""

    budgets = federated_budgets([_budget(1000.0)], [_grant(100.0, grantee="plane-c")], plane_id="plane-b", now=NOW)

    assert _limit(budgets) == 1000.0


# --------------------------------------------------------------------------- the licence


def test_a_grantee_is_bounded_by_its_grant_and_not_by_its_local_limit() -> None:
    """The correctness fix, in one assertion.

    Three shards each reporting compliance while the tenant spends three times
    its limit is the bug; this is the line that stops it.
    """

    budgets = federated_budgets([_budget(1000.0)], [_grant(100.0)], plane_id="plane-b", now=NOW)

    assert _limit(budgets) == 100.0


def test_several_grants_to_one_plane_add_up() -> None:
    """A home plane may slice its limit more than once, and both slices count."""

    budgets = federated_budgets([_budget(1000.0)], [_grant(100.0), _grant(50.0)], plane_id="plane-b", now=NOW)

    assert _limit(budgets) == 150.0


def test_a_local_limit_tighter_than_the_grant_still_binds() -> None:
    """A grant may not raise a ceiling a local operator set.

    The tighter of the two wins in both directions, which is the only rule under
    which neither party can be surprised.
    """

    budgets = federated_budgets([_budget(40.0)], [_grant(100.0)], plane_id="plane-b", now=NOW)

    assert _limit(budgets) == 40.0


def test_a_soft_local_budget_becomes_hard_under_a_grant() -> None:
    """Otherwise the grantee overspends its slice and the bound is void.

    A soft budget is a local operator's choice about their own money. A grant is
    somebody else's money, and it is not theirs to soften.
    """

    budgets = federated_budgets([_budget(1000.0, hard=False)], [_grant(100.0)], plane_id="plane-b", now=NOW)

    assert budgets[0].hard_limit is True
    assert _limit(budgets) == 100.0


def test_a_grantee_without_durable_budget_accounting_has_zero_authority() -> None:
    """A grant cannot replace the ledger that accumulates consumption."""

    budgets = federated_budgets([], [_grant(100.0)], plane_id="plane-b", now=NOW)

    assert _limit(budgets) == 0.0
    assert budgets[0].hard_limit is True
    assert budgets[0].spent == 0.0


# --------------------------------------------------------------------------- lapsing


def test_an_expired_grant_authorizes_nothing() -> None:
    """A grant that cannot lapse turns a home plane that goes away permanently
    into unbounded spend on every peer that ever held one."""

    budgets = federated_budgets([], [_grant(100.0, hours=-2)], plane_id="plane-b", now=NOW)

    assert _limit(budgets) == 0.0
    assert budgets[0].hard_limit is True


def test_a_revoked_grant_authorizes_nothing() -> None:
    """State, not only time: a home plane must be able to take a slice back."""

    budgets = federated_budgets([], [_grant(100.0, state=GrantState.REVOKED)], plane_id="plane-b", now=NOW)

    assert _limit(budgets) == 0.0
    assert budgets[0].hard_limit is True


def test_an_expired_grant_closes_even_a_funded_local_budget() -> None:
    """Inactive inbound history must not fall back to independent local rights."""

    budgets = federated_budgets([_budget(40.0, spent=10.0)], [_grant(100.0, hours=-2)], plane_id="plane-b", now=NOW)

    assert _limit(budgets) == 0.0
    assert budgets[0].hard_limit is True
    assert budgets[0].spent == 10.0


# --------------------------------------------------------------------------- end to end


async def test_admission_refuses_work_beyond_the_granted_headroom() -> None:
    """Through the real scheduling path, not the adapter alone.

    ``federated_budgets`` passing its own tests proves nothing about whether
    admission calls it -- which is precisely the shape this phase exists to fix.
    """

    from test_controller import principal

    from eyvan.records import (
        CellRecord,
        CellState,
        CommandExecution,
        ResourceVector,
        Role,
        WorkloadDefinition,
    )
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService(), plane_id="plane-b")
    async with repository.transaction() as tx:
        await tx.put_budget(_budget(1000.0))
        await tx.put_budget_grant(_grant(0.0, anchor=utcnow()))
        await tx.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=8, memory_gib=64),
                allocatable=ResourceVector(cpu=8, memory_gib=64),
                endpoint="http://agent:8090",
                unit_prices={"cpu": 1.0},
            )
        )
    await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            estimated_duration_seconds=3600,
        ),
        idempotency_key="train",
    )

    result = await service.schedule_once(principal(Role.FLEET_ADMIN))

    # A local limit of 1000 would admit this; a zero grant does not.
    assert not result.placements


async def test_the_same_plane_admits_the_work_once_it_holds_a_grant() -> None:
    """The gate must not block the configuration it exists to permit.

    Same fleet, same workload, same local budget -- only the grant differs, so a
    refusal here would mean the mechanism blocks everything rather than the
    right thing.
    """

    from test_controller import principal

    from eyvan.records import (
        CellRecord,
        CellState,
        CommandExecution,
        ResourceVector,
        Role,
        WorkloadDefinition,
    )
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService(), plane_id="plane-b")
    async with repository.transaction() as tx:
        await tx.put_budget(_budget(1000.0))
        await tx.put_budget_grant(_grant(500.0, anchor=utcnow()))
        await tx.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=8, memory_gib=64),
                allocatable=ResourceVector(cpu=8, memory_gib=64),
                endpoint="http://agent:8090",
                unit_prices={"cpu": 1.0},
            )
        )
    await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            estimated_duration_seconds=3600,
        ),
        idempotency_key="train",
    )

    result = await service.schedule_once(principal(Role.FLEET_ADMIN))

    assert result.placements
