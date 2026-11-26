# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Federation: many planes, one tenant, one budget that actually binds.

``BudgetRecord.spent`` is one counter in one database. A tenant with projects on
three planes therefore has three independent budgets, each enforcing its own
limit and each reporting full compliance while the tenant spends three times what
its operator set. Every plane's answer is locally true, which is why nothing
surfaces the problem today.

Grants bound it without consensus. The home plane owns the total and cuts slices
for peers; because it refuses to cut more than the limit holds, the fleet-wide
total is bounded by construction and no plane consults another at admission time.

The properties asserted here are the ones that make that argument hold:
the home plane cannot over-issue, local spend counts against the same limit,
re-issuing retires admission rights without refunding issued authority, a grant
expires without anyone acting, and reported spend never walks backwards.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.records import BudgetRecord, Principal, Role, RoleGrant, utcnow
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import InputError, NotFoundError
from eyvan.server.service.federation import (
    BudgetGrantRecord,
    GrantState,
    PlanePeerRecord,
    federated_spend,
    issuable_amount,
    outstanding_total,
)

_HOME = "us-east5"
_PEER = "europe-west4"


def _admin() -> Principal:
    return Principal(
        subject="root",
        grants=(RoleGrant(role=Role.FLEET_ADMIN), RoleGrant(role=Role.ORG_ADMIN, org_id="acme")),
    )


async def _service(*, plane_id: str = _HOME) -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(repository=repository, plane_id=plane_id)


def _grant(**overrides: object) -> BudgetGrantRecord:
    now = utcnow()
    defaults: dict[str, object] = {
        "id": "bgr-1",
        "org_id": "acme",
        "project_id": "research",
        "home_plane_id": _HOME,
        "grantee_plane_id": _PEER,
        "amount": 100.0,
        "issued_at": now,
        "expires_at": now + timedelta(days=7),
    }
    return BudgetGrantRecord(**{**defaults, **overrides})


async def _seed(service: ControlPlaneService, *, limit: float = 1000.0, spent: float = 0.0) -> None:
    async with service.repository.transaction() as tx:
        await tx.put_budget(
            BudgetRecord(org_id="acme", project_id="research", limit=limit, spent=spent, updated_by="root")
        )
        await tx.put_plane_peer(PlanePeerRecord(plane_id=_PEER, endpoint="https://eu.example.test"))


def test_a_plane_cannot_grant_budget_to_itself() -> None:
    """It already holds the limit; a self-grant would double-count it."""

    with pytest.raises(ValueError, match="itself"):
        _grant(grantee_plane_id=_HOME)


def test_a_grant_must_expire_after_it_is_issued() -> None:
    """A grant that cannot lapse makes a vanished home plane an unbounded spend."""

    now = utcnow()
    with pytest.raises(ValueError, match="expire"):
        _grant(issued_at=now, expires_at=now)


def test_a_peer_endpoint_must_be_an_http_url() -> None:
    with pytest.raises(ValueError, match="http"):
        PlanePeerRecord(plane_id=_PEER, endpoint="eu.example.test")


def test_issuable_headroom_counts_both_grants_and_local_spend() -> None:
    """A home plane that spent most of the budget cannot grant the rest away."""

    now = utcnow()
    grants = [_grant(amount=300.0)]

    assert issuable_amount(1000.0, grants, now=now) == 700.0
    assert issuable_amount(1000.0, grants, now=now, local_spent=500.0) == 200.0
    assert issuable_amount(1000.0, grants, now=now, local_spent=900.0) == 0.0, "never negative"


def test_an_expired_grant_retains_its_headroom() -> None:
    now = utcnow()
    stale = _grant(amount=400.0, issued_at=now - timedelta(days=30), expires_at=now - timedelta(days=1))

    assert not stale.active_at(now)
    assert issuable_amount(1000.0, [stale], now=now) == 600.0
    assert outstanding_total([stale], now=now) == 400.0


def test_a_revoked_grant_retains_its_headroom() -> None:
    now = utcnow()
    revoked = _grant(amount=400.0, state=GrantState.REVOKED)

    assert not revoked.active_at(now)
    assert issuable_amount(1000.0, [revoked], now=now) == 600.0
    assert outstanding_total([revoked], now=now) == 400.0


def test_replacing_a_grant_does_not_exclude_its_retained_authority() -> None:
    """Replacement is not proof that the peer finished using the old slice."""

    now = utcnow()
    grants = [_grant(id="bgr-1", amount=900.0)]

    assert issuable_amount(1000.0, grants, now=now) == 100.0
    assert issuable_amount(1000.0, grants, now=now, replacing="bgr-1") == 100.0


def test_federated_spend_sums_local_and_reported() -> None:
    total = federated_spend(120.0, [_grant(reported_spend=80.0), _grant(id="bgr-2", reported_spend=40.0)])

    assert total == 240.0


@pytest.mark.asyncio
async def test_a_plane_without_an_identity_cannot_issue_grants() -> None:
    """Silently issuing from an anonymous plane would make the home unattributable."""

    service = await _service(plane_id="")
    await _seed(service)

    with pytest.raises(InputError, match="plane_id"):
        await service.issue_budget_grant(
            _admin(),
            org_id="acme",
            project_id="research",
            grantee_plane_id=_PEER,
            amount=100.0,
            expires_at=utcnow() + timedelta(days=7),
            idempotency_key="grant-1",
        )


@pytest.mark.asyncio
async def test_a_grant_to_an_unregistered_peer_is_refused() -> None:
    """Peers are configured, never discovered: an unknown plane has no standing."""

    service = await _service()
    async with service.repository.transaction() as tx:
        await tx.put_budget(BudgetRecord(org_id="acme", project_id="research", limit=1000.0, updated_by="root"))

    with pytest.raises(NotFoundError, match="peer"):
        await service.issue_budget_grant(
            _admin(),
            org_id="acme",
            project_id="research",
            grantee_plane_id=_PEER,
            amount=100.0,
            expires_at=utcnow() + timedelta(days=7),
            idempotency_key="grant-1",
        )


@pytest.mark.asyncio
async def test_a_grant_within_the_limit_is_issued() -> None:
    service = await _service()
    await _seed(service, limit=1000.0)

    grant = await service.issue_budget_grant(
        _admin(),
        org_id="acme",
        project_id="research",
        grantee_plane_id=_PEER,
        amount=400.0,
        expires_at=utcnow() + timedelta(days=7),
        idempotency_key="grant-1",
    )

    assert grant.amount == 400.0
    assert grant.home_plane_id == _HOME
    assert grant.grantee_plane_id == _PEER
    assert grant.state is GrantState.ACTIVE


@pytest.mark.asyncio
async def test_the_home_plane_refuses_to_grant_past_its_limit() -> None:
    """This single check is the entire cross-plane guarantee."""

    service = await _service()
    await _seed(service, limit=1000.0)

    with pytest.raises(InputError, match="exceeds"):
        await service.issue_budget_grant(
            _admin(),
            org_id="acme",
            project_id="research",
            grantee_plane_id=_PEER,
            amount=1500.0,
            expires_at=utcnow() + timedelta(days=7),
            idempotency_key="grant-1",
        )


@pytest.mark.asyncio
async def test_local_spend_reduces_what_can_be_granted_away() -> None:
    """Otherwise the home plane's own spend plus its grants exceed the limit."""

    service = await _service()
    await _seed(service, limit=1000.0, spent=800.0)

    with pytest.raises(InputError, match="exceeds"):
        await service.issue_budget_grant(
            _admin(),
            org_id="acme",
            project_id="research",
            grantee_plane_id=_PEER,
            amount=300.0,
            expires_at=utcnow() + timedelta(days=7),
            idempotency_key="grant-1",
        )


@pytest.mark.asyncio
async def test_reissuing_replaces_admission_but_retains_both_monetary_slices() -> None:
    """A replacement uses remaining headroom, not unproven released rights."""

    service = await _service()
    await _seed(service, limit=1000.0)
    expires = utcnow() + timedelta(days=7)
    first = await service.issue_budget_grant(
        _admin(),
        org_id="acme",
        project_id="research",
        grantee_plane_id=_PEER,
        amount=400.0,
        expires_at=expires,
        idempotency_key="grant-1",
    )

    with pytest.raises(InputError, match="exceeds"):
        await service.issue_budget_grant(
            _admin(),
            org_id="acme",
            project_id="research",
            grantee_plane_id=_PEER,
            amount=900.0,
            expires_at=expires,
            idempotency_key="grant-2",
        )
    async with service.repository.transaction() as tx:
        retained = await tx.get_budget_grant(first.id)
    assert retained is not None and retained.state is GrantState.ACTIVE

    second = await service.issue_budget_grant(
        _admin(),
        org_id="acme",
        project_id="research",
        grantee_plane_id=_PEER,
        amount=600.0,
        expires_at=expires,
        idempotency_key="grant-2",
    )

    assert second.id != first.id
    assert second.amount == 600.0
    async with service.repository.transaction() as tx:
        retained = await tx.get_budget_grant(first.id)
        grants = await tx.list_budget_grants(org_id="acme", project_id="research")
    assert retained is not None and retained.state is GrantState.REVOKED
    assert issuable_amount(1000.0, grants, now=utcnow()) == 0.0
    status = await service.federated_budget_status(_admin(), org_id="acme", project_id="research")
    assert status.granted_away == 1000.0
    assert status.active_grants == 1


@pytest.mark.asyncio
async def test_a_replayed_grant_request_returns_the_same_grant() -> None:
    service = await _service()
    await _seed(service, limit=1000.0)
    expires = utcnow() + timedelta(days=7)
    arguments = {
        "org_id": "acme",
        "project_id": "research",
        "grantee_plane_id": _PEER,
        "amount": 400.0,
        "expires_at": expires,
        "idempotency_key": "grant-1",
    }

    first = await service.issue_budget_grant(_admin(), **arguments)
    second = await service.issue_budget_grant(_admin(), **arguments)

    assert second.id == first.id
    status = await service.federated_budget_status(_admin(), org_id="acme", project_id="research")
    assert status.granted_away == 400.0, "a replay must not grant twice"


@pytest.mark.asyncio
async def test_revoking_stops_new_admission_without_reclaiming_spend() -> None:
    """The money is gone; pretending otherwise makes the ledger disagree with the bill."""

    service = await _service()
    await _seed(service, limit=1000.0)
    grant = await service.issue_budget_grant(
        _admin(),
        org_id="acme",
        project_id="research",
        grantee_plane_id=_PEER,
        amount=400.0,
        expires_at=utcnow() + timedelta(days=7),
        idempotency_key="grant-1",
    )
    await service.record_grant_spend(grant_id=grant.id, reported_spend=250.0)

    revoked = await service.revoke_budget_grant(_admin(), grant_id=grant.id)

    assert revoked.state is GrantState.REVOKED
    assert revoked.reported_spend == 250.0, "revocation is not a refund"


@pytest.mark.asyncio
async def test_revoking_twice_is_idempotent() -> None:
    service = await _service()
    await _seed(service, limit=1000.0)
    grant = await service.issue_budget_grant(
        _admin(),
        org_id="acme",
        project_id="research",
        grantee_plane_id=_PEER,
        amount=400.0,
        expires_at=utcnow() + timedelta(days=7),
        idempotency_key="grant-1",
    )
    await service.revoke_budget_grant(_admin(), grant_id=grant.id)

    again = await service.revoke_budget_grant(_admin(), grant_id=grant.id)

    assert again.state is GrantState.REVOKED


@pytest.mark.asyncio
async def test_reported_spend_never_walks_backwards() -> None:
    """A peer restarting and reporting a fresh counter must not look like a refund."""

    service = await _service()
    await _seed(service, limit=1000.0)
    grant = await service.issue_budget_grant(
        _admin(),
        org_id="acme",
        project_id="research",
        grantee_plane_id=_PEER,
        amount=400.0,
        expires_at=utcnow() + timedelta(days=7),
        idempotency_key="grant-1",
    )
    await service.record_grant_spend(grant_id=grant.id, reported_spend=300.0)

    regressed = await service.record_grant_spend(grant_id=grant.id, reported_spend=10.0)

    assert regressed.reported_spend == 300.0


@pytest.mark.asyncio
async def test_the_federated_status_reveals_the_overspend_a_local_budget_hides() -> None:
    """The whole reason this exists: three compliant planes, one exceeded limit."""

    service = await _service()
    await _seed(service, limit=1000.0, spent=400.0)
    grant = await service.issue_budget_grant(
        _admin(),
        org_id="acme",
        project_id="research",
        grantee_plane_id=_PEER,
        amount=500.0,
        expires_at=utcnow() + timedelta(days=7),
        idempotency_key="grant-1",
    )
    await service.record_grant_spend(grant_id=grant.id, reported_spend=450.0)

    status = await service.federated_budget_status(_admin(), org_id="acme", project_id="research")

    assert status.local_spent == 400.0
    assert status.reported_peer_spend == 450.0
    assert status.total_spent == 850.0, "neither plane alone would report this"
    assert status.stale_reports == 0


@pytest.mark.asyncio
async def test_a_peer_that_has_never_reported_is_named_not_counted_as_idle() -> None:
    """ "The peers are idle" and "the peers are not talking" are different answers."""

    service = await _service()
    await _seed(service, limit=1000.0)
    await service.issue_budget_grant(
        _admin(),
        org_id="acme",
        project_id="research",
        grantee_plane_id=_PEER,
        amount=500.0,
        expires_at=utcnow() + timedelta(days=7),
        idempotency_key="grant-1",
    )

    status = await service.federated_budget_status(_admin(), org_id="acme", project_id="research")

    assert status.reported_peer_spend == 0.0
    assert status.stale_reports == 1


@pytest.mark.asyncio
async def test_an_elapsed_grant_is_marked_expired_without_anyone_acting() -> None:
    """An unreachable home plane must fail closed, not leave a peer spending forever."""

    service = await _service()
    await _seed(service, limit=1000.0)
    now = utcnow()
    grant = await service.issue_budget_grant(
        _admin(),
        org_id="acme",
        project_id="research",
        grantee_plane_id=_PEER,
        amount=400.0,
        expires_at=now + timedelta(hours=1),
        idempotency_key="grant-1",
        now=now,
    )

    expired = await service.expire_budget_grants_once(now=now + timedelta(hours=2))

    assert expired == 1
    async with service.repository.transaction() as tx:
        stored = await tx.get_budget_grant(grant.id)
    assert stored is not None
    assert stored.state is GrantState.EXPIRED


@pytest.mark.asyncio
async def test_the_expiry_sweep_is_idempotent() -> None:
    service = await _service()
    await _seed(service, limit=1000.0)
    now = utcnow()
    await service.issue_budget_grant(
        _admin(),
        org_id="acme",
        project_id="research",
        grantee_plane_id=_PEER,
        amount=400.0,
        expires_at=now + timedelta(hours=1),
        idempotency_key="grant-1",
        now=now,
    )
    later = now + timedelta(hours=2)
    await service.expire_budget_grants_once(now=later)

    assert await service.expire_budget_grants_once(now=later) == 0


@pytest.mark.asyncio
async def test_an_expired_grant_only_allows_a_new_slice_within_retained_headroom() -> None:
    service = await _service()
    await _seed(service, limit=1000.0)
    now = utcnow()
    first = await service.issue_budget_grant(
        _admin(),
        org_id="acme",
        project_id="research",
        grantee_plane_id=_PEER,
        amount=900.0,
        expires_at=now + timedelta(hours=1),
        idempotency_key="grant-1",
        now=now,
    )
    later = now + timedelta(hours=2)

    with pytest.raises(InputError, match="exceeds"):
        await service.issue_budget_grant(
            _admin(),
            org_id="acme",
            project_id="research",
            grantee_plane_id=_PEER,
            amount=900.0,
            expires_at=later + timedelta(days=7),
            idempotency_key="grant-2",
            now=later,
        )
    replacement = await service.issue_budget_grant(
        _admin(),
        org_id="acme",
        project_id="research",
        grantee_plane_id=_PEER,
        amount=100.0,
        expires_at=later + timedelta(days=7),
        idempotency_key="grant-2",
        now=later,
    )

    assert replacement.amount == 100.0
    async with service.repository.transaction() as tx:
        retained = await tx.get_budget_grant(first.id)
        grants = await tx.list_budget_grants(org_id="acme", project_id="research")
    assert retained is not None and not retained.active_at(later)
    assert issuable_amount(1000.0, grants, now=later) == 0.0
    status = await service.federated_budget_status(_admin(), org_id="acme", project_id="research", now=later)
    assert status.granted_away == 1000.0
    assert status.active_grants == 1


@pytest.mark.asyncio
async def test_a_plane_does_not_register_itself_as_a_peer() -> None:
    service = await _service()

    with pytest.raises(InputError, match="own peer"):
        await service.put_plane_peer(_admin(), plane_id=_HOME, endpoint="https://home.example.test")


@pytest.mark.asyncio
async def test_registering_a_peer_twice_preserves_its_creation_time() -> None:
    service = await _service()
    first = await service.put_plane_peer(_admin(), plane_id=_PEER, endpoint="https://eu.example.test")

    second = await service.put_plane_peer(
        _admin(), plane_id=_PEER, endpoint="https://eu2.example.test", description="moved"
    )

    assert second.created_at == first.created_at
    assert second.endpoint == "https://eu2.example.test"
    assert [peer.plane_id for peer in await service.list_plane_peers(_admin())] == [_PEER]


# --------------------------------------------------------------------------- reachability


def test_a_deployed_plane_takes_its_identity_from_its_settings(tmp_path) -> None:
    """The seam that made all of the above true and none of it real.

    ``EyvanSettings.plane_id`` existed and was read from ``EYVAN_PLANE_ID``; the
    service parameter existed and every test above set it directly; and
    ``create_app`` passed neither. So a deployed plane always had an empty
    identity, which means it refused to issue grants *and* skipped consulting
    them at admission -- federation configurable, tested, and off everywhere.

    Asserted through ``create_app`` rather than the constructor because the
    constructor was never the thing that was wrong.
    """

    from eyvan.config import EyvanSettings
    from eyvan.server.api.app import create_app

    app = create_app(
        settings=EyvanSettings(
            artifact_token_key="k" * 32,
            artifact_store_uri=str(tmp_path / "managed-artifacts"),
            plane_id="us-plane",
        )
    )

    assert app.state.service.plane_id == "us-plane"


def test_the_federation_commands_are_attached() -> None:
    """Grants were built, tested, wired into admission -- and had no command.

    A mechanism that can only be invoked by hand-writing HTTP against a route
    that did not exist bounds nothing, which is what "entirely unreachable"
    meant in the plan.
    """

    from eyvan.sdk import cli

    assert "federation" in cli.main.commands
    groups = cli.main.commands["federation"].commands
    assert set(groups) == {"peer", "grant", "quota-grant"}
    assert set(groups["grant"].commands) == {"list", "issue", "revoke"}
    assert set(groups["quota-grant"].commands) == {"list", "issue", "revoke"}
    assert set(groups["peer"].commands) == {"list", "add"}


def test_issuing_a_grant_from_the_cli_requires_an_expiry() -> None:
    """The one option with no safe default.

    A grant that cannot lapse turns a home plane going away permanently into
    unbounded spend on every peer that ever held one. The record refuses to be
    built without an expiry; this asserts the CLI refuses before the request is
    ever sent, so the refusal does not depend on reaching a server.
    """

    from click.testing import CliRunner

    from eyvan.sdk import cli

    result = CliRunner().invoke(
        cli.main,
        ["--user", "root", "--role", "fleet_admin", "federation", "grant", "issue", "eu-plane", "--amount", "10"],
    )

    assert result.exit_code != 0
    assert "--expires-at" in result.output
