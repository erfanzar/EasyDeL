# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Granting access without an identity-provider change.

``eyvan_grants`` in an OIDC token was the only source of authority. Onboarding a
tenant meant teaching an identity provider to mint a bespoke claim that
deserializes into ``tuple[RoleGrant, ...]`` -- a claim named nowhere in the
README -- and revoking access meant another IdP change. A deployment whose IdP
could not be taught a custom claim could not use Eyvan's authorization at all.

Durable grants are a second source, unioned with the claim rather than replacing
it, so a deployment already minting the claim keeps working unchanged.

Most of what follows is about escalation, because whoever can issue grants can
otherwise write themselves a better one. The gate scales with what is being
handed out rather than being a single "may administer grants" permission.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import Permission, Principal, Role, RoleGrant, RoleGrantRecord
from eyvan.server.api.app import create_app
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.authz import Authorizer
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import AuthorizationError
from eyvan.server.service.scheduling import CoreSchedulingService

FLEET_ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _service() -> ControlPlaneService:
    return ControlPlaneService(InMemoryRepository(), scheduler=CoreSchedulingService())


def _principal(role: Role, *, org: str = "acme", project: str = "research", subject: str = "p") -> Principal:
    if role is Role.FLEET_ADMIN:
        return Principal(subject=subject, grants=(RoleGrant(role=role),))
    if role is Role.ORG_ADMIN:
        return Principal(subject=subject, grants=(RoleGrant(role=role, org_id=org),))
    return Principal(subject=subject, grants=(RoleGrant(role=role, org_id=org, project_id=project),))


def _record(
    role: Role,
    *,
    subject: str = "alice",
    org: str | None = "acme",
    project: str | None = "research",
    expires_at: datetime | None = None,
) -> RoleGrantRecord:
    return RoleGrantRecord(
        subject=subject,
        grant=RoleGrant(role=role, org_id=org, project_id=project),
        expires_at=expires_at,
        granted_by="unused",
    )


# --------------------------------------------------------------------------- issuing


async def test_a_durable_grant_confers_authority() -> None:
    """The capability that did not exist without an IdP change."""

    service = _service()
    await service.issue_role_grant(FLEET_ADMIN, _record(Role.PROJECT_OPERATOR), idempotency_key="g1")

    grants = await service.active_grants_for("alice")

    assert [grant.role for grant in grants] == [Role.PROJECT_OPERATOR]
    assert Authorizer().allows(
        Principal(subject="alice", grants=grants), Permission.SUBMIT, org_id="acme", project_id="research"
    )


async def test_the_issuer_is_recorded_from_the_caller_not_the_request() -> None:
    """An accountability field a caller can set is not accountability."""

    service = _service()

    stored, _ = await service.issue_role_grant(FLEET_ADMIN, _record(Role.PROJECT_OPERATOR), idempotency_key="g1")

    assert stored.granted_by == "root"


async def test_re_issuing_the_same_grant_overwrites_rather_than_accumulating() -> None:
    """Holding a role twice means nothing, and a list that grows on retry is a leak."""

    service = _service()
    await service.issue_role_grant(FLEET_ADMIN, _record(Role.PROJECT_OPERATOR), idempotency_key="g1")
    await service.issue_role_grant(FLEET_ADMIN, _record(Role.PROJECT_OPERATOR), idempotency_key="g2")

    assert len(await service.list_role_grants(FLEET_ADMIN)) == 1


async def test_issuing_is_idempotent_on_its_key() -> None:
    """Onboarding runs from configuration management, which retries."""

    service = _service()
    first, replayed_first = await service.issue_role_grant(
        FLEET_ADMIN, _record(Role.PROJECT_OPERATOR), idempotency_key="g1"
    )
    second, replayed_second = await service.issue_role_grant(
        FLEET_ADMIN, _record(Role.PROJECT_OPERATOR), idempotency_key="g1"
    )

    assert (replayed_first, replayed_second) == (False, True)
    assert first.granted_at == second.granted_at


# --------------------------------------------------------------------------- escalation


async def test_a_project_admin_cannot_grant_organization_administration() -> None:
    """The escalation that matters: writing yourself a better grant.

    A project admin may delegate inside their project. Appointing an
    organization administrator would hand out authority over every sibling
    project, including the power to reassign their own.
    """

    service = _service()

    with pytest.raises(AuthorizationError):
        await service.issue_role_grant(
            _principal(Role.PROJECT_ADMIN),
            _record(Role.ORG_ADMIN, project=None),
            idempotency_key="g1",
        )


async def test_nobody_but_a_fleet_admin_grants_fleet_administration() -> None:
    """It is unscoped and reaches every tenant, so it is the operator's to give."""

    service = _service()

    for role in (
        Role.ORG_ADMIN,
        Role.PROJECT_ADMIN,
        Role.PROJECT_OPERATOR,
        Role.PROJECT_SUBMITTER,
        Role.PROJECT_VIEWER,
    ):
        with pytest.raises(AuthorizationError):
            await service.issue_role_grant(
                _principal(role),
                _record(Role.FLEET_ADMIN, org=None, project=None),
                idempotency_key=f"g-{role.value}",
            )


async def test_an_org_admin_grants_only_inside_its_own_organization() -> None:
    """Otherwise a tenant's admin writes grants into a tenant they do not run."""

    service = _service()

    stored, _ = await service.issue_role_grant(
        _principal(Role.ORG_ADMIN, org="acme"),
        _record(Role.PROJECT_OPERATOR, org="acme"),
        idempotency_key="own",
    )
    assert stored.grant.org_id == "acme"

    with pytest.raises(AuthorizationError):
        await service.issue_role_grant(
            _principal(Role.ORG_ADMIN, org="acme"),
            _record(Role.PROJECT_OPERATOR, org="rival"),
            idempotency_key="other",
        )


async def test_a_project_admin_delegates_only_inside_its_own_project() -> None:
    """Adding an operator to your own team is ordinary; adding one to a sibling is not."""

    service = _service()

    stored, _ = await service.issue_role_grant(
        _principal(Role.PROJECT_ADMIN, project="research"),
        _record(Role.PROJECT_OPERATOR, project="research"),
        idempotency_key="own",
    )
    assert stored.grant.project_id == "research"

    with pytest.raises(AuthorizationError):
        await service.issue_role_grant(
            _principal(Role.PROJECT_ADMIN, project="research"),
            _record(Role.PROJECT_OPERATOR, project="platform"),
            idempotency_key="sibling",
        )


async def test_a_viewer_grants_nothing_at_all() -> None:
    """Read access must not be a path to write access."""

    service = _service()

    for role in Role:
        with pytest.raises(AuthorizationError):
            await service.issue_role_grant(
                _principal(Role.PROJECT_VIEWER),
                _record(
                    role,
                    org=None if role is Role.FLEET_ADMIN else "acme",
                    project=None if role in {Role.FLEET_ADMIN, Role.ORG_ADMIN} else "research",
                ),
                idempotency_key=f"g-{role.value}",
            )


# --------------------------------------------------------------------------- revocation


async def test_revocation_takes_the_same_authority_as_issuance() -> None:
    """Nobody should be able to strip access they could not have given.

    Stripping is a denial of service against another tenant's team, and it needs
    no request body to get wrong.
    """

    service = _service()
    await service.issue_role_grant(FLEET_ADMIN, _record(Role.PROJECT_OPERATOR), idempotency_key="g1")

    with pytest.raises(AuthorizationError):
        await service.revoke_role_grant(
            _principal(Role.ORG_ADMIN, org="rival"), ("alice", "project_operator", "acme", "research")
        )

    assert await service.revoke_role_grant(FLEET_ADMIN, ("alice", "project_operator", "acme", "research")) is True
    assert await service.active_grants_for("alice") == ()


async def test_revoking_a_grant_that_is_not_there_reports_it() -> None:
    """A silent success would read as "access removed" when nothing was."""

    service = _service()

    assert await service.revoke_role_grant(FLEET_ADMIN, ("ghost", "project_viewer", "acme", "research")) is False


async def test_an_expired_grant_confers_nothing() -> None:
    """Time-boxed elevation is only elevation if it actually ends."""

    service = _service()
    await service.issue_role_grant(
        FLEET_ADMIN,
        _record(Role.PROJECT_OPERATOR, expires_at=datetime.now(UTC) + timedelta(minutes=5)),
        idempotency_key="g1",
    )

    assert len(await service.active_grants_for("alice")) == 1
    assert await service.active_grants_for("alice", now=datetime.now(UTC) + timedelta(hours=1)) == ()


async def test_a_grant_cannot_expire_before_it_is_issued() -> None:
    """An already-dead grant reads as access someone believes they gave."""

    with pytest.raises(ValueError, match="after the moment it was granted"):
        RoleGrantRecord(
            subject="alice",
            grant=RoleGrant(role=Role.PROJECT_VIEWER, org_id="acme", project_id="research"),
            granted_by="root",
            granted_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) - timedelta(minutes=1),
        )


# --------------------------------------------------------------------------- listing


async def test_listing_grants_is_not_a_staff_directory() -> None:
    """A grant names a person and what they can reach.

    An unfiltered listing would tell any tenant who works at every other one.
    """

    service = _service()
    await service.issue_role_grant(FLEET_ADMIN, _record(Role.PROJECT_OPERATOR, org="acme"), idempotency_key="g1")
    await service.issue_role_grant(
        FLEET_ADMIN, _record(Role.PROJECT_OPERATOR, subject="bob", org="rival"), idempotency_key="g2"
    )

    scoped = await service.list_role_grants(_principal(Role.ORG_ADMIN, org="acme"))

    assert [item.subject for item in scoped] == ["alice"]
    assert len(await service.list_role_grants(FLEET_ADMIN)) == 2


async def test_an_operator_cannot_enumerate_grants_at_all() -> None:
    """Only someone who administers a scope has business reading who can reach it."""

    service = _service()
    await service.issue_role_grant(FLEET_ADMIN, _record(Role.PROJECT_OPERATOR), idempotency_key="g1")

    assert await service.list_role_grants(_principal(Role.PROJECT_OPERATOR)) == []


# --------------------------------------------------------------------------- the seam


def _client() -> TestClient:
    return TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True, artifact_token_key="k" * 32)))


def _headers(user: str, roles: str, **extra: str) -> dict[str, str]:
    return {"X-Eyvan-User": user, "X-Eyvan-Roles": roles, **extra}


def test_a_durable_grant_authorizes_a_real_request() -> None:
    """The seam that makes the records real rather than storage nobody consults.

    Without merging durable grants into the authenticated principal, the routes
    would issue grants, the ledger would record them, and no request would ever
    be authorized by one.
    """

    client = _client()
    admin = _headers("root", "fleet_admin")
    alice = _headers("alice", "viewer", **{"X-Eyvan-Org": "acme", "X-Eyvan-Project": "research"})
    submission = {
        "org_id": "acme",
        "project_id": "research",
        "spec": {"name": "t", "execution": {"kind": "command", "command": "echo hi"}},
    }

    before = client.post("/api/v1/workloads", json=submission, headers={**alice, "Idempotency-Key": "w1"})
    issued = client.post(
        "/api/v1/grants",
        json={"subject": "alice", "grant": {"role": "project_operator", "org_id": "acme", "project_id": "research"}},
        headers={**admin, "Idempotency-Key": "g1"},
    )
    after = client.post("/api/v1/workloads", json=submission, headers={**alice, "Idempotency-Key": "w2"})

    assert before.status_code == 403
    assert issued.status_code == 201
    assert after.status_code == 201


def test_revocation_takes_effect_on_the_next_request() -> None:
    """Read per request, not cached: "I removed their access" has to be true now.

    A cache would make it true eventually, which is the wrong answer during the
    incident that prompted the revocation.
    """

    client = _client()
    admin = _headers("root", "fleet_admin")
    alice = _headers("alice", "viewer", **{"X-Eyvan-Org": "acme", "X-Eyvan-Project": "research"})
    submission = {
        "org_id": "acme",
        "project_id": "research",
        "spec": {"name": "t", "execution": {"kind": "command", "command": "echo hi"}},
    }
    client.post(
        "/api/v1/grants",
        json={"subject": "alice", "grant": {"role": "project_operator", "org_id": "acme", "project_id": "research"}},
        headers={**admin, "Idempotency-Key": "g1"},
    )
    assert (
        client.post("/api/v1/workloads", json=submission, headers={**alice, "Idempotency-Key": "w1"}).status_code == 201
    )

    revoked = client.request(
        "DELETE",
        "/api/v1/grants",
        params={"subject": "alice", "role": "project_operator", "org_id": "acme", "project_id": "research"},
        headers=admin,
    )

    assert revoked.status_code == 204
    assert (
        client.post("/api/v1/workloads", json=submission, headers={**alice, "Idempotency-Key": "w2"}).status_code == 403
    )


def test_a_claim_grant_survives_alongside_a_durable_one() -> None:
    """Union, not replacement -- and the distinguishing case needs both present.

    A subject holding only a claim grant short-circuits before the merge, so it
    proves nothing about the merge itself. This one holds a claim grant in one
    project and a durable grant in another: if the durable source replaced the
    claim rather than joining it, the first project would stop working the
    moment anyone was granted anything anywhere.
    """

    client = _client()
    admin = _headers("root", "fleet_admin")
    dave = _headers("dave", "operator", **{"X-Eyvan-Org": "acme", "X-Eyvan-Project": "research"})
    client.post(
        "/api/v1/grants",
        json={"subject": "dave", "grant": {"role": "project_operator", "org_id": "acme", "project_id": "platform"}},
        headers={**admin, "Idempotency-Key": "g1"},
    )

    def submit(project: str, key: str) -> int:
        return client.post(
            "/api/v1/workloads",
            json={
                "org_id": "acme",
                "project_id": project,
                "spec": {"name": "t", "execution": {"kind": "command", "command": "echo hi"}},
            },
            headers={**dave, "Idempotency-Key": key},
        ).status_code

    # From the claim.
    assert submit("research", "w1") == 201
    # From the durable grant.
    assert submit("platform", "w2") == 201


def test_the_claim_still_works_on_its_own() -> None:
    """A deployment already minting ``eyvan_grants`` must keep working unchanged.

    The two sources are unioned, not swapped: this is an addition to how
    authority arrives, not a migration everyone has to perform.
    """

    client = _client()
    operator = _headers("carol", "operator", **{"X-Eyvan-Org": "acme", "X-Eyvan-Project": "research"})

    response = client.post(
        "/api/v1/workloads",
        json={
            "org_id": "acme",
            "project_id": "research",
            "spec": {"name": "t", "execution": {"kind": "command", "command": "echo hi"}},
        },
        headers={**operator, "Idempotency-Key": "w1"},
    )

    assert response.status_code == 201
    assert client.get("/api/v1/grants", headers=_headers("root", "fleet_admin")).json() == []
