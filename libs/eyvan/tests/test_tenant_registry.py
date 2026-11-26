# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Knowing which tenants exist, and refusing the ones that do not.

``org_id`` and ``project_id`` were free-form strings on every route, validated
for length and nothing else. Nothing could list tenants, and a typo in
``PUT /quotas`` created a quota for a scope that would never submit anything --
indistinguishable from success, and discovered only when someone asked why the
quota was not binding.

The enforcement here is deliberately *scoped* rather than global. An
organization nobody registered is the pre-registry world and is left alone, so
upgrading breaks nothing. Registering one is the operator stating they manage
tenancy on this plane, and from that point its projects have to be real. A
global switch defaulting to off would have left the typo in place for every
deployment that did not read the release notes; one defaulting to on would have
broken every existing deployment on upgrade.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import (
    CommandExecution,
    OrganizationRecord,
    Principal,
    ProjectRecord,
    QuotaRecord,
    ResourceLimits,
    Role,
    RoleGrant,
    WorkloadDefinition,
)
from eyvan.server.api.app import create_app
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import AuthorizationError, ConflictError
from eyvan.server.service.scheduling import CoreSchedulingService

ROOT = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _principal(role: Role, *, org: str = "acme", project: str | None = "research") -> Principal:
    if role is Role.FLEET_ADMIN:
        return Principal(subject="p", grants=(RoleGrant(role=role),))
    if role is Role.ORG_ADMIN:
        return Principal(subject="p", grants=(RoleGrant(role=role, org_id=org),))
    return Principal(subject="p", grants=(RoleGrant(role=role, org_id=org, project_id=project),))


def _service() -> ControlPlaneService:
    return ControlPlaneService(InMemoryRepository(), scheduler=CoreSchedulingService())


async def _registered(service: ControlPlaneService, *, org: str = "acme", project: str | None = "research") -> None:
    await service.register_organization(
        ROOT, OrganizationRecord(id=org, owner="alice", created_by="root"), idempotency_key=f"org-{org}"
    )
    if project is not None:
        await service.register_project(
            ROOT,
            ProjectRecord(org_id=org, id=project, owner="alice", created_by="alice"),
            idempotency_key=f"prj-{org}-{project}",
        )


def _spec(name: str = "train") -> WorkloadDefinition:
    return WorkloadDefinition(name=name, execution=CommandExecution(command="python train.py"))


# --------------------------------------------------------------------------- registration


async def test_a_registered_organization_can_be_listed() -> None:
    """The question nothing could answer: which tenants does this plane serve."""

    service = _service()
    await _registered(service)

    assert [organization.id for organization in await service.list_organizations(ROOT)] == ["acme"]


async def test_registration_is_idempotent_on_its_key() -> None:
    """Onboarding runs from configuration management, which retries."""

    service = _service()
    first, replayed_first = await service.register_organization(
        ROOT, OrganizationRecord(id="acme", owner="alice", created_by="root"), idempotency_key="o1"
    )
    second, replayed_second = await service.register_organization(
        ROOT, OrganizationRecord(id="acme", owner="alice", created_by="root"), idempotency_key="o1"
    )

    assert (replayed_first, replayed_second) == (False, True)
    assert first.created_at == second.created_at


async def test_a_replayed_registration_returns_the_organization_not_a_project() -> None:
    """The idempotency binding dispatches on record type, and both are org-scoped.

    A project carries an ``org_id`` too, so binding it before the organization
    check would file every project under its organization's id -- and a replay
    would then return the wrong record entirely.
    """

    service = _service()
    await _registered(service)

    project, replayed = await service.register_project(
        ROOT,
        ProjectRecord(org_id="acme", id="research", owner="alice", created_by="alice"),
        idempotency_key="prj-acme-research",
    )

    assert replayed is True
    assert isinstance(project, ProjectRecord)
    assert project.key == ("acme", "research")


async def test_a_project_cannot_be_registered_under_an_unregistered_organization() -> None:
    """Auto-creating the parent from a typo in the child is the original defect."""

    service = _service()

    with pytest.raises(ConflictError, match="is not registered"):
        await service.register_project(
            ROOT,
            ProjectRecord(org_id="ghost", id="research", owner="alice", created_by="alice"),
            idempotency_key="p1",
        )


async def test_two_organizations_may_each_hold_a_project_of_the_same_name() -> None:
    """Project names are unique within a tenant, and only within one.

    A global project id would make the second organization's ``research``
    project impossible to create, for a reason no user could see.
    """

    service = _service()
    await _registered(service, org="acme")
    await _registered(service, org="rival")

    assert [project.key for project in await service.list_projects(ROOT, org_id="acme")] == [("acme", "research")]
    assert [project.key for project in await service.list_projects(ROOT, org_id="rival")] == [("rival", "research")]


# --------------------------------------------------------------------------- authorization


async def test_only_a_fleet_admin_registers_an_organization() -> None:
    """Otherwise a tenant mints itself scopes nobody allocated.

    Those scopes then collect quota and budget rows, so self-registration is not
    a naming question -- it is a way to acquire governance state.
    """

    service = _service()

    for role in (
        Role.ORG_ADMIN,
        Role.PROJECT_ADMIN,
        Role.PROJECT_OPERATOR,
        Role.PROJECT_SUBMITTER,
        Role.PROJECT_VIEWER,
    ):
        with pytest.raises(AuthorizationError):
            await service.register_organization(
                _principal(role),
                OrganizationRecord(id="acme", owner="alice", created_by="x"),
                idempotency_key=f"o-{role.value}",
            )


async def test_an_org_admin_registers_projects_in_its_own_organization_only() -> None:
    """Project creation is an organization-level act, not a project-level one.

    A project admin holds ``MANAGE_PROJECT``, and creating a *sibling* project
    is not something their grant should reach.
    """

    service = _service()
    await _registered(service, org="acme", project=None)
    await _registered(service, org="rival", project=None)

    stored, _ = await service.register_project(
        _principal(Role.ORG_ADMIN, org="acme"),
        ProjectRecord(org_id="acme", id="platform", owner="bob", created_by="bob"),
        idempotency_key="p-own",
    )
    assert stored.key == ("acme", "platform")

    with pytest.raises(AuthorizationError):
        await service.register_project(
            _principal(Role.ORG_ADMIN, org="acme"),
            ProjectRecord(org_id="rival", id="platform", owner="bob", created_by="bob"),
            idempotency_key="p-other",
        )

    with pytest.raises(AuthorizationError):
        await service.register_project(
            _principal(Role.PROJECT_ADMIN, org="acme", project="research"),
            ProjectRecord(org_id="acme", id="sneaky", owner="bob", created_by="bob"),
            idempotency_key="p-project-admin",
        )


async def test_a_project_admin_cannot_re_register_its_own_project() -> None:
    """The case that distinguishes the two plausible gates.

    ``MANAGE_PROJECT`` would allow this -- a project admin holds it for exactly
    this project. But registration also sets ``owner`` and ``enabled``, so a
    project admin who could re-register would be able to reassign accountability
    for their own project and to re-enable one their organization disabled.
    Registration is an organization-level act; administering the project's
    contents is not.
    """

    service = _service()
    await _registered(service)

    with pytest.raises(AuthorizationError):
        await service.register_project(
            _principal(Role.PROJECT_ADMIN, org="acme", project="research"),
            ProjectRecord(org_id="acme", id="research", owner="someone-else", created_by="p"),
            idempotency_key="p-reassign",
        )


async def test_listing_organizations_does_not_leak_the_customer_list() -> None:
    """A viewer in one tenant asking "which tenants exist" must not learn the rest."""

    service = _service()
    await _registered(service, org="acme", project=None)
    await _registered(service, org="rival", project=None)

    visible = await service.list_organizations(_principal(Role.PROJECT_VIEWER, org="acme"))

    assert [organization.id for organization in visible] == ["acme"]
    assert len(await service.list_organizations(ROOT)) == 2


# --------------------------------------------------------------------------- enforcement


async def test_a_quota_for_a_misspelled_project_is_refused_once_the_org_is_registered() -> None:
    """The concrete defect: ``PUT /quotas`` with a typo silently succeeded.

    The quota then bound nothing, and the only symptom was a project running
    without the limit someone believed they had configured.
    """

    service = _service()
    await _registered(service)

    with pytest.raises(ConflictError, match="researchh"):
        await service.put_quota(
            ROOT,
            QuotaRecord(org_id="acme", project_id="researchh", limits=ResourceLimits(cpu=8), updated_by="root"),
            idempotency_key="q1",
        )


async def test_an_unregistered_organization_is_left_alone() -> None:
    """Upgrading must not break a plane that has registered nothing.

    Enforcement arrives with registration, so the pre-registry world keeps
    working exactly as before rather than needing a flag nobody would find.
    """

    service = _service()

    stored, _ = await service.put_quota(
        ROOT,
        QuotaRecord(org_id="unregistered", project_id="anything", limits=ResourceLimits(cpu=8), updated_by="root"),
        idempotency_key="q1",
    )

    assert stored.org_id == "unregistered"


async def test_a_workload_for_an_unregistered_project_is_refused() -> None:
    """Same rule at the busiest entry point, or the registry means nothing."""

    service = _service()
    await _registered(service)

    with pytest.raises(ConflictError, match="not registered"):
        await service.submit_workload(
            _principal(Role.PROJECT_OPERATOR, project="ghost"),
            org_id="acme",
            project_id="ghost",
            spec=_spec(),
            idempotency_key="w1",
        )


async def test_a_disabled_organization_accepts_no_new_work() -> None:
    """Disabling is how a tenant is offboarded without orphaning its history."""

    service = _service()
    await _registered(service)
    await service.register_organization(
        ROOT,
        OrganizationRecord(id="acme", owner="alice", created_by="root", enabled=False),
        idempotency_key="o-disable",
    )

    with pytest.raises(ConflictError, match="disabled"):
        await service.submit_workload(
            _principal(Role.PROJECT_OPERATOR),
            org_id="acme",
            project_id="research",
            spec=_spec(),
            idempotency_key="w1",
        )


async def test_a_disabled_project_stops_only_itself() -> None:
    """Offboarding one team must not stop the organization around it."""

    service = _service()
    await _registered(service)
    await service.register_project(
        ROOT,
        ProjectRecord(org_id="acme", id="platform", owner="bob", created_by="bob", enabled=False),
        idempotency_key="p-platform",
    )

    workload, _ = await service.submit_workload(
        _principal(Role.PROJECT_OPERATOR, project="research"),
        org_id="acme",
        project_id="research",
        spec=_spec(),
        idempotency_key="w-live",
    )
    assert workload.project_id == "research"

    with pytest.raises(ConflictError, match="disabled"):
        await service.submit_workload(
            _principal(Role.PROJECT_OPERATOR, project="platform"),
            org_id="acme",
            project_id="platform",
            spec=_spec(),
            idempotency_key="w-stopped",
        )


async def test_disabling_an_organization_does_not_strand_work_already_running() -> None:
    """Settlement must survive offboarding, or disabling a tenant leaks leases and money.

    This is why enforcement lives on the paths that accept *new* work rather
    than in the admission-scope lock every path takes.
    """

    service = _service()
    await _registered(service)
    workload, _ = await service.submit_workload(
        _principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=_spec(),
        idempotency_key="w1",
    )
    await service.register_organization(
        ROOT,
        OrganizationRecord(id="acme", owner="alice", created_by="root", enabled=False),
        idempotency_key="o-disable",
    )

    cancelled, _ = await service.cancel_workload(ROOT, workload.id, idempotency_key="cancel-1")

    assert cancelled.id == workload.id


# --------------------------------------------------------------------------- http surface


def _client() -> TestClient:
    return TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True, artifact_token_key="k" * 32)))


def _headers(role: str, **extra: str) -> dict[str, str]:
    return {"X-Eyvan-User": "root", "X-Eyvan-Roles": role, **extra}


def test_the_routes_register_and_list_over_http() -> None:
    """A service method with no route is reachable only from inside the process."""

    client = _client()
    admin = _headers("fleet_admin")

    created = client.put(
        "/api/v1/organizations/acme",
        json={"owner": "alice", "display_name": "Acme Research"},
        headers={**admin, "Idempotency-Key": "o1"},
    )
    project = client.put(
        "/api/v1/organizations/acme/projects/research",
        json={"owner": "alice"},
        headers={**admin, "Idempotency-Key": "p1"},
    )

    assert created.status_code == 200
    assert created.json()["created_by"] == "root"
    assert project.status_code == 200
    assert [item["id"] for item in client.get("/api/v1/organizations", headers=admin).json()] == ["acme"]
    assert [item["id"] for item in client.get("/api/v1/organizations/acme/projects", headers=admin).json()] == [
        "research"
    ]


def test_a_replayed_registration_is_labelled_on_the_response() -> None:
    """Configuration management retries, and a caller has to tell the difference."""

    client = _client()
    admin = _headers("fleet_admin")
    payload = {"owner": "alice"}

    first = client.put("/api/v1/organizations/acme", json=payload, headers={**admin, "Idempotency-Key": "o1"})
    second = client.put("/api/v1/organizations/acme", json=payload, headers={**admin, "Idempotency-Key": "o1"})

    assert "Idempotent-Replay" not in first.headers
    assert second.headers.get("Idempotent-Replay") == "true"


def test_the_registration_route_refuses_a_non_admin_over_http() -> None:
    """Authorization has to hold at the boundary, not only in the service."""

    client = _client()

    response = client.put(
        "/api/v1/organizations/acme",
        json={"owner": "alice"},
        headers={
            **_headers("operator", **{"X-Eyvan-Org": "acme", "X-Eyvan-Project": "research"}),
            "Idempotency-Key": "o1",
        },
    )

    assert response.status_code == 403


def test_an_unknown_organization_reads_as_not_found_rather_than_empty() -> None:
    """An empty answer would read as "registered with nothing in it"."""

    client = _client()

    assert client.get("/api/v1/organizations/ghost", headers=_headers("fleet_admin")).status_code == 404
