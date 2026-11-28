# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Authenticated effective-authority introspection for browser composition."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import Permission, Principal, Role, RoleGrant
from eyvan.server.api.app import create_app

ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}


def _headers(role: str, *, subject: str = "alice", org: str = "acme", project: str = "research") -> dict[str, str]:
    return {
        "X-Eyvan-User": subject,
        "X-Eyvan-Roles": role,
        "X-Eyvan-Org": org,
        "X-Eyvan-Project": project,
    }


def _client(*, verifier=None) -> TestClient:
    return TestClient(
        create_app(
            settings=EyvanSettings(dev_auth_enabled=True, artifact_token_key="k" * 32),
            verifier=verifier,
        )
    )


def _register_scope(client: TestClient, org: str, project: str) -> None:
    organization = client.put(
        f"/api/v1/organizations/{org}",
        json={"owner": f"owner@{org}"},
        headers={**ADMIN, "Idempotency-Key": f"org-{org}"},
    )
    assert organization.status_code in {200, 201}, organization.text
    registered = client.put(
        f"/api/v1/organizations/{org}/projects/{project}",
        json={"owner": f"owner@{org}", "display_name": project.title()},
        headers={**ADMIN, "Idempotency-Key": f"project-{org}-{project}"},
    )
    assert registered.status_code in {200, 201}, registered.text


@pytest.mark.parametrize(
    ("role", "expected_permissions"),
    [
        (Role.FLEET_ADMIN, tuple(Permission)),
        (
            Role.ORG_ADMIN,
            (
                Permission.READ,
                Permission.SUBMIT,
                Permission.OPERATE,
                Permission.MANAGE_PROJECT,
                Permission.MANAGE_OWN_CAPACITY,
            ),
        ),
        (
            Role.PROJECT_ADMIN,
            (Permission.READ, Permission.SUBMIT, Permission.OPERATE, Permission.MANAGE_PROJECT),
        ),
        (Role.PROJECT_OPERATOR, (Permission.READ, Permission.SUBMIT, Permission.OPERATE)),
        (Role.PROJECT_SUBMITTER, (Permission.SUBMIT,)),
        (Role.PROJECT_VIEWER, (Permission.READ,)),
    ],
    ids=lambda value: value.value if isinstance(value, Role) else None,
)
def test_session_reports_the_enforced_permissions_for_every_dashboard_role(
    role: Role,
    expected_permissions: tuple[Permission, ...],
) -> None:
    client = _client()
    _register_scope(client, "acme", "research")

    response = client.get(
        "/api/v1/session",
        params={"org_id": "acme", "project_id": "research"},
        headers=_headers(role.value),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    scope = next(item for item in body["scopes"] if (item["org_id"], item["project_id"]) == ("acme", "research"))
    assert scope["roles"] == [role.value]
    assert scope["permissions"] == [permission.value for permission in expected_permissions]
    assert body["global_permissions"] == (
        [permission.value for permission in Permission] if role is Role.FLEET_ADMIN else []
    )


def test_session_unions_claim_and_durable_grants_before_reporting_authority() -> None:
    client = _client()
    _register_scope(client, "acme", "research")
    issued = client.post(
        "/api/v1/grants",
        json={
            "subject": "alice",
            "grant": {"role": "project_submitter", "org_id": "acme", "project_id": "research"},
        },
        headers={**ADMIN, "Idempotency-Key": "submitter-grant"},
    )
    assert issued.status_code == 201, issued.text

    response = client.get(
        "/api/v1/session",
        params={"org_id": "acme", "project_id": "research"},
        headers=_headers("viewer"),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert {grant["role"] for grant in body["grants"]} == {"project_viewer", "project_submitter"}
    assert body["scopes"] == [
        {
            "org_id": "acme",
            "project_id": "research",
            "display_name": "Research",
            "registered": True,
            "enabled": True,
            "accepts_new_work": True,
            "roles": ["project_submitter", "project_viewer"],
            "permissions": ["read", "submit"],
        }
    ]


def test_scope_inventory_expands_org_and_fleet_grants_without_leaking_other_tenants() -> None:
    client = _client()
    for org, project in (("acme", "research"), ("acme", "platform"), ("rival", "research")):
        _register_scope(client, org, project)

    project = client.get("/api/v1/session", headers=_headers("viewer")).json()
    organization = client.get("/api/v1/session", headers=_headers("org_admin")).json()
    fleet = client.get("/api/v1/session", headers=_headers("fleet_admin")).json()

    assert {(scope["org_id"], scope["project_id"]) for scope in project["scopes"]} == {("acme", "research")}
    assert {(scope["org_id"], scope["project_id"]) for scope in organization["scopes"]} == {
        ("acme", "platform"),
        ("acme", "research"),
    }
    assert {(scope["org_id"], scope["project_id"]) for scope in fleet["scopes"]} == {
        ("acme", "platform"),
        ("acme", "research"),
        ("rival", "research"),
    }


def test_preferred_scope_is_an_authorized_hint_not_a_way_to_mint_authority() -> None:
    client = _client()

    response = client.get(
        "/api/v1/session",
        params={"org_id": "rival", "project_id": "secret"},
        headers=_headers("viewer"),
    )

    assert response.status_code == 200
    assert {(scope["org_id"], scope["project_id"]) for scope in response.json()["scopes"]} == {("acme", "research")}


def test_project_submitter_can_enqueue_but_cannot_read_or_operate_workloads() -> None:
    client = _client()
    headers = _headers("submitter")
    submission = client.post(
        "/api/v1/workloads",
        json={
            "org_id": "acme",
            "project_id": "research",
            "spec": {"name": "submit-only", "execution": {"kind": "command", "command": "echo ok"}},
        },
        headers={**headers, "Idempotency-Key": "submit-only"},
    )

    assert submission.status_code == 201, submission.text
    workload_id = submission.json()["id"]
    assert (
        client.get("/api/v1/workloads", params={"org_id": "acme", "project_id": "research"}, headers=headers).status_code
        == 403
    )
    assert (
        client.post(
            f"/api/v1/workloads/{workload_id}/cancel",
            headers={**headers, "Idempotency-Key": "cannot-cancel"},
        ).status_code
        == 403
    )


def test_session_requires_authentication() -> None:
    assert _client().get("/api/v1/session").status_code == 401


class _ViewerVerifier:
    async def verify(self, token: str) -> Principal:
        assert token == "verified"
        return Principal(
            subject="bearer-user",
            grants=(RoleGrant(role=Role.PROJECT_VIEWER, org_id="acme", project_id="research"),),
        )


def test_bearer_identity_takes_precedence_over_development_admin_headers() -> None:
    client = _client(verifier=_ViewerVerifier())

    response = client.get(
        "/api/v1/session",
        params={"org_id": "acme", "project_id": "research"},
        headers={**_headers("fleet_admin"), "Authorization": "Bearer verified"},
    )

    assert response.status_code == 200
    assert response.json()["subject"] == "bearer-user"
    assert response.json()["global_permissions"] == []
    assert response.json()["scopes"][0]["permissions"] == ["read"]


def test_registered_org_missing_project_is_visible_but_accepts_no_mutations() -> None:
    client = _client()
    _register_scope(client, "acme", "research")

    response = client.get(
        "/api/v1/session",
        params={"org_id": "acme", "project_id": "missing"},
        headers=_headers("operator", project="missing"),
    )

    assert response.status_code == 200, response.text
    scope = next(item for item in response.json()["scopes"] if item["project_id"] == "missing")
    assert scope["registered"] is False
    assert scope["enabled"] is True
    assert scope["accepts_new_work"] is False
    assert {"read", "submit", "operate"} <= set(scope["permissions"])


def test_disabled_registered_scope_remains_readable_and_selectable_but_history_only() -> None:
    client = _client()
    _register_scope(client, "acme", "research")
    disabled = client.put(
        "/api/v1/organizations/acme/projects/research",
        json={"owner": "owner@acme", "display_name": "Research", "enabled": False},
        headers={**ADMIN, "Idempotency-Key": "disable-research"},
    )
    assert disabled.status_code == 200, disabled.text

    response = client.get(
        "/api/v1/session",
        params={"org_id": "acme", "project_id": "research"},
        headers=_headers("operator"),
    )

    scope = response.json()["scopes"][0]
    assert scope["registered"] is True
    assert scope["enabled"] is False
    assert scope["accepts_new_work"] is False
    assert "read" in scope["permissions"]


def test_pre_registry_scope_keeps_legacy_mutation_compatibility() -> None:
    response = _client().get(
        "/api/v1/session",
        params={"org_id": "legacy", "project_id": "anything"},
        headers=_headers("operator", org="legacy", project="anything"),
    )

    assert response.status_code == 200, response.text
    scope = response.json()["scopes"][0]
    assert scope["registered"] is False
    assert scope["accepts_new_work"] is True


def test_scoped_session_query_materializes_org_wildcard_only_inside_its_org() -> None:
    client = _client()

    same_org = client.get(
        "/api/v1/session",
        params={"org_id": "acme", "project_id": "other"},
        headers=_headers("org_admin"),
    )
    rival_org = client.get(
        "/api/v1/session",
        params={"org_id": "rival", "project_id": "other"},
        headers=_headers("org_admin"),
    )

    assert same_org.status_code == 200
    same_scope = next(scope for scope in same_org.json()["scopes"] if scope["project_id"] == "other")
    assert same_scope["org_id"] == "acme"
    assert "read" in same_scope["permissions"]
    assert same_scope["accepts_new_work"] is True
    assert not any((scope["org_id"], scope["project_id"]) == ("rival", "other") for scope in rival_org.json()["scopes"])
