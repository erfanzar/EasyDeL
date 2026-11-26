# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Organization and project registration for the Eyvan CLI.

Onboarding a tenant used to mean configuring an identity provider to mint a
custom claim, and nothing anywhere recorded that the tenant existed. These
commands are the other half of the registry: a route with no CLI is reachable
only by writing the HTTP call by hand, which is how ``max_failure_retries``
spent its life.

Mutations route through ``OperationsCommandService.mutate`` for the same reason
the governance commands do -- these routes require an ``Idempotency-Key``, and
the shared service owns the one factory that produces them.
"""

from __future__ import annotations

from datetime import UTC, datetime

import click
from pydantic import ValidationError

from eyvan._internal.json_types import model_json
from eyvan.records import Role, RoleGrant
from eyvan.sdk.commands.common import emit_result
from eyvan.sdk.commands.operations import OperationsCommandService
from eyvan.sdk.commands.state import CliSession
from eyvan.server.api.schemas import (
    OrganizationRegisterRequest,
    ProjectRegisterRequest,
    RoleGrantIssueRequest,
)


def _service(session: CliSession) -> OperationsCommandService:
    return OperationsCommandService(session)


def _labels(pairs: tuple[str, ...]) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` options into a label map."""

    labels: dict[str, str] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator or not name.strip():
            raise click.ClickException(f"label {pair!r} must be KEY=VALUE")
        labels[name.strip()] = value.strip()
    return labels


@click.group("org")
def org() -> None:
    """Register and inspect the organizations this control plane serves."""


@org.command("list")
@click.pass_obj
def list_organizations(session: CliSession) -> None:
    """List organizations visible to this identity.

    A fleet administrator sees every tenant; anyone else sees only the ones they
    hold a grant for, so this cannot be used to read the customer list.
    """

    _service(session).read("/api/v1/organizations")


@org.command("get")
@click.argument("org_id")
@click.pass_obj
def get_organization(session: CliSession, org_id: str) -> None:
    """Show one registered organization."""

    _service(session).read(f"/api/v1/organizations/{org_id}")


@org.command("create")
@click.argument("org_id")
@click.option("--owner", required=True, help="Subject accountable for this tenant.")
@click.option("--display-name", default="", help="Human-readable name.")
@click.option("--label", "labels", multiple=True, metavar="KEY=VALUE")
@click.option(
    "--disabled",
    is_flag=True,
    help="Register without accepting work. Existing history stays readable either way.",
)
@click.option("--idempotency-key")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def create_organization(
    session: CliSession,
    org_id: str,
    owner: str,
    display_name: str,
    labels: tuple[str, ...],
    disabled: bool,
    idempotency_key: str | None,
    as_json: bool,
) -> None:
    """Register a tenant, or update the one already registered under this id."""

    request = OrganizationRegisterRequest(
        owner=owner,
        display_name=display_name,
        labels=_labels(labels),
        enabled=not disabled,
    )
    service = _service(session)
    response = session.client.request(
        "PUT",
        f"/api/v1/organizations/{org_id}",
        json=model_json(request),
        headers={"Idempotency-Key": service.keys.create("organization.register", idempotency_key)},
    )
    emit_result(service.renderer, response, f"Registered organization {org_id}", as_json=as_json)


@click.group("project")
def project() -> None:
    """Register and inspect projects inside an organization."""


@project.command("list")
@click.option("--org", "org_id")
@click.pass_obj
def list_projects(session: CliSession, org_id: str | None) -> None:
    """List the projects registered in one organization."""

    scope = session.workload_scope(org_id=org_id)
    _service(session).read(f"/api/v1/organizations/{scope.org_id}/projects")


@project.command("create")
@click.argument("project_id")
@click.option("--org", "org_id")
@click.option("--owner", required=True, help="Subject accountable for this project.")
@click.option("--display-name", default="", help="Human-readable name.")
@click.option("--label", "labels", multiple=True, metavar="KEY=VALUE")
@click.option("--disabled", is_flag=True, help="Register without accepting work.")
@click.option("--idempotency-key")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def create_project(
    session: CliSession,
    project_id: str,
    org_id: str | None,
    owner: str,
    display_name: str,
    labels: tuple[str, ...],
    disabled: bool,
    idempotency_key: str | None,
    as_json: bool,
) -> None:
    """Register a project, or update the one already registered under this id.

    The organization must already exist: creating it implicitly from a typo in
    the project would reproduce the very problem the registry removes.
    """

    scope = session.workload_scope(org_id=org_id)
    request = ProjectRegisterRequest(
        owner=owner,
        display_name=display_name,
        labels=_labels(labels),
        enabled=not disabled,
    )
    service = _service(session)
    response = session.client.request(
        "PUT",
        f"/api/v1/organizations/{scope.org_id}/projects/{project_id}",
        json=model_json(request),
        headers={"Idempotency-Key": service.keys.create("project.register", idempotency_key)},
    )
    emit_result(service.renderer, response, f"Registered project {scope.org_id}/{project_id}", as_json=as_json)


@click.group("grant")
def grant() -> None:
    """Issue and revoke durable role grants.

    Before these existed, authority arrived only as an ``eyvan_grants`` OIDC
    claim, so granting someone access meant an identity-provider change and
    revoking it meant another one.
    """


@grant.command("list")
@click.option("--subject", help="Show only grants held by this subject.")
@click.option("--org", "org_id", help="Show only grants scoped to this organization.")
@click.pass_obj
def list_grants(session: CliSession, subject: str | None, org_id: str | None) -> None:
    """List durable grants this identity is entitled to see."""

    params: dict[str, str] = {}
    if subject is not None:
        params["subject"] = subject
    if org_id is not None:
        params["org_id"] = org_id
    _service(session).read("/api/v1/grants", params=params)


@grant.command("add")
@click.argument("subject")
@click.option(
    "--role",
    required=True,
    type=click.Choice([role.value for role in Role]),
    help="Role to grant.",
)
@click.option("--org", "org_id", help="Organization scope. Omitted for fleet_admin.")
@click.option("--project", "project_id", help="Project scope, for project roles.")
@click.option(
    "--expires-at",
    type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]),
    help="Drop the grant after this instant. Omit for indefinite access.",
)
@click.option("--reason", default="", help="Why this was granted, recorded in the ledger.")
@click.option("--idempotency-key")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def add_grant(
    session: CliSession,
    subject: str,
    role: str,
    org_id: str | None,
    project_id: str | None,
    expires_at: datetime | None,
    reason: str,
    idempotency_key: str | None,
    as_json: bool,
) -> None:
    """Grant SUBJECT a role, effective on their next request."""

    try:
        request = RoleGrantIssueRequest(
            subject=subject,
            grant=RoleGrant(role=Role(role), org_id=org_id, project_id=project_id),
            expires_at=None if expires_at is None else expires_at.replace(tzinfo=UTC),
            reason=reason,
        )
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    service = _service(session)
    response = session.client.request(
        "POST",
        "/api/v1/grants",
        json=model_json(request),
        headers={"Idempotency-Key": service.keys.create("grant.issue", idempotency_key)},
    )
    scope = "/".join(part for part in (org_id, project_id) if part) or "every tenant"
    emit_result(service.renderer, response, f"Granted {subject} the {role} role on {scope}", as_json=as_json)


@grant.command("revoke")
@click.argument("subject")
@click.option("--role", required=True, type=click.Choice([role.value for role in Role]))
@click.option("--org", "org_id", default="")
@click.option("--project", "project_id", default="")
@click.pass_obj
def revoke_grant(session: CliSession, subject: str, role: str, org_id: str, project_id: str) -> None:
    """Withdraw a grant from SUBJECT, effective on their next request."""

    session.client.request(
        "DELETE",
        "/api/v1/grants",
        params={"subject": subject, "role": role, "org_id": org_id, "project_id": project_id},
    )
    click.echo(f"revoked {role} from {subject}")


def register_tenancy_commands(main: click.Group) -> None:
    """Attach the tenancy command groups to the CLI root."""

    for command in (org, project, grant):
        main.add_command(command)


__all__ = ("grant", "org", "project", "register_tenancy_commands")
