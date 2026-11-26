# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Peer planes and budget grants for the Eyvan CLI.

Federated budgets were built, tested, wired into admission -- and unreachable.
There was no route and no command, so the only way to cut a peer a slice of a
tenant's budget was to write the HTTP call by hand against a service method the
API never exposed. A mechanism nobody can invoke bounds nothing.

The commands are deliberately blunt about money. ``grant issue`` prints the
grant it created including the amount and the expiry, ``grant list`` hides
expired slices unless asked, and ``grant revoke`` returns the record rather than
a bare success, because ``reported_spend`` on that record is the difference
between "this peer can no longer spend" and "this peer never spent".
"""

from __future__ import annotations

from datetime import UTC, datetime

import click
from pydantic import ValidationError

from eyvan._internal.json_types import model_json
from eyvan.records import ResourceLimits
from eyvan.sdk.commands.common import emit_result
from eyvan.sdk.commands.operations import OperationsCommandService
from eyvan.sdk.commands.state import CliSession
from eyvan.server.api.schemas import (
    BudgetGrantIssueRequest,
    PlanePeerPutRequest,
    QuotaGrantIssueRequest,
)


def _service(session: CliSession) -> OperationsCommandService:
    return OperationsCommandService(session)


@click.group("federation")
def federation() -> None:
    """Administer peer control planes and the budget slices issued to them.

    One Eyvan per region is already correct for scheduling -- placement is
    regional, so a plane that only knows its own cells is not missing anything
    it could act on. Budgets are where that breaks: a tenant with projects on
    three planes otherwise has three independent limits, each locally compliant
    while the tenant spends three times what its operator set.
    """


@federation.group("peer")
def peer() -> None:
    """Register the control planes this one may grant budget to."""


@peer.command("list")
@click.pass_obj
def list_peers(session: CliSession) -> None:
    """List the control planes this one knows about."""

    _service(session).read("/api/v1/federation/peers")


@peer.command("add")
@click.argument("plane_id")
@click.option("--endpoint", required=True, help="The peer's base URL, http(s).")
@click.option("--description", default="", help="Free text, for whoever reads the peer list next.")
@click.option(
    "--disabled",
    is_flag=True,
    help="Register without allowing new grants. Existing grants are unaffected; revoke those explicitly.",
)
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def add_peer(
    session: CliSession,
    plane_id: str,
    endpoint: str,
    description: str,
    disabled: bool,
    as_json: bool,
) -> None:
    """Register PLANE_ID as a peer, or update the registration.

    Peers are written down, never discovered. A plane that could join by
    asserting it is a tenant's home plane could claim that tenant's entire
    budget, and nothing in this design has the standing to refuse the claim.
    """

    try:
        request = PlanePeerPutRequest(endpoint=endpoint, description=description, enabled=not disabled)
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    service = _service(session)
    response = session.client.request("PUT", f"/api/v1/federation/peers/{plane_id}", json=model_json(request))
    emit_result(service.renderer, response, f"Registered peer plane {plane_id}", as_json=as_json)


@federation.group("grant")
def grant() -> None:
    """Issue, list and revoke slices of a tenant's budget."""


@grant.command("list")
@click.option("--org", "org_id", help="Show only grants for this organization.")
@click.option("--project", "project_id", help="Show only grants for this project.")
@click.option(
    "--include-inactive",
    is_flag=True,
    help="Include revoked and expired slices. Off by default; on when asking why a grantee stopped admitting.",
)
@click.pass_obj
def list_grants(session: CliSession, org_id: str | None, project_id: str | None, include_inactive: bool) -> None:
    """List the budget slices this plane has issued or been issued."""

    params: dict[str, str] = {}
    if org_id is not None:
        params["org_id"] = org_id
    if project_id is not None:
        params["project_id"] = project_id
    if include_inactive:
        params["include_inactive"] = "true"
    _service(session).read("/api/v1/federation/budget-grants", params=params)


@grant.command("issue")
@click.argument("grantee_plane_id")
@click.option("--org", "org_id", help="Organization whose budget is being sliced.")
@click.option("--project", "project_id", help="Project whose budget is being sliced.")
@click.option("--amount", type=float, required=True, help="How much of the limit the peer may spend.")
@click.option(
    "--expires-at",
    type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]),
    required=True,
    help="When the slice lapses. Required: a grant that cannot lapse makes a home plane going away permanent spend.",
)
@click.option("--idempotency-key")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def issue_grant(
    session: CliSession,
    grantee_plane_id: str,
    org_id: str | None,
    project_id: str | None,
    amount: float,
    expires_at: datetime,
    idempotency_key: str | None,
    as_json: bool,
) -> None:
    """Grant GRANTEE_PLANE_ID a slice of one tenant's budget.

    Re-issuing to a peer that already holds a slice *replaces* it rather than
    adding to it, so a peer's authority is always one number an operator can
    read. The home plane refuses to issue past its own limit, counting both what
    it has already granted away and what it has itself spent.
    """

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    try:
        request = BudgetGrantIssueRequest(
            org_id=scope.org_id,
            project_id=scope.project_id,
            grantee_plane_id=grantee_plane_id,
            amount=amount,
            expires_at=expires_at.replace(tzinfo=UTC) if expires_at.tzinfo is None else expires_at,
        )
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    service = _service(session)
    response = session.client.request(
        "POST",
        "/api/v1/federation/budget-grants",
        json=model_json(request),
        headers={"Idempotency-Key": service.keys.create("budget.grant", idempotency_key)},
    )
    emit_result(service.renderer, response, "Issued a budget grant", as_json=as_json)


@grant.command("revoke")
@click.argument("grant_id")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def revoke_grant(
    session: CliSession,
    grant_id: str,
    as_json: bool,
) -> None:
    """Withdraw GRANT_ID, stopping new admission on the grantee.

    Says nothing about what already ran: that money is spent, and clawing it
    back would make the home plane's ledger disagree with the invoice.
    """

    service = _service(session)
    response = session.client.request("DELETE", f"/api/v1/federation/budget-grants/{grant_id}")
    emit_result(service.renderer, response, f"Revoked budget grant {grant_id}", as_json=as_json)


@federation.group("quota-grant")
def quota_grant() -> None:
    """Issue, list and revoke slices of a tenant's resource quota.

    The capacity half of `federation grant`. Money and chips are separate limits
    and a fleet can hold one while breaching the other: a tenant on a small
    budget can still exhaust an accelerator pool everyone else is queued behind,
    because what is scarce is chips.
    """


@quota_grant.command("list")
@click.option("--org", "org_id", help="Show only grants for this organization.")
@click.option("--project", "project_id", help="Show only grants for this project.")
@click.option("--include-inactive", is_flag=True, help="Include revoked and expired slices.")
@click.pass_obj
def list_quota_grants(session: CliSession, org_id: str | None, project_id: str | None, include_inactive: bool) -> None:
    """List the quota slices this plane has issued or been issued."""

    params: dict[str, str] = {}
    if org_id is not None:
        params["org_id"] = org_id
    if project_id is not None:
        params["project_id"] = project_id
    if include_inactive:
        params["include_inactive"] = "true"
    _service(session).read("/api/v1/federation/quota-grants", params=params)


@quota_grant.command("issue")
@click.argument("grantee_plane_id")
@click.option("--org", "org_id", help="Organization whose quota is being sliced.")
@click.option("--project", "project_id", help="Project whose quota is being sliced.")
@click.option("--cpu", type=float, default=0.0, help="CPU cores the peer may hold concurrently.")
@click.option("--memory-gib", type=float, default=0.0, help="Memory the peer may hold concurrently.")
@click.option("--gpu", type=float, default=0.0, help="GPUs the peer may hold concurrently.")
@click.option("--tpu", type=float, default=0.0, help="TPU chips the peer may hold concurrently.")
@click.option(
    "--expires-at",
    type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]),
    required=True,
    help="When the slice lapses. Required, for the same reason a budget slice's expiry is.",
)
@click.option("--idempotency-key")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def issue_quota_grant(
    session: CliSession,
    grantee_plane_id: str,
    org_id: str | None,
    project_id: str | None,
    cpu: float,
    memory_gib: float,
    gpu: float,
    tpu: float,
    expires_at: datetime,
    idempotency_key: str | None,
    as_json: bool,
) -> None:
    """Grant GRANTEE_PLANE_ID a slice of one tenant's quota.

    Dimensions left unset are zero rather than unlimited. The home plane refuses
    if the slice overruns its remaining headroom in *any* dimension, and names
    every dimension that overran.
    """

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    try:
        request = QuotaGrantIssueRequest(
            org_id=scope.org_id,
            project_id=scope.project_id,
            grantee_plane_id=grantee_plane_id,
            limits=ResourceLimits(cpu=cpu, memory_gib=memory_gib, gpu=gpu, tpu=tpu),
            expires_at=expires_at.replace(tzinfo=UTC) if expires_at.tzinfo is None else expires_at,
        )
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    service = _service(session)
    response = session.client.request(
        "POST",
        "/api/v1/federation/quota-grants",
        json=model_json(request),
        headers={"Idempotency-Key": service.keys.create("quota.grant", idempotency_key)},
    )
    emit_result(service.renderer, response, "Issued a quota grant", as_json=as_json)


@quota_grant.command("revoke")
@click.argument("grant_id")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def revoke_quota_grant(
    session: CliSession,
    grant_id: str,
    as_json: bool,
) -> None:
    """Withdraw GRANT_ID, stopping new admission on the grantee.

    Does not disturb what is already running: a quota is a concurrent ceiling,
    and evicting live work to enforce a ceiling change would make revocation an
    outage rather than an administrative act.
    """

    service = _service(session)
    response = session.client.request("DELETE", f"/api/v1/federation/quota-grants/{grant_id}")
    emit_result(service.renderer, response, f"Revoked quota grant {grant_id}", as_json=as_json)


def register_federation_commands(main: click.Group) -> None:
    """Attach the federation command group to the CLI root."""

    main.add_command(federation)


__all__ = ("federation", "register_federation_commands")
