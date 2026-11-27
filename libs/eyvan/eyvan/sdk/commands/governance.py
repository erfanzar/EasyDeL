# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Quota, budget, and policy commands for the Eyvan CLI.

Ten governance routes shipped with no CLI at all, which is a real gap rather than
a cosmetic one: ``eyvan doctor`` and ``eyvan workload explain`` exist to answer
"why is my job stuck", and the top three answers are quota, budget, and policy --
none of which an operator could read, let alone change, without writing the HTTP
call by hand.

Every mutation goes through ``OperationsCommandService.mutate`` rather than
calling the client directly. That is not stylistic: these routes *require* an
``Idempotency-Key``, and routing through the shared service reuses the one
``IdempotencyKeyFactory`` instead of leaving each command to remember.
"""

from __future__ import annotations

import json
import typing as tp

import click
from pydantic import ValidationError

from eyvan._internal.json_types import JsonValue, is_json_object, model_json
from eyvan.records import CapacityClass, CrossRegionTransferPolicy, EffectiveQuota, ResourceLimits
from eyvan.sdk.commands.common import JsonRenderer, emit_result
from eyvan.sdk.commands.operations import OperationsCommandService
from eyvan.sdk.commands.state import CliSession
from eyvan.sdk.terminal import stream_is_terminal
from eyvan.server.api.schemas import BudgetPutRequest, PolicyPutRequest, QuotaPutRequest


def _service(session: CliSession) -> OperationsCommandService:
    return OperationsCommandService(session)


def _put(
    session: CliSession,
    path: str,
    request: BudgetPutRequest | PolicyPutRequest | QuotaPutRequest,
    *,
    operation: str,
    idempotency_key: str | None,
) -> None:
    """Send one governance PUT through the shared idempotency factory.

    ``exclude_unset`` is load-bearing for quotas: the server infers
    ``constrained_resources`` from the limits' ``model_fields_set``, so a
    dimension the caller left unset must also be absent from the wire --
    serializing the pydantic defaults would constrain it after all.
    """

    service = _service(session)
    service.renderer.emit(
        session.client.request(
            "PUT",
            path,
            json=model_json(request, exclude_unset=True),
            headers={"Idempotency-Key": service.keys.create(operation, idempotency_key)},
        )
    )


@click.group("quota")
def quota() -> None:
    """Read and set hierarchical resource quotas."""


@quota.command("list")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.pass_obj
def list_quotas(session: CliSession, org_id: str | None, project_id: str | None) -> None:
    """List quota records in a scope."""

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    params: dict[str, str | int] = {"org_id": scope.org_id}
    if project_id is not None:
        params["project_id"] = project_id
    _service(session).read("/api/v1/quotas", params=params)


@quota.command("get")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.pass_obj
def get_quota(session: CliSession, org_id: str | None, project_id: str | None) -> None:
    """Show the quota that currently applies to this scope.

    The *effective* quota, not a stored row: the hierarchy resolves org, project
    and principal scopes, and knowing which one is binding is the whole question.
    """

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    payload = session.client.request(
        "GET",
        "/api/v1/quotas/current",
        params={"org_id": scope.org_id, "project_id": scope.project_id},
    )
    _emit_effective_quota(payload)
    JsonRenderer().emit(payload)


def _announce_quota_scope(org_id: str, project_id: str | None, principal_id: str | None) -> None:
    """Say which scope a quota is about to be written at.

    A quota widens what it is written to, so the failure that matters is
    setting one broader than intended -- and that is exactly what omitting
    ``--project`` on this subcommand does, whether or not the session has one.
    """

    stream = click.get_text_stream("stderr")
    if not stream_is_terminal(stream):
        return
    if principal_id is not None:
        where = f"principal {principal_id} in {org_id}/{project_id}"
    elif project_id is not None:
        where = f"project {org_id}/{project_id}"
    else:
        where = f"the whole {org_id} organization; pass --project to narrow it"
    click.echo(f"Eyvan: setting the quota for {where}.", file=stream)


def _emit_effective_quota(payload: JsonValue) -> None:
    """Write a one-line human reading to stderr, leaving JSON stdout unchanged.

    "No quota is set" is the ordinary state of a fresh fleet and the most likely
    answer to this question, and it is invisible in a document whose every
    ``quota`` field is ``null``. An empty answer that is never spelled out reads
    as a broken command.
    """

    stream = click.get_text_stream("stderr")
    if not stream_is_terminal(stream) or not is_json_object(payload):
        return
    try:
        effective = EffectiveQuota.model_validate(payload)
    except ValidationError:
        return
    if effective.unlimited:
        where = effective.project_id or effective.org_id
        click.echo(f"Eyvan: no quota is set for {where}; nothing here limits what it may run.", file=stream)
        return
    click.echo("Eyvan: every scope below applies; the tightest one binds first.", file=stream)
    for entry in effective.scopes:
        if entry.quota is None:
            click.echo(f"  {entry.scope:<10} no limit declared", file=stream)
            continue
        limits = ", ".join(f"{name}={value:g}" for name, value in sorted(entry.quota.limits.as_dict().items()) if value)
        click.echo(f"  {entry.scope:<10} {limits or 'zero on every resource'}", file=stream)


@quota.command("set")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--principal", "principal_id", help="Narrow the quota to one principal.")
@click.option("--cpu", type=float, help="CPU cores the scope may use; omit to leave it unconstrained.")
@click.option("--memory-gib", type=float, help="GiB of memory the scope may use; omit to leave it unconstrained.")
@click.option("--gpu", type=int, help="GPUs the scope may use; omit to leave it unconstrained.")
@click.option("--tpu", type=int, help="TPU chips the scope may use; omit to leave it unconstrained.")
@click.option("--artifact-bytes", type=int, help="Ceiling on stored artifact bytes; omit for unlimited.")
@click.option("--weight", type=float, default=1.0, show_default=True, help="Fair-share weight.")
@click.option("--idempotency-key")
@click.pass_obj
def set_quota(
    session: CliSession,
    org_id: str | None,
    project_id: str | None,
    principal_id: str | None,
    cpu: float | None,
    memory_gib: float | None,
    gpu: int | None,
    tpu: int | None,
    artifact_bytes: int | None,
    weight: float,
    idempotency_key: str | None,
) -> None:
    """Set the quota for one scope.

    Only the dimensions named on the command line become constraints; an
    unnamed dimension is left open, not set to zero. ``--cpu 0`` is how a
    dimension is explicitly closed.
    """

    # The org is resolved from the session; the project is *not*, and must not
    # be -- omitting --project is how an org-wide quota is written, so filling
    # it in from the session would make that unwritable from any normal
    # session. The asymmetry is deliberate and invisible, which is the danger:
    # `eyvan --project research quota set --cpu 64` writes an org quota, wider
    # than the person almost certainly meant. So the scope is stated before it
    # is written.
    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    _announce_quota_scope(scope.org_id, project_id, principal_id)
    # Built only from the flags that were passed. The server reads
    # ``constrained_resources`` off the limits' ``model_fields_set`` -- a
    # dimension absent there is unconstrained -- so constructing this with the
    # CLI defaults bound every unnamed dimension to zero and refused every
    # workload in the scope that asked for any of it.
    limits = ResourceLimits(
        **{
            name: value
            for name, value in (("cpu", cpu), ("memory_gib", memory_gib), ("gpu", gpu), ("tpu", tpu))
            if value is not None
        }
    )
    try:
        request = QuotaPutRequest(
            org_id=scope.org_id,
            project_id=project_id,
            principal_id=principal_id,
            limits=limits,
            artifact_bytes_limit=artifact_bytes,
            weight=weight,
        )
    except (ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _put(session, "/api/v1/quotas", request, operation="quota", idempotency_key=idempotency_key)


@quota.command("delete")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--principal", "principal_id")
@click.pass_obj
def delete_quota(session: CliSession, org_id: str | None, project_id: str | None, principal_id: str | None) -> None:
    """Remove a quota record, restoring the next-broadest scope."""

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    params: dict[str, str | int] = {"org_id": scope.org_id}
    if project_id is not None:
        params["project_id"] = project_id
    if principal_id is not None:
        params["principal_id"] = principal_id
    session.client.request("DELETE", "/api/v1/quotas", params=params)
    click.echo("deleted")


@click.group("budget")
def budget() -> None:
    """Read and set monetary admission budgets."""


@budget.command("list")
@click.option("--org", "org_id")
@click.pass_obj
def list_budgets(session: CliSession, org_id: str | None) -> None:
    """List budgets in an organization."""

    scope = session.workload_scope(org_id=org_id)
    _service(session).read("/api/v1/budgets", params={"org_id": scope.org_id})


@budget.command("get")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.pass_obj
def get_budget(session: CliSession, org_id: str | None, project_id: str | None) -> None:
    """Show one budget record."""

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    _service(session).read(f"/api/v1/budgets/{scope.org_id}/{scope.project_id}")


@budget.command("status")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.pass_obj
def budget_status(session: CliSession, org_id: str | None, project_id: str | None) -> None:
    """Show a budget's period window and projected end-of-period spend.

    The forecast assumes every active lease runs to the end of the window --
    deliberately pessimistic, because the number exists to warn before a hard
    limit starts refusing admission.
    """

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    _service(session).read(f"/api/v1/budgets/{scope.org_id}/{scope.project_id}/status")


@budget.command("roll")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def roll_budget(session: CliSession, org_id: str | None, project_id: str | None, as_json: bool) -> None:
    """Close an elapsed budget period without waiting for the next settlement.

    Not a force-reset: a period that has not elapsed is left alone, because
    zeroing it would erase spend the limit is still meant to bind.
    """

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    response = session.client.request("POST", f"/api/v1/budgets/{scope.org_id}/{scope.project_id}/roll")
    emit_result(
        JsonRenderer(), response, f"Rolled the budget period for {scope.org_id}/{scope.project_id}", as_json=as_json
    )


@budget.command("set")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--limit", type=float, required=True)
@click.option("--period", default="monthly", show_default=True)
@click.option(
    "--alert-at",
    "alert_thresholds",
    multiple=True,
    type=float,
    help="Fraction of the limit that raises an alert; repeatable (e.g. --alert-at 0.8).",
)
@click.option("--soft", is_flag=True, help="Report overspend instead of refusing admission.")
@click.option("--idempotency-key")
@click.pass_obj
def set_budget(
    session: CliSession,
    org_id: str | None,
    project_id: str | None,
    limit: float,
    period: str,
    alert_thresholds: tuple[float, ...],
    soft: bool,
    idempotency_key: str | None,
) -> None:
    """Set a project's budget."""

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    try:
        request = BudgetPutRequest(
            limit=limit,
            period=period,
            alert_thresholds=tuple(sorted(set(alert_thresholds))),
            hard_limit=not soft,
        )
    except (ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _put(
        session,
        f"/api/v1/budgets/{scope.org_id}/{scope.project_id}",
        request,
        operation="budget",
        idempotency_key=idempotency_key,
    )


@click.group("policy")
def policy() -> None:
    """Read and set project scheduling policy."""


@policy.command("list")
@click.option("--org", "org_id")
@click.pass_obj
def list_policies(session: CliSession, org_id: str | None) -> None:
    """List policies in an organization."""

    scope = session.workload_scope(org_id=org_id)
    _service(session).read("/api/v1/policies", params={"org_id": scope.org_id})


@policy.command("get")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.pass_obj
def get_policy(session: CliSession, org_id: str | None, project_id: str | None) -> None:
    """Show one project policy."""

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    _service(session).read(f"/api/v1/policies/{scope.org_id}/{scope.project_id}")


@policy.command("set")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--allow-provider", "allowed_providers", multiple=True)
@click.option("--allow-region", "allowed_regions", multiple=True)
@click.option(
    "--allow-capacity-class",
    "allowed_capacity_classes",
    multiple=True,
    type=click.Choice(tuple(item.value for item in CapacityClass)),
)
@click.option("--deny-accelerator-type", "denied_accelerator_types", multiple=True)
@click.option(
    "--allow-secret-reference",
    "allowed_secret_references",
    multiple=True,
    metavar="ROOT",
    help=(
        "Scheme/authority/path-segment roots this project's workloads may mount with --secret. Unlike every other "
        "allow-list here, omitting this permits NOTHING: naming a secret grants the reader the control plane's own "
        "identity, so the safe default is closed. Repeat to allow several roots."
    ),
)
@click.option("--max-priority", type=int, default=1000, show_default=True)
@click.option(
    "--no-preemption",
    is_flag=True,
    help="Refuse to preempt this project's workloads for higher-priority work.",
)
@click.option(
    "--cross-region-transfer",
    type=click.Choice(tuple(item.value for item in CrossRegionTransferPolicy)),
    default=None,
    help="Policy for cross-region GCS transfers from compute hosts (allow, warn, or ban).",
)
@click.option(
    "--ban-cross-region-transfer",
    is_flag=True,
    default=False,
    help="Shorthand for --cross-region-transfer ban.",
)
@click.option(
    "--allow-cross-region-bucket",
    "cross_region_allowed_buckets",
    multiple=True,
    metavar="BUCKET",
    help="GCS bucket name exempt from cross-region transfer checks (repeatable).",
)
@click.option(
    "--strict-same-region",
    is_flag=True,
    default=False,
    help="Disallow continental multi-region buckets (sets cross_region_allow_multiregion=false).",
)
@click.option("--idempotency-key")
@click.pass_obj
def set_policy(
    session: CliSession,
    org_id: str | None,
    project_id: str | None,
    allowed_providers: tuple[str, ...],
    allowed_regions: tuple[str, ...],
    allowed_capacity_classes: tuple[str, ...],
    denied_accelerator_types: tuple[str, ...],
    allowed_secret_references: tuple[str, ...],
    max_priority: int,
    no_preemption: bool,
    cross_region_transfer: str | None,
    ban_cross_region_transfer: bool,
    cross_region_allowed_buckets: tuple[str, ...],
    strict_same_region: bool,
    idempotency_key: str | None,
) -> None:
    """Set a project's scheduling policy.

    An empty allow-list means "no restriction", not "allow nothing" -- which is
    why these options are additive and omitting one leaves that dimension open.

    ``--allow-secret-reference`` is the one exception, and it is deliberate:
    there the empty list permits nothing. Every other dimension bounds *where*
    work may run; that one bounds which credentials a submitter may borrow from
    the control plane's own identity, and a permissive default there would be a
    privilege escalation available to anyone who can write a spec.
    """

    if (
        ban_cross_region_transfer
        and cross_region_transfer is not None
        and cross_region_transfer != CrossRegionTransferPolicy.BAN.value
    ):
        raise click.UsageError(
            f"--ban-cross-region-transfer conflicts with --cross-region-transfer {cross_region_transfer}"
        )
    resolved_cross_region_policy = (
        CrossRegionTransferPolicy.BAN
        if ban_cross_region_transfer
        else CrossRegionTransferPolicy(cross_region_transfer)
        if cross_region_transfer is not None
        else CrossRegionTransferPolicy.ALLOW
    )

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    try:
        request = PolicyPutRequest(
            allowed_providers=allowed_providers,
            allowed_regions=allowed_regions,
            allowed_capacity_classes=tp.cast(tp.Any, allowed_capacity_classes),
            denied_accelerator_types=denied_accelerator_types,
            allowed_secret_references=allowed_secret_references,
            max_priority=max_priority,
            preemption_enabled=not no_preemption,
            cross_region_transfer_policy=resolved_cross_region_policy,
            cross_region_allowed_buckets=cross_region_allowed_buckets,
            cross_region_allow_multiregion=not strict_same_region,
        )
    except (ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _put(
        session,
        f"/api/v1/policies/{scope.org_id}/{scope.project_id}",
        request,
        operation="policy",
        idempotency_key=idempotency_key,
    )


@click.command("system-reconcile")
@click.pass_obj
def system_reconcile(session: CliSession) -> None:
    """Run one scheduling reconcile pass on the leader.

    Returns 409 when this replica does not hold leadership, which is an ordinary
    answer rather than an error: exactly one replica reconciles at a time.
    """

    payload = session.client.request("POST", "/api/v1/system/reconcile")
    click.echo(json.dumps(payload, sort_keys=True, indent=2, default=str))


@click.group("webhook")
def webhook() -> None:
    """Push this project's lifecycle events somewhere, instead of polling for them."""


@webhook.command("list")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.pass_obj
def list_webhooks(session: CliSession, org_id: str | None, project_id: str | None) -> None:
    """List this project's subscriptions.

    Signing keys are not returned. They are what a receiver verifies a delivery
    with, and one anybody can read back proves nothing about who sent it.
    """

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    _service(session).read(
        "/api/v1/webhooks",
        params={"org_id": scope.org_id, "project_id": scope.project_id},
    )


@webhook.command("add")
@click.argument("url")
@click.option(
    "--secret",
    required=True,
    help="Key deliveries are signed with, as X-Eyvan-Signature: sha256=... over the body. "
    "Configure your receiver with it first; it is never readable back.",
)
@click.option(
    "--entity-type",
    "entity_types",
    multiple=True,
    help="Only deliver events about these, for example workload or cell. Repeatable; default is everything.",
)
@click.option(
    "--action",
    "actions",
    multiple=True,
    help="Only deliver these actions, for example workload.settled. Repeatable; default is everything.",
)
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.pass_obj
def add_webhook(
    session: CliSession,
    url: str,
    secret: str,
    entity_types: tuple[str, ...],
    actions: tuple[str, ...],
    org_id: str | None,
    project_id: str | None,
) -> None:
    """Subscribe URL to this project's events.

    Delivery starts from now rather than from the beginning of the ledger: a new
    integration replaying a month of history into an incident channel is the
    first impression this would otherwise make.
    """

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    JsonRenderer().emit(
        session.client.request(
            "POST",
            "/api/v1/webhooks",
            json={
                "org_id": scope.org_id,
                "project_id": scope.project_id,
                "url": url,
                "secret": secret,
                "entity_types": list(entity_types),
                "actions": list(actions),
            },
        )
    )


@webhook.command("remove")
@click.argument("webhook_id")
@click.pass_obj
def remove_webhook(session: CliSession, webhook_id: str) -> None:
    """Stop delivering to one subscription and forget it."""

    JsonRenderer().emit(session.client.request("DELETE", f"/api/v1/webhooks/{webhook_id}"))


def register_governance_commands(main: click.Group) -> None:
    """Attach the governance command groups to the CLI root."""

    for command in (quota, budget, policy, webhook, system_reconcile):
        main.add_command(command)


__all__ = ("budget", "policy", "quota", "register_governance_commands", "system_reconcile", "webhook")
