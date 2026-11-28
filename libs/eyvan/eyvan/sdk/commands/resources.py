# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One way to read any kind of record: ``eyvan list`` and ``eyvan get``.

Every noun had its own command, each with its own flags and its own name for
the same idea. That is fine to read and miserable to script: a tool that walks
the fleet has to learn one grammar per resource, and a person has to remember
whether cells live under ``capacity`` or somewhere else. Ray solved this with
``ray list`` / ``ray get`` over a fixed resource vocabulary; this is the same.

The per-noun commands stay. They carry meaning this cannot -- ``workload
explain`` is not a projection of a record -- and nothing here replaces them.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

import click

from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.state import CliSession
from eyvan.sdk.tables import Column, render
from eyvan.sdk.tables import field as at


@dataclass(frozen=True, slots=True)
class Resource:
    """One kind of record, and the routes that read it.

    Attributes:
        name: The plural name used on the command line.
        list_path: Route returning every record of this kind.
        get_path: Route returning one, with ``{id}`` where the identifier goes.
            ``None`` for a kind with no addressable single record.
        scoped: Whether reading requires an org and project.
        summary: One line, shown when the vocabulary is listed.
        filters: Query parameters this kind accepts through ``--filter``.
        aliases: Other spellings accepted for ``name``.
        org_scoped: Whether reading requires only an org, not a project.
    """

    name: str
    list_path: str
    get_path: str | None
    scoped: bool
    summary: str
    filters: tuple[str, ...] = ()
    aliases: tuple[str, ...] = field(default_factory=tuple)
    org_scoped: bool = False


#: Every resource this surface reads. Each entry is a route that already exists
#: -- this adds a vocabulary, never an endpoint, so nothing here can be reachable
#: in one client and absent in another.
RESOURCES: tuple[Resource, ...] = (
    Resource(
        name="workloads",
        list_path="/api/v1/workloads",
        get_path="/api/v1/workloads/{id}",
        scoped=True,
        summary="jobs, in every lifecycle state",
        filters=("state",),
        aliases=("workload", "jobs", "job"),
    ),
    Resource(
        name="cells",
        list_path="/api/v1/cells",
        get_path="/api/v1/cells/{id}",
        scoped=False,
        summary="machines the fleet currently holds",
        aliases=("cell", "nodes", "node"),
    ),
    Resource(
        name="pools",
        list_path="/api/v1/capacity-pools",
        get_path="/api/v1/capacity-pools/{id}",
        scoped=False,
        summary="declared capacity a fleet may draw on",
        filters=("fleet_name",),
        aliases=("pool", "capacity-pools", "capacity-pool"),
    ),
    Resource(
        name="leases",
        list_path="/api/v1/leases",
        get_path="/api/v1/leases/{id}",
        scoped=True,
        summary="fenced claims on a cell, held by one workload",
        filters=("workload_id",),
        aliases=("lease",),
    ),
    Resource(
        name="events",
        list_path="/api/v1/events",
        get_path=None,
        scoped=True,
        summary="the audit record, newest last",
        filters=("entity_type", "entity_id", "after_sequence", "limit"),
        aliases=("event", "audit"),
    ),
    Resource(
        name="fleets",
        list_path="/api/v1/fleets",
        get_path="/api/v1/fleets/{id}",
        scoped=True,
        summary="fleets this scope may submit to",
        aliases=("fleet",),
    ),
    Resource(
        name="quotas",
        list_path="/api/v1/quotas",
        get_path=None,
        scoped=False,
        summary="declared limits per scope",
        filters=("project_id", "principal_id"),
        aliases=("quota",),
        org_scoped=True,
    ),
    Resource(
        name="budgets",
        list_path="/api/v1/budgets",
        get_path=None,
        scoped=False,
        summary="declared spend ceilings per scope",
        aliases=("budget",),
        org_scoped=True,
    ),
)

#: What each kind shows when it is listed for a person.
#:
#: One row per record and the few fields that answer "what is this and what is
#: it doing" -- the durable record answers far more and is what `--json` is for.
#: A kind with no entry here falls back to `--json`, so adding a resource can
#: never produce a blank table.
LISTING_COLUMNS: dict[str, tuple[Column, ...]] = {
    "workloads": (
        Column("id", at("id")),
        Column("name", at("spec", "name")),
        Column("state", at("state")),
        Column("shape", at("spec", "accelerator_type")),
        Column("class", at("spec", "capacity_class")),
        Column("submitted by", at("created_by")),
    ),
    "cells": (
        Column("id", at("id")),
        Column("state", at("state")),
        Column("backend", at("capacity_backend")),
        Column("zone", at("zone")),
        Column("pool", at("pool_id")),
    ),
    "pools": (
        Column("id", at("id")),
        Column("fleet", at("fleet_name")),
        Column("shape", at("accelerator_type")),
        Column("class", at("capacity_class")),
        Column("zone", at("zone")),
        Column("max", at("max_cells"), align=">"),
        Column("enabled", at("enabled")),
    ),
    "leases": (
        Column("id", at("id")),
        Column("workload", at("workload_id")),
        Column("cell", at("cell_id")),
        Column("state", at("state")),
        Column("cost", at("settled_cost"), align=">"),
    ),
    "events": (
        Column("sequence", at("sequence"), align=">"),
        Column("action", at("action")),
        Column("actor", at("actor")),
        Column("target", at("entity_id")),
        Column("at", at("recorded_at")),
    ),
    "fleets": (Column("name", at("name")),),
}


_BY_NAME: dict[str, Resource] = {
    spelling: resource for resource in RESOURCES for spelling in (resource.name, *resource.aliases)
}


def resolve(kind: str) -> Resource:
    """Return the resource ``kind`` names, or refuse with the vocabulary.

    An unknown noun lists what is known rather than failing bare. Guessing the
    name of a resource is the first thing anyone does with a command like this,
    and a bare "unknown resource" makes the second guess no better informed.
    """

    resource = _BY_NAME.get(kind.strip().lower())
    if resource is not None:
        return resource
    known = ", ".join(entry.name for entry in RESOURCES)
    raise click.UsageError(f"unknown resource {kind!r}; Eyvan reads: {known}")


def _params(
    session: CliSession,
    resource: Resource,
    *,
    org_id: str | None,
    project_id: str | None,
    filters: tp.Mapping[str, str],
) -> dict[str, str]:
    """Build the query for one read, refusing filters the kind does not accept."""

    params: dict[str, str] = {}
    if resource.scoped:
        scope = session.workload_scope(org_id=org_id, project_id=project_id)
        params["org_id"] = scope.org_id
        params["project_id"] = scope.project_id
    elif resource.org_scoped:
        params["org_id"] = session.workload_scope(org_id=org_id).org_id
        # Organization listings must not implicitly narrow to the session's
        # project. Only routes with an optional project query accept --project.
        if project_id is not None and "project_id" in resource.filters:
            params["project_id"] = project_id
    for key, value in filters.items():
        if key not in resource.filters:
            accepted = ", ".join(resource.filters) if resource.filters else "none"
            raise click.UsageError(f"{resource.name} accepts no filter {key!r}; it accepts: {accepted}")
        params[key] = value
    return params


def _filters(values: tuple[str, ...]) -> dict[str, str]:
    """Parse repeated ``--filter KEY=VALUE`` without accepting ambiguous input."""

    parsed: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key.strip() or not item.strip():
            raise click.UsageError("--filter values must use KEY=VALUE")
        parsed[key.strip()] = item.strip()
    return parsed


@click.command("list")
@click.argument("kind", required=False)
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option(
    "--filter",
    "filters",
    multiple=True,
    metavar="KEY=VALUE",
    help="Narrow the listing. Each resource accepts its own keys; an unaccepted one is refused by name.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the durable records instead of a table.")
@click.pass_obj
def list_resource(
    session: CliSession,
    kind: str | None,
    org_id: str | None,
    project_id: str | None,
    filters: tuple[str, ...],
    as_json: bool,
) -> None:
    """List every record of KIND.

    With no KIND this prints the vocabulary, so the first thing anyone types
    tells them what else they can type.
    """

    if kind is None:
        for resource in RESOURCES:
            click.echo(f"{resource.name:<12}{resource.summary}")
        return
    resource = resolve(kind)
    payload = session.client.request(
        "GET",
        resource.list_path,
        params=_params(session, resource, org_id=org_id, project_id=project_id, filters=_filters(filters)),
    )
    columns = LISTING_COLUMNS.get(resource.name)
    if as_json or columns is None or not isinstance(payload, list):
        JsonRenderer().emit(payload)
        return
    if resource.name == "fleets":
        # The submit-fleet endpoint returns names, not fleet-status objects.
        # Normalize only the human view; --json retains the original list[str].
        payload = [{"name": item} if isinstance(item, str) else item for item in payload]
    for line in render(payload, columns, empty=f"No {resource.name} match." if filters else f"No {resource.name}."):
        click.echo(line)


@click.command("get")
@click.argument("kind")
@click.argument("resource_id")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.pass_obj
def get_resource(
    session: CliSession,
    kind: str,
    resource_id: str,
    org_id: str | None,
    project_id: str | None,
) -> None:
    """Get the one record of KIND named RESOURCE_ID."""

    resource = resolve(kind)
    if resource.get_path is None:
        raise click.UsageError(
            f"{resource.name} are not addressed one at a time; use 'eyvan list {resource.name}' with a filter"
        )
    JsonRenderer().emit(
        session.client.request(
            "GET",
            resource.get_path.format(id=resource_id),
            params=_params(session, resource, org_id=org_id, project_id=project_id, filters={}),
        )
    )
    # `get` stays the record. It is the accessor -- `eyvan list` is the view,
    # and something is piping this one.


def register_resource_commands(group: click.Group) -> None:
    """Attach the generic read surface to the root command group."""

    group.add_command(list_resource)
    group.add_command(get_resource)


__all__ = (
    "RESOURCES",
    "Resource",
    "get_resource",
    "list_resource",
    "register_resource_commands",
    "resolve",
)
