# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cell, lease, event, and preemption commands for the Eyvan CLI."""

from __future__ import annotations

from dataclasses import dataclass, field

import click

from eyvan._internal.json_types import model_json
from eyvan.sdk.client.idempotency import IdempotencyKeyFactory
from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.state import CliSession
from eyvan.sdk.tables import Column, render
from eyvan.sdk.tables import field as at
from eyvan.server.api.schemas import PreemptionForceFinalizeRequest


@dataclass(frozen=True, slots=True)
class OperationsCommandService:
    """Typed application service for control-plane operations."""

    session: CliSession
    renderer: JsonRenderer = field(default_factory=JsonRenderer)
    keys: IdempotencyKeyFactory = field(default_factory=lambda: IdempotencyKeyFactory(prefix="cli"))

    def read(self, path: str, *, params: dict[str, str | int] | None = None) -> None:
        """Read and render one control-plane collection."""

        self.renderer.emit(self.session.client.request("GET", path, params=params))

    def mutate(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        idempotency_key: str | None,
        request: PreemptionForceFinalizeRequest | None = None,
    ) -> None:
        """Apply and render one idempotent control-plane mutation."""

        self.renderer.emit(
            self.session.client.request(
                method,
                path,
                json=model_json(request) if request is not None else None,
                headers={"Idempotency-Key": self.keys.create(operation, idempotency_key)},
            )
        )


#: Shared with `eyvan list cells`, so the two spellings cannot disagree about
#: what a cell is worth showing.
CELL_COLUMNS = (
    Column("id", at("id")),
    Column("state", at("state")),
    Column("backend", at("capacity_backend")),
    Column("zone", at("zone")),
    Column("pool", at("pool_id")),
)


@click.command("cells")
@click.option("--json", "as_json", is_flag=True, help="Emit the durable records instead of a table.")
@click.option("--all", "include_all", is_flag=True, help="Include historical deleted cells in table output.")
@click.pass_obj
def list_cells(session: CliSession, as_json: bool, include_all: bool) -> None:
    """List the machines this fleet currently holds."""

    payload = session.client.request("GET", "/api/v1/cells")
    if as_json or not isinstance(payload, list):
        JsonRenderer().emit(payload)
        return
    if not include_all:
        active = [
            cell for cell in payload if not (isinstance(cell, dict) and str(cell.get("state", "")).lower() == "deleted")
        ]
        empty = (
            "This fleet holds no active cells (pass --all to include deleted history)."
            if payload
            else "This fleet holds no cells."
        )
        for line in render(active, CELL_COLUMNS, empty=empty):
            click.echo(line)
        return
    for line in render(payload, CELL_COLUMNS, empty="This fleet holds no cells."):
        click.echo(line)


@click.command("cell-drain")
@click.argument("cell_id")
@click.option("--idempotency-key")
@click.pass_obj
def drain_cell(session: CliSession, cell_id: str, idempotency_key: str | None) -> None:
    """Drain a cell without terminating its running workloads."""

    OperationsCommandService(session).mutate(
        "POST",
        f"/api/v1/cells/{cell_id}/drain",
        operation="drain",
        idempotency_key=idempotency_key,
    )


@click.command("cell-delete")
@click.argument("cell_id")
@click.option("--idempotency-key")
@click.pass_obj
def delete_cell(session: CliSession, cell_id: str, idempotency_key: str | None) -> None:
    """Release a drained cell after all leases have ended."""

    OperationsCommandService(session).mutate(
        "DELETE",
        f"/api/v1/cells/{cell_id}",
        operation="delete",
        idempotency_key=idempotency_key,
    )


@click.command("preemption-force-finalize")
@click.argument("command_id")
@click.option("--reason", required=True)
@click.option("--idempotency-key")
@click.pass_obj
def force_finalize_preemption(
    session: CliSession,
    command_id: str,
    reason: str,
    idempotency_key: str | None,
) -> None:
    """Attest stopped victims are terminal and retry plan settlement."""

    OperationsCommandService(session).mutate(
        "POST",
        f"/api/v1/preemptions/{command_id}/force-finalize",
        operation="force-finalize",
        idempotency_key=idempotency_key,
        request=PreemptionForceFinalizeRequest(reason=reason),
    )


@click.command("workload-force-finalize")
@click.argument("workload_id")
@click.option("--reason", required=True)
@click.option("--idempotency-key")
@click.pass_obj
def force_finalize_workload(
    session: CliSession,
    workload_id: str,
    reason: str,
    idempotency_key: str | None,
) -> None:
    """Settle a workload whose stop the worker will never confirm.

    Break glass, and it gives up the cell rather than returning it: the process
    may still be running on a host nobody could stop, so scheduling onto it
    again would be worse than the deadlock this escapes.
    """

    OperationsCommandService(session).mutate(
        "POST",
        f"/api/v1/workloads/{workload_id}/force-finalize",
        operation="workload-force-finalize",
        idempotency_key=idempotency_key,
        request=PreemptionForceFinalizeRequest(reason=reason),
    )


@click.command("leases")
@click.option("--org", "org_id", required=True)
@click.option("--project", "project_id", required=True)
@click.pass_obj
def list_leases(session: CliSession, org_id: str, project_id: str) -> None:
    """List capacity leases for a project."""

    OperationsCommandService(session).read(
        "/api/v1/leases",
        params={"org_id": org_id, "project_id": project_id},
    )


@click.command("events")
@click.option("--org", "org_id", required=True)
@click.option("--project", "project_id", required=True)
@click.option("--after", type=int, default=0, show_default=True)
@click.option("--follow", "-f", is_flag=True, help="Stream new events until interrupted.")
@click.pass_obj
def list_events(session: CliSession, org_id: str, project_id: str, after: int, follow: bool) -> None:
    """List ordered audit events for a project.

    With ``--follow`` this streams the existing server-sent-event route rather
    than polling: the ledger's sequence is a monotonic high-watermark and the
    route already honours ``Last-Event-ID``, so a reconnect resumes exactly where
    it stopped instead of replaying or skipping.
    """

    if not follow:
        OperationsCommandService(session).read(
            "/api/v1/events",
            params={"org_id": org_id, "project_id": project_id, "after_sequence": after},
        )
        return
    stream = session.client.stream_lines(
        "GET",
        "/api/v1/events/stream",
        params={"org_id": org_id, "project_id": project_id, "after_sequence": after},
        headers={"Accept": "text/event-stream"},
    )
    try:
        for line in stream:
            # Only data frames carry an event; comments (":" keep-alives) and the
            # blank frame separators are protocol, not content, and printing them
            # would make the output unpipeable.
            if line.startswith("data:"):
                click.echo(line[len("data:") :].strip())
    except KeyboardInterrupt:  # pragma: no cover - interactive exit
        pass


def register_operation_commands(group: click.Group) -> None:
    """Attach operational commands to the root command group."""

    for command in (
        list_cells,
        drain_cell,
        delete_cell,
        force_finalize_preemption,
        force_finalize_workload,
        list_leases,
        list_events,
    ):
        group.add_command(command)


__all__ = ("OperationsCommandService", "register_operation_commands")
