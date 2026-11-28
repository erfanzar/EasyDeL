# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One read-only statement against the control plane's own database.

The typed commands answer the questions someone anticipated. Incidents are made
of the other kind -- "which workloads on cells in this zone were dispatched in
the ten minutes before the pool went unhealthy" is a join nobody added a route
for, and the honest alternative to this command is an operator opening ``psql``
on the head with an account that can also write, leaving no ledger entry behind.

So the console exists, and the two things that make it safe are enforced on the
server rather than here: the statement runs in a transaction PostgreSQL will not
let it write in, and every attempt is recorded whether it succeeded or not. This
module only carries the statement across and prints what came back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import click

from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.state import CliSession

#: Ceiling the route enforces. Matched here so an impossible request is refused
#: locally with a Click error rather than as a 422 the user has to decode.
MAX_QUERY_ROWS = 10_000


@dataclass(frozen=True, slots=True)
class QueryCommandService:
    """Typed application service used by the query Click callback."""

    session: CliSession
    renderer: JsonRenderer = field(default_factory=JsonRenderer)

    def run(self, statement: str, *, limit: int) -> None:
        """Emit the server's answer unchanged, truncation flag included."""

        self.renderer.emit(
            self.session.client.request(
                "POST",
                "/api/v1/query",
                json={"statement": statement, "limit": limit},
            )
        )


@click.group("process")
def process() -> None:
    """Ask the control-plane process what it is doing right now."""


@process.command("stacks")
@click.pass_obj
def process_stacks(session: CliSession) -> None:
    """Dump every live thread's stack in the control plane.

    The command for a plane that answers health checks and advances nothing.
    Every external signal looks the same during a wedge; the stack is the thing
    that differs.
    """

    JsonRenderer().emit(session.client.request("GET", "/api/v1/process/stacks"))


@process.command("allocations")
@click.pass_obj
def process_allocations(session: CliSession) -> None:
    """Report the largest tracked allocations in the control plane.

    Read ``already_tracing`` first. When it is false the tracker was started by
    this call, so the snapshot covers milliseconds rather than the leak you are
    looking for -- and a small clean snapshot is exactly what a leak looks like
    from that angle.
    """

    JsonRenderer().emit(session.client.request("GET", "/api/v1/process/allocations"))


@click.command("query")
@click.argument("statement")
@click.option(
    "--limit",
    type=click.IntRange(1, MAX_QUERY_ROWS),
    default=200,
    show_default=True,
    help="Rows to return before the answer is truncated.",
)
@click.pass_obj
def query(session: CliSession, statement: str, limit: int) -> None:
    """Run one read-only STATEMENT against the control plane's database.

    Requires fleet administration, not project operation: a statement reads
    whatever the schema holds across every tenant, and no filter applied to the
    rows afterwards would make that a project-scoped power.

    The answer carries ``truncated``. Read it -- a result that exactly fills
    ``--limit`` looks identical to one that was cut short, and the difference is
    usually the row that would have explained the incident.
    """

    QueryCommandService(session).run(statement, limit=limit)


def register_query_commands(group: click.Group) -> None:
    """Attach the read-only console and process introspection to the CLI root."""

    group.add_command(query)
    group.add_command(process)


__all__ = (
    "MAX_QUERY_ROWS",
    "QueryCommandService",
    "process",
    "query",
    "register_query_commands",
)
