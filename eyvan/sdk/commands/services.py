# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Managed-service inspection commands for the Eyvan CLI.

Health probing, generation-fenced replica replacement and bounded autoscaling
all decide and record durably, and until these commands existed the only way to
read any of it was a SQL client or the raw event stream. An operator asked to
diagnose a service at three in the morning does not have either.

Both commands are read-only and emit the server's document unchanged. That is
deliberate: the health answer has three values and the summariser is the exact
place a third value gets rounded to one of the other two.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import click

from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.state import CliSession

#: Ceiling the route enforces. Matched here so an impossible request is refused
#: locally with a Click error rather than as a 422 the user has to decode.
MAX_DECISION_HISTORY = 500


@dataclass(frozen=True, slots=True)
class ServiceCommandService:
    """Typed application service used by managed-service Click callbacks."""

    session: CliSession
    renderer: JsonRenderer = field(default_factory=JsonRenderer)

    def health(self, service_id: str) -> None:
        """Emit one service's replica health exactly as the server reports it."""

        self.renderer.emit(self.session.client.request("GET", f"/api/v1/workloads/{service_id}/service-health"))

    def decisions(self, service_id: str, *, limit: int) -> None:
        """Emit one service's newest committed decisions, unchanged."""

        self.renderer.emit(
            self.session.client.request(
                "GET",
                f"/api/v1/workloads/{service_id}/service-decisions",
                params={"limit": limit},
            )
        )


@click.group("service")
def service() -> None:
    """Inspect a managed service: replica health and decision history."""


@service.command("health")
@click.argument("service_id")
@click.pass_obj
def service_health(session: CliSession, service_id: str) -> None:
    """Report every replica's health, keeping UNKNOWN distinct from unhealthy.

    ``unknown`` means this control plane could not establish the replica's
    state -- a timeout, a refused connection, a service nobody probes at all. It
    is reported as its own count and never folded into ``unhealthy``, because
    acting on the two the same way is how one bad network minute tears down a
    fleet of accelerators.
    """

    ServiceCommandService(session).health(service_id)


@service.command("decisions")
@click.argument("service_id")
@click.option(
    "--limit",
    type=click.IntRange(1, MAX_DECISION_HISTORY),
    default=50,
    show_default=True,
    help="Newest decisions to return.",
)
@click.pass_obj
def service_decisions(session: CliSession, service_id: str, limit: int) -> None:
    """List what was observed, what was decided and why, newest first.

    Each entry carries the observation the decision was taken from, so "why did
    it scale in at 04:12" is answerable from the record rather than from state
    that has since moved on.
    """

    ServiceCommandService(session).decisions(service_id, limit=limit)


def register_service_commands(group: click.Group) -> None:
    """Attach managed-service commands to the root command group."""

    group.add_command(service)


__all__ = (
    "MAX_DECISION_HISTORY",
    "ServiceCommandService",
    "register_service_commands",
    "service",
    "service_decisions",
    "service_health",
)
