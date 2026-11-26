# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Eyvan command composition root."""

from __future__ import annotations

import sys
from collections.abc import Mapping

try:  # readline is unavailable on Windows and in some minimal containers.
    import readline  # Enables line editing and history for Rich prompts.
except ImportError:
    readline = None  # type: ignore[assignment]  # Rich prompts degrade gracefully without it.

import click
from click.core import ParameterSource

from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.client.identity import resolve_control_plane_identity
from eyvan.sdk.commands.artifacts import register_artifact_commands
from eyvan.sdk.commands.assistant import register_assistant_commands
from eyvan.sdk.commands.autoscale import register_autoscale_commands
from eyvan.sdk.commands.capacity import register_capacity_commands
from eyvan.sdk.commands.connect import register_connect_commands
from eyvan.sdk.commands.contexts import register_context_commands
from eyvan.sdk.commands.dashboard import register_dashboard_commands
from eyvan.sdk.commands.demo import register_demo_commands
from eyvan.sdk.commands.doctor import register_doctor_commands
from eyvan.sdk.commands.federation import register_federation_commands
from eyvan.sdk.commands.fleets import register_fleet_commands
from eyvan.sdk.commands.gcp import register_gcp_commands
from eyvan.sdk.commands.governance import register_governance_commands
from eyvan.sdk.commands.head import register_head_commands
from eyvan.sdk.commands.http import EyvanApiError, EyvanClient
from eyvan.sdk.commands.mcp import register_mcp_commands
from eyvan.sdk.commands.operations import register_operation_commands
from eyvan.sdk.commands.proof import register_proof_commands
from eyvan.sdk.commands.query import register_query_commands
from eyvan.sdk.commands.race import register_race_commands
from eyvan.sdk.commands.research import register_research_commands
from eyvan.sdk.commands.resources import register_resource_commands
from eyvan.sdk.commands.runtime import register_runtime_commands
from eyvan.sdk.commands.services import register_service_commands
from eyvan.sdk.commands.soak import register_soak_commands
from eyvan.sdk.commands.soak_monitor import register_soak_monitor_commands
from eyvan.sdk.commands.start import register_start_commands
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.commands.team import register_team_commands
from eyvan.sdk.commands.team_up import register_team_up_commands
from eyvan.sdk.commands.tenancy import register_tenancy_commands
from eyvan.sdk.commands.track import register_track_commands
from eyvan.sdk.commands.workloads import register_workload_commands
from eyvan.sdk.contexts import DEFAULT_BASE_URL, ContextConfigError, FileContextStore
from eyvan.sdk.node import discover_node_context, discover_node_token_or_warn
from eyvan.sdk.team_credentials import team_token_or_warn
from eyvan.sdk.terminal import stream_is_terminal, terminal_console

__all__ = ("EyvanApiError", "EyvanClient", "main")


@click.group()
@click.option("--url", envvar="EYVAN_URL", show_default=DEFAULT_BASE_URL)
@click.option("--token", envvar="EYVAN_TOKEN")
@click.option("--user", envvar="EYVAN_DEV_USER")
@click.option("--org", envvar="EYVAN_ORG")
@click.option("--project", envvar="EYVAN_PROJECT")
@click.option("--role", envvar="EYVAN_ROLE", default="operator", show_default=True)
@click.pass_context
def main(
    ctx: click.Context,
    url: str | None,
    token: str | None,
    user: str | None,
    org: str | None,
    project: str | None,
    role: str,
) -> None:
    """Operate workloads and fleet governance through Eyvan."""

    if ctx.invoked_subcommand == "demo":
        # The demo owns a private disposable runtime. A saved context, remote
        # URL or node credential must never redirect it to an existing fleet.
        return
    context_store = FileContextStore()
    try:
        context_config = context_store.load()
    except ContextConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    # The operator's own context first; then what this machine is part of. On a
    # head that makes every `eyvan` command work with no flags at all, and it is
    # what keeps the endpoint non-loopback there -- loopback is the signal that
    # elects a local development plane on the machine already running the real one.
    active_context = context_config.active or discover_node_context()
    overrides = ContextOverrides(base_url=url, org_id=org, project_id=project)
    endpoint = ControlPlaneEndpoint.parse(url or (active_context.base_url if active_context else DEFAULT_BASE_URL))
    org_id = org or (active_context.org_id if active_context else None)
    project_id = project or (active_context.project_id if active_context else None)

    if (
        token is None
        and user is None
        and endpoint.is_loopback
        and ctx.get_parameter_source("role") is ParameterSource.DEFAULT
    ):
        role = "fleet_admin"
        if stream_is_terminal(sys.stderr):
            # This prints above *every* command run against a local plane, so it
            # is built to recede: one dim line, phrased as where you are rather
            # than as a warning about what was done to you. Only the role is
            # left undimmed, because it is the single word worth catching.
            #
            # Naming the endpoint is not decoration. Two development planes on
            # one machine differ only by port, and "which one did that command
            # hit" is the question this line then answers for free.
            terminal_console(sys.stderr).print(
                f"[dim]development plane[/dim] {endpoint.url} "
                f"[dim]· acting as[/dim] fleet_admin [dim](--role to change)[/dim]"
            )
    identity = resolve_control_plane_identity(
        endpoint,
        token=token
        # A personal token from `eyvan login` for exactly this plane, if any.
        or team_token_or_warn(endpoint.url, warn=lambda message: click.echo(message, err=True))
        or discover_node_token_or_warn(
            endpoint,
            warn=lambda message: click.echo(message, err=True),
        ),
        dev_user=user,
        dev_org=org_id,
        dev_project=project_id,
        dev_roles=role,
    )
    ctx.obj = CliSession(
        client=EyvanClient(endpoint.url, headers=identity.headers()),
        identity=identity,
        context_store=context_store,
        context_config=context_config,
        base_url=endpoint.url,
        org_id=org_id,
        project_id=project_id,
        overrides=overrides,
        client_factory=_client_for,
    )


def _client_for(url: str, headers: Mapping[str, str]) -> EyvanClient:
    """Build a CLI client, resolving ``EyvanClient`` at call time for embedding and tests."""

    return EyvanClient(url, headers=headers)


for register_commands in (
    register_start_commands,
    register_connect_commands,
    register_context_commands,
    register_demo_commands,
    register_doctor_commands,
    register_fleet_commands,
    register_gcp_commands,
    register_head_commands,
    register_workload_commands,
    register_operation_commands,
    register_autoscale_commands,
    register_artifact_commands,
    register_capacity_commands,
    register_governance_commands,
    register_federation_commands,
    register_tenancy_commands,
    register_team_commands,
    register_team_up_commands,
    register_assistant_commands,
    register_research_commands,
    register_mcp_commands,
    register_proof_commands,
    register_query_commands,
    register_race_commands,
    register_resource_commands,
    register_runtime_commands,
    register_service_commands,
    register_soak_commands,
    register_soak_monitor_commands,
    register_dashboard_commands,
    register_track_commands,
):
    register_commands(main)


if __name__ == "__main__":
    main()
