# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persistent context commands for the Eyvan CLI.

The context decides who a submission is *from* -- its org and project -- and
that is not obvious from the fleet somebody names, because ``init(fleet=...)``
overrides the fleet and inherits the identity. A context printed as JSON made
the two fields that matter look like two of five, so this renders the identity
as a sentence and keeps ``--json`` for anything parsing.
"""

from __future__ import annotations

import click
from pydantic import ValidationError

from eyvan._internal.json_types import model_json
from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.state import CliSession
from eyvan.sdk.contexts import ContextConfigError, EyvanContext

_renderer = JsonRenderer()


def _render(active_context: str | None, context: EyvanContext) -> None:
    """Print one context as the few facts a submission is decided by."""

    lines = [f"Context {active_context or '(unnamed)'}"]
    for label, value in (
        ("submits as", f"{context.org_id}/{context.project_id}"),
        ("fleet", context.fleet),
        ("plane", context.base_url),
        ("autoscale cache", context.autoscale_cache_dir),
    ):
        if value:
            lines.append(f"  {label:<16} {value}")
    click.echo("\n".join(lines))


@click.command("use")
@click.argument("fleet")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def use_context(session: CliSession, fleet: str, org_id: str | None, project_id: str | None, as_json: bool) -> None:
    """Save FLEET and its account scope as the active context."""

    try:
        context = session.select_context(fleet, org_id=org_id, project_id=project_id)
    except (ContextConfigError, ValidationError, ValueError) as exc:
        raise click.ClickException(f"cannot save Eyvan context: {exc}") from exc
    if as_json:
        _renderer.emit(
            {
                "active_context": session.context_config.active_context,
                **model_json(context, exclude_none=True),
            }
        )
        return
    _render(session.context_config.active_context, context)


@click.command("context")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def show_context(session: CliSession, as_json: bool) -> None:
    """Show the active persistent context."""

    active = session.active_context
    if active is None:
        raise click.ClickException("no active context; run `eyvan use FLEET --org ORG --project PROJECT`")
    if as_json:
        _renderer.emit(
            {
                "active_context": session.context_config.active_context,
                **model_json(active, exclude_none=True),
            }
        )
        return
    _render(session.context_config.active_context, active)


def register_context_commands(group: click.Group) -> None:
    """Attach context commands to the root command group."""

    group.add_command(use_context)
    group.add_command(show_context)


__all__ = ("register_context_commands", "show_context", "use_context")
