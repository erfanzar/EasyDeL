# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``eyvan assistant claude-code``: the plane host's Claude Code sign-in (fleet admins).

The Claude Code assistant provider answers through the ``claude`` CLI installed
on the plane host, with that host's own Claude subscription. These commands
show whether it is installed and signed in, and start a sign-in. The sign-in
runs and completes **on the plane host**: Claude Code opens the host's browser.
"""

from __future__ import annotations

import typing as tp

import click

from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.state import CliSession

STATUS_PATH = "/api/v1/assistant/claude-code/status"
LOGIN_PATH = "/api/v1/assistant/claude-code/login"


def _object(payload: object) -> dict[str, tp.Any]:
    if not isinstance(payload, dict):
        raise click.ClickException("the plane returned an unexpected answer")
    return payload


def _yes_no(value: object) -> str:
    return "yes" if value is True else ("no" if value is False else "unknown")


@click.group("claude-code")
def claude_code_group() -> None:
    """The plane host's Claude Code CLI, used by the Claude Code assistant provider."""


@claude_code_group.command("status")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def claude_code_status(session: CliSession, as_json: bool) -> None:
    """Whether Claude Code is installed and signed in on the plane host (fleet admins)."""

    payload = _object(session.client.request("GET", STATUS_PATH))
    if as_json:
        JsonRenderer().emit(payload)
        return
    click.echo(f"installed:    {_yes_no(payload.get('installed'))}")
    if payload.get("version"):
        click.echo(f"version:      {payload['version']}")
    click.echo(f"signed in:    {_yes_no(payload.get('logged_in'))}")
    if payload.get("auth_method"):
        click.echo(f"auth method:  {payload['auth_method']}")
    if payload.get("subscription_type"):
        click.echo(f"plan:         {payload['subscription_type']}")
    if payload.get("detail"):
        click.echo(f"note:         {payload['detail']}")


@claude_code_group.command("login")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def claude_code_login(session: CliSession, as_json: bool) -> None:
    """Start `claude auth login` on the plane host (fleet admins).

    The sign-in completes on the plane host: Claude Code opens that host's
    browser. When the plane printed a sign-in link it is shown here too.
    """

    payload = _object(session.client.request("POST", LOGIN_PATH))
    if as_json:
        JsonRenderer().emit(payload)
        return
    click.echo("Claude Code sign-in started on the plane host; finish it in the browser there.")
    if payload.get("url"):
        click.echo(f"sign-in link: {payload['url']}")
    click.echo("Check progress with `eyvan assistant claude-code status`.")


__all__ = ("claude_code_group",)
