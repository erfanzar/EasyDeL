# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Public terminal contracts shared by Eyvan command renderers."""

from __future__ import annotations

import io
import typing as tp

import click
import pytest
from rich.console import Console

from eyvan.sdk.commands.autoscale import autoscale_console
from eyvan.sdk.commands.fleet_status import fleet_status_console, is_interactive_output

ConsoleFactory = tp.Callable[..., Console]


class _OutputBuffer(io.StringIO):
    def __init__(self, *, is_terminal: bool) -> None:
        super().__init__()
        self._is_terminal = is_terminal

    def isatty(self) -> bool:
        return self._is_terminal


@pytest.mark.parametrize("factory", (autoscale_console, fleet_status_console))
def test_command_consoles_share_conflicting_color_precedence(
    monkeypatch: pytest.MonkeyPatch,
    factory: ConsoleFactory,
) -> None:
    stream = _OutputBuffer(is_terminal=False)
    monkeypatch.setattr(click, "get_text_stream", lambda _name: stream)

    console = factory(environment={"NO_COLOR": "1", "FORCE_COLOR": "1"})
    console.print("[bold red]visible[/]")

    assert console.is_terminal is False
    assert stream.getvalue() == "visible\n"


@pytest.mark.parametrize("factory", (autoscale_console, fleet_status_console))
def test_command_consoles_allow_explicit_terminal_layout_without_violating_no_color(
    monkeypatch: pytest.MonkeyPatch,
    factory: ConsoleFactory,
) -> None:
    stream = _OutputBuffer(is_terminal=False)
    monkeypatch.setattr(click, "get_text_stream", lambda _name: stream)

    console = factory(
        force_terminal=True,
        environment={"NO_COLOR": "1", "FORCE_COLOR": "1"},
    )
    console.print("[bold red]visible[/]")

    assert console.is_terminal is True
    assert stream.getvalue() == "visible\n"


def test_machine_output_selection_uses_physical_tty_not_force_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _OutputBuffer(is_terminal=False)
    monkeypatch.setattr(click, "get_text_stream", lambda _name: stream)
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    assert is_interactive_output() is False
