# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic terminal detection and Rich console construction for the Eyvan SDK."""

from __future__ import annotations

import os
import sys
import typing as tp
from collections.abc import Mapping
from dataclasses import dataclass

from rich.console import Console


def stream_is_terminal(stream: tp.TextIO) -> bool:
    """Return the stream's physical TTY state without consulting color variables.

    Commands use this physical check when choosing human output versus JSON so
    ``FORCE_COLOR`` can never turn redirected machine output into a table.
    """

    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except (OSError, ValueError):
        return False


def _enabled(environment: Mapping[str, str], name: str) -> bool:
    """Match Rich's established non-empty environment-variable semantics."""

    return environment.get(name, "") != ""


@dataclass(frozen=True, slots=True)
class TerminalRendering:
    """Resolved layout and ANSI-color capabilities for one output stream.

    Layout precedence is explicit ``force_terminal``, then ``NO_COLOR``
    suppressing an implicit ``FORCE_COLOR`` promotion, then ``FORCE_COLOR``,
    then the stream's physical TTY state. ``NO_COLOR`` always disables ANSI
    styling, including when explicit terminal layout is requested.
    """

    is_terminal: bool
    color_enabled: bool

    @classmethod
    def resolve(
        cls,
        stream: tp.TextIO,
        *,
        force_terminal: bool | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> TerminalRendering:
        """Resolve terminal layout and color without delegating precedence to Rich."""

        resolved_environment = os.environ if environment is None else environment
        no_color = _enabled(resolved_environment, "NO_COLOR")
        force_color = _enabled(resolved_environment, "FORCE_COLOR")
        if force_terminal is not None:
            is_terminal = force_terminal
        elif force_color and not no_color:
            is_terminal = True
        else:
            is_terminal = stream_is_terminal(stream)
        return cls(is_terminal=is_terminal, color_enabled=is_terminal and not no_color)


def cyan(text: str, *, stream: tp.TextIO | None = None) -> str:
    """Return ``text`` wrapped in ANSI cyan when the stream renders color.

    Worker-log prefixes are attribution, not content: they should recede next
    to what the job actually printed. Piped or redirected output gets the bare
    string -- escape codes in a log file are corruption, not decoration.
    """

    resolved = TerminalRendering.resolve(sys.stdout if stream is None else stream)
    if not resolved.color_enabled:
        return text
    return f"\033[36m{text}\033[0m"


def print_block(console: Console, renderable: object) -> None:
    """Print a table or panel with soft wrapping switched off for this call.

    The console is built with ``soft_wrap=True`` because that is right for
    prose: Rich emits the line and the terminal decides where it breaks. It is
    wrong for anything with a border. A table rendered wider than the real
    terminal is wrapped by the terminal at a byte boundary, which lands halfway
    through a multi-byte box-drawing character -- so the rule turns into a
    replacement glyph and the next row's text is written over what is left of
    it. That is what `eyvan autoscale generate` looked like on a narrow window.

    With soft wrapping off for the call, Rich crops and wraps to the width it
    measured, and every border stays whole.
    """

    console.print(renderable, soft_wrap=False)


def terminal_console(
    stream: tp.TextIO,
    *,
    force_terminal: bool | None = None,
    environment: Mapping[str, str] | None = None,
) -> Console:
    """Create an Eyvan Rich console with resolved, immutable rendering semantics."""

    resolved_environment = dict(os.environ if environment is None else environment)
    rendering = TerminalRendering.resolve(
        stream,
        force_terminal=force_terminal,
        environment=resolved_environment,
    )
    return Console(
        file=stream,
        highlight=False,
        soft_wrap=True,
        force_terminal=rendering.is_terminal,
        color_system="auto" if rendering.color_enabled else None,
        no_color=not rendering.color_enabled,
        _environ=resolved_environment,
    )


__all__ = ("TerminalRendering", "print_block", "stream_is_terminal", "terminal_console")
