# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A fleet's operator-written bash init script, kept as one marked ``runtime.init_commands`` entry.

Pool runtime setup already runs ``init_commands`` on every new host, after the
worker environment exists and before the worker unit starts. A script edited in
the dashboard becomes exactly one more entry there, recognisable by its first
line, so it can be read back, replaced or removed without touching the entries
a fleet YAML declares.

The script is written to a private temporary file and run with ``bash`` and
stdin from ``/dev/null``: fed on stdin instead, any command in it that reads
stdin (``apt``, ``read``, ``ssh``) would swallow the rest of the script. A
non-zero exit fails the host's bootstrap, which is what a failed setup should do.
"""

from __future__ import annotations

import hashlib
import re
import typing as tp

#: First line of the managed entry; nothing else starts with it.
MARKER = "# eyvan:fleet-init-script v1"
# Not an ``EYVAN_*`` name: scripts/enumerate_env_surface.py reads every quoted
# ``"EYVAN_..."`` literal as an environment knob.
_DELIMITER = "END_OF_EYVAN_FLEET_INIT_SCRIPT"
#: Upper bound on one script. Startup metadata is size-limited and a setup
#: script that large belongs in an artifact the script downloads.
MAX_SCRIPT_BYTES = 32 * 1024

_ENTRY = re.compile(
    re.escape(MARKER) + r"\n.*?<<'" + _DELIMITER + r"'\n(?P<script>.*?)\n?" + _DELIMITER + r"\n",
    re.DOTALL,
)


class InitScriptError(ValueError):
    """The script cannot be stored safely (too large, NUL bytes, the delimiter line)."""


def normalize(script: str) -> str:
    """Return the script as stored: newlines unified, trailing whitespace trimmed; validate it."""

    text = script.replace("\r\n", "\n").replace("\r", "\n").rstrip()
    if "\x00" in text:
        raise InitScriptError("the init script must not contain NUL bytes")
    if len(text.encode("utf-8")) > MAX_SCRIPT_BYTES:
        raise InitScriptError(f"the init script must be at most {MAX_SCRIPT_BYTES} bytes")
    if any(line.strip() == _DELIMITER for line in text.split("\n")):
        raise InitScriptError(f"the init script must not contain a line reading {_DELIMITER}")
    return text


def render(script: str) -> str:
    """Return the ``init_commands`` entry that runs ``script`` (already normalized)."""

    return (
        f"{MARKER}\n"
        'eyvan_fleet_init="$(mktemp)"\n'
        f"cat > \"$eyvan_fleet_init\" <<'{_DELIMITER}'\n"
        f"{script}\n"
        f"{_DELIMITER}\n"
        'bash "$eyvan_fleet_init" </dev/null\n'
        'rm -f "$eyvan_fleet_init"'
    )


def is_entry(command: str) -> bool:
    """Return whether an ``init_commands`` entry is the managed fleet script."""

    return command.startswith(MARKER + "\n")


def extract(init_commands: tp.Iterable[str]) -> str | None:
    """Return the managed script in ``init_commands``, or ``None`` when there is none."""

    for command in init_commands:
        if is_entry(command):
            match = _ENTRY.match(command if command.endswith("\n") else command + "\n")
            return match.group("script") if match else None
    return None


def with_script(init_commands: tp.Sequence[str], script: str | None) -> tuple[str, ...]:
    """Return ``init_commands`` with the managed entry replaced (appended last), or removed when ``script`` is empty."""

    kept = tuple(command for command in init_commands if not is_entry(command))
    if not script:
        return kept
    return (*kept, render(script))


def digest(script: str | None) -> str | None:
    """A short content identity for audit events (the script itself may hold sensitive setup)."""

    return None if not script else hashlib.sha256(script.encode("utf-8")).hexdigest()[:16]


__all__ = (
    "MARKER",
    "MAX_SCRIPT_BYTES",
    "InitScriptError",
    "digest",
    "extract",
    "is_entry",
    "normalize",
    "render",
    "with_script",
)
