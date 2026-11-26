# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""No command may make the durable record its only answer.

Eyvan stores records built to be complete: a workload is sixty keys, a capacity
pool is thirty-five, an apply request is seventeen pools of those. Printing one
is the right response to something parsing and the wrong response to a person,
and for a long time it was the only response either got -- ``workload status``,
``capacity fleets``, ``context``, ``doctor`` and ``up --dry-run`` each emitted a
document where a sentence was wanted.

Those were fixed one at a time as somebody hit them, which is the wrong shape of
fix: the next command added emits a record too, and nobody notices until it is
in front of a user. So the rule is enforced here instead of remembered.

**The rule:** a command that can emit a stored record must offer ``--json``. The
flag is what makes the record a deliberate request rather than the default, and
its presence is the thing this asserts -- a command that renders for a person
and never emits a record needs no flag and is not required to have one.
"""

from __future__ import annotations

import inspect

import click

from eyvan.sdk.cli import main

#: Emitting one of these is emitting the durable document.
_EMITTERS = ("renderer.emit(", "_renderer.emit(")

#: Emitting through this is emitting it *only when asked*, which is the point.
_GUARDED = "emit_result("


def _commands() -> list[tuple[str, click.Command]]:
    """Return every leaf command in the CLI, addressed as a person types it."""

    found: list[tuple[str, click.Command]] = []

    def walk(command: click.Command, path: tuple[str, ...]) -> None:
        if isinstance(command, click.Group):
            for name, sub in command.commands.items():
                walk(sub, (*path, name))
            return
        found.append((" ".join(path), command))

    walk(main, ())
    return found


def _emits_a_record(command: click.Command) -> bool:
    """Return whether this command's own callback prints a stored record."""

    callback = command.callback
    if callback is None:
        return False
    try:
        source = inspect.getsource(callback)
    except (OSError, TypeError):  # pragma: no cover - a callback without source
        return False
    return any(token in source for token in _EMITTERS)


def _offers_json(command: click.Command) -> bool:
    return any("--json" in (param.opts or ()) for param in command.params)


def test_every_command_that_emits_a_record_offers_json() -> None:
    """A record is available on request, never as the only thing on offer."""

    offenders = [name for name, command in _commands() if _emits_a_record(command) and not _offers_json(command)]
    assert offenders == [], (
        "these commands print the durable record with no --json flag, which means "
        f"the record is their default answer rather than an opt-in: {offenders}"
    )


def test_the_commands_that_regressed_still_render_for_a_person() -> None:
    """The five that were caught in the wild, pinned by name.

    The check above is structural and would pass if one of these grew a --json
    flag and kept dumping by default. These are the specific ones somebody hit,
    so they are asserted to actually offer the choice.
    """

    by_name = dict(_commands())
    for name in ("context", "use", "doctor", "up", "capacity fleets"):
        assert name in by_name, f"{name} is no longer a command; update this test"
        assert _offers_json(by_name[name]), f"`eyvan {name}` must offer --json"


def test_no_command_service_emits_a_record_without_being_asked() -> None:
    """The check above reads callbacks, and half the CLI delegates.

    `workload cancel` printed a sixty-key record while passing every structural
    check: its callback emits nothing, and the `emit` lives one call away in
    `WorkloadCommandService.mutate`. Reading callbacks alone declared the whole
    delegating half of the CLI compliant.

    So the invariant is asserted where the emit actually is: a function in the
    commands package that prints the durable record must also take `as_json`,
    which is the only thing that makes printing it a choice the caller made.
    ``emit_result`` satisfies this by construction, which is why it exists.
    """

    import ast
    from pathlib import Path

    import eyvan.sdk.commands as commands_package

    package = Path(commands_package.__file__).parent
    offenders: list[str] = []
    for module in sorted(package.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.get_source_segment(module.read_text(encoding="utf-8"), node) or ""
            if not any(token in body for token in _EMITTERS):
                continue
            # However the choice is carried: a parameter, a field the service was
            # built with, or one it forwards. What matters is that something
            # decided, not the shape of the decision.
            threaded = (
                any(argument.arg == "as_json" for argument in [*node.args.args, *node.args.kwonlyargs])
                or "as_json" in body
            )
            if not threaded:
                offenders.append(f"{module.name}::{node.name}")
    # Named rather than hidden. Each of these still prints the durable record
    # with no way for the caller to decline it, and each is a place somebody
    # will eventually be handed sixty keys they did not ask for -- `workload
    # cancel` was on this list until an operator hit it mid-session and said so.
    #
    # This set may shrink and must never grow. A new entry means a command was
    # written that dumps by default, which is the thing being prevented.
    known = {
        "governance.py::_put",
        "operations.py::read",
        "operations.py::mutate",
        "query.py::run",
        "services.py::health",
        "services.py::decisions",
        "workloads.py::_emit_workload_response",
        "workloads.py::run",
        "workloads.py::submit",
        "workloads.py::submit_batch",
        "workloads.py::submit_dag",
        "workloads.py::process_status",
        "workloads.py::explain",
    }
    new_offenders = sorted(set(offenders) - known)
    assert new_offenders == [], (
        "these print the durable record with no way for the caller to decline it; "
        f"route them through emit_result and take as_json: {new_offenders}"
    )
    fixed = sorted(known - set(offenders))
    assert fixed == [], f"these were fixed -- remove them from `known`: {fixed}"
