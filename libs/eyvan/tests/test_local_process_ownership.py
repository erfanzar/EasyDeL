# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Worker-ownership matching must mean the same thing on Linux and on macOS.

The launch nonce is spelled ``--worker-token=<nonce>``, so it is part of an
argument and never an argument by itself. The ``/proc`` branch compared it
against whole argv elements and so never matched on Linux: every ownership check
answered "not my worker", ``stop`` set its message and signalled nothing, and
``retire`` waited its full timeout before declaring the workload INDETERMINATE
without ever sending SIGKILL. macOS has no ``/proc`` and fell through to the
``ps`` branch, which compares against the whole command line -- so the only
platform that could stop a native-local workload was the one Eyvan is not
deployed on.

Four end-to-end tests in ``test_local_runtime.py`` failed on every Linux CI run
and passed on every developer machine. These pin the matcher itself, on any
platform.
"""

from __future__ import annotations

import secrets

from eyvan.worker.runtime.local_process import _arguments_carry_token


def _linux_cmdline(*arguments: str) -> list[bytes]:
    """Build argv exactly as ``/proc/<pid>/cmdline`` reports it."""

    return "\0".join(arguments).encode().split(b"\0")


def test_the_nonce_is_found_inside_the_flag_that_carries_it() -> None:
    token = secrets.token_urlsafe(24)
    arguments = _linux_cmdline(
        "/usr/bin/python3",
        "-m",
        "eyvan.worker.runtime.local_worker",
        f"--worker-token={token}",
        "--record=/var/lib/eyvan/workload-1.json",
    )

    assert _arguments_carry_token(arguments, token.encode())


def test_a_different_worker_is_not_claimed() -> None:
    """A nonce that is not there must not match, or stop would signal a stranger."""

    arguments = _linux_cmdline("/usr/bin/python3", f"--worker-token={secrets.token_urlsafe(24)}")

    assert not _arguments_carry_token(arguments, secrets.token_urlsafe(24).encode())


def test_an_empty_command_line_claims_nothing() -> None:
    """A reaped process reads as empty, and must not be mistaken for a match."""

    assert not _arguments_carry_token(b"".split(b"\0"), b"nonce")


def test_the_token_is_also_found_when_passed_as_its_own_argument() -> None:
    """The launcher's spelling must not be what makes ownership work."""

    token = secrets.token_urlsafe(24)

    assert _arguments_carry_token(_linux_cmdline("python3", "--worker-token", token), token.encode())
