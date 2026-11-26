# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A stopped soak must still hand over what it learned.

A long soak is started in the background, and a background job inherits SIGINT
ignored -- so the KeyboardInterrupt path that writes the receipt is unreachable
for exactly the runs that need it. SIGTERM is what an operator sends, and its
default action ends the process with the whole longitudinal state in memory. The
terminated P8 lost the identities of three persistent violations that way, and
they were the only thing seventy-eight hours had produced that nothing else had.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from eyvan.sdk.commands import soak as soak_module


def test_sigterm_is_routed_into_the_graceful_exit() -> None:
    """The handler must be installed while the run is in progress."""

    installed: list[object] = []
    with soak_module._terminate_as_interrupt():
        installed.append(signal.getsignal(signal.SIGTERM))

    assert callable(installed[0])
    with pytest.raises(KeyboardInterrupt):
        installed[0](signal.SIGTERM, None)


def test_the_previous_handler_is_restored() -> None:
    """A soak embedded in a larger program must not change its signal behaviour."""

    before = signal.getsignal(signal.SIGTERM)

    with soak_module._terminate_as_interrupt():
        pass

    assert signal.getsignal(signal.SIGTERM) is before


def test_a_terminated_run_still_writes_its_receipt(tmp_path: Path) -> None:
    """The property the lost P8 needed: SIGTERM leaves evidence behind."""

    output = tmp_path / "receipt.json"
    program = textwrap.dedent(
        f"""
        import time
        from eyvan.sdk.commands.soak import _terminate_as_interrupt, write_receipt_atomic

        received = "completed"
        with _terminate_as_interrupt():
            try:
                print("ready", flush=True)
                for _ in range(600):
                    time.sleep(0.05)
            except KeyboardInterrupt:
                received = "interrupted"
        write_receipt_atomic({str(output)!r}, {{"status": received}})
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "ready"

    process.send_signal(signal.SIGTERM)
    process.wait(timeout=30)

    deadline = time.monotonic() + 10
    while not output.exists() and time.monotonic() < deadline:
        time.sleep(0.05)

    assert output.exists(), "SIGTERM ended the process without a receipt"
    assert json.loads(output.read_text())["status"] == "interrupted"
