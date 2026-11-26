# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The external P8 monitor: measure the control plane from outside, then post.

``eyvan soak`` is an evaluator. It reads what it is told and refuses to invent
the rest, which is why every live invocation so far has ended `inconclusive`
with `rss_growth_evidence_missing` and its five siblings. This module is the
missing producer.

Everything here is arranged around one rule: **a partial sample is not a
sample.** The closed ``eyvan.soak.external-evidence.v1`` schema has no encoding
for "I could not find out", so a monitor that fails to establish one of its six
measurements emits nothing at all and says which one failed. The evaluator then
reports the specific ``*_evidence_missing`` code it already has, and a soak ends
inconclusive rather than passing on a zero nobody measured.

Every probe therefore returns ``None`` for *unknown* and never a fallback:

* a ``ps`` that is not installed, a pid that has exited, an ``lsof`` that timed
  out -- unknown, not zero descriptors;
* a state document that is not where it was said to be -- unknown, not an empty
  database;
* an invariant response whose violation list was truncated -- unknown, because
  the I5 finding this needs may be one of the ones that was dropped;
* a cell in a state this build does not recognise -- unknown, because a rolling
  upgrade must not be able to turn a state it cannot read into "cleaned up".

The monitor only reads. Its single write is the evidence POST, which touches no
fleet state.
"""

from __future__ import annotations

import json
import math
import os
import re
import selectors
import shlex
import subprocess
import time
import typing as tp
from pathlib import Path

import click

from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.records import CellState
from eyvan.sdk.commands.state import CliSession
from eyvan.soak_contract import MAX_SAFE_COUNTER, is_opaque_id
from eyvan.soak_evidence import ExternalEvidenceSample, build_external_evidence_sample

#: Where an accepted sample is posted.
SOAK_EVIDENCE_PATH: tp.Final[str] = "/api/v1/system/soak-evidence"

#: The invariant whose absence means the ledger reconciles: "a budget's spend
#: equals what its settled leases actually cost".
SPEND_RECONCILIATION_INVARIANT: tp.Final[str] = "I5"

#: The only cell state that owns nothing. Every other state -- including
#: ``DELETING`` -- is capacity, or a provider call, still outstanding.
_RELEASED_CELL_STATE: tp.Final[str] = CellState.DELETED.value

#: Cell states this build understands. A value outside it is not assumed
#: terminal; the whole cleanup measurement becomes unknown instead.
_KNOWN_CELL_STATES: tp.Final[frozenset[str]] = frozenset(state.value for state in CellState)

#: Largest stdout or stderr an inventory command may print before it is unusable.
#: A count is at most twenty digits; large diagnostics also make the probe unknown.
MAX_INVENTORY_OUTPUT_BYTES: tp.Final[int] = 4_096

#: Per-stream acquisition ceiling for ps/lsof, whose descriptor listings can be
#: much longer than an inventory count. Oversized output is unknown, not truncated.
MAX_PROCESS_OUTPUT_BYTES: tp.Final[int] = 1_048_576

#: Largest pid file the monitor will read.
MAX_PID_FILE_BYTES: tp.Final[int] = 64

#: Largest cell listing the monitor will judge cleanup from. Far past any fleet
#: Eyvan runs; a response longer than this is not a fleet inventory, and the
#: answer becomes unknown rather than derived from whatever arrived.
MAX_CLEANUP_CELLS: tp.Final[int] = 10_000

_DIGITS = re.compile(r"[0-9]{1,19}")


def _bounded_count(text: str) -> int | None:
    """Return a bounded non-negative integer parsed from exact digits, or None."""

    stripped = text.strip()
    if _DIGITS.fullmatch(stripped) is None:
        return None
    value = int(stripped)
    return value if 0 <= value <= MAX_SAFE_COUNTER else None


def _run_command(
    argv: tp.Sequence[str], *, timeout_s: float, max_output_bytes: int = MAX_PROCESS_OUTPUT_BYTES
) -> subprocess.CompletedProcess[str] | None:
    """Capture bounded streams, killing and reaping a timed-out or oversized child."""

    if not math.isfinite(timeout_s) or timeout_s <= 0 or max_output_bytes <= 0:
        return None
    deadline = time.monotonic() + timeout_s
    try:
        # No shell, and no inherited interpreter steering for embedded Python.
        # Read both pipes together so a full stderr pipe cannot deadlock stdout.
        with subprocess.Popen(
            list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=provider_cli_env()
        ) as process:
            try:
                assert process.stdout is not None and process.stderr is not None
                stdout = bytearray()
                stderr = bytearray()
                with selectors.DefaultSelector() as selector:
                    for pipe, buffer in ((process.stdout, stdout), (process.stderr, stderr)):
                        os.set_blocking(pipe.fileno(), False)
                        selector.register(pipe, selectors.EVENT_READ, buffer)
                    while selector.get_map():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            return None
                        for key, _events in selector.select(remaining):
                            buffer = key.data
                            try:
                                chunk = os.read(key.fd, min(65_536, max_output_bytes + 1 - len(buffer)))
                            except BlockingIOError:
                                continue
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            buffer.extend(chunk)
                            if len(buffer) > max_output_bytes:
                                return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                returncode = process.wait(timeout=remaining)
                # Match subprocess text mode's encoding and universal newlines,
                # but decode only after byte-bounded acquisition has completed.
                out = stdout.decode(process.stdout.encoding).replace("\r\n", "\n").replace("\r", "\n")
                err = stderr.decode(process.stderr.encoding).replace("\r\n", "\n").replace("\r", "\n")
                return subprocess.CompletedProcess(list(argv), returncode, out, err)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def read_pid_file(path: Path) -> int | None:
    """Read a control-plane pid from a file, or return None.

    Re-read every tick rather than resolved once, because a P8 run restarts the
    control plane on purpose: a pid captured at start-up names a dead process
    for the rest of the soak, and measuring a dead process is exactly the
    silence this monitor exists to make visible.

    Args:
        path: The pid file written by the supervisor.

    Returns:
        The pid, or ``None`` when the file is missing, oversized, or does not
        contain a single positive integer.
    """

    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_PID_FILE_BYTES + 1)
    except OSError:
        return None
    if len(raw) > MAX_PID_FILE_BYTES:
        return None
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        return None
    parsed = _bounded_count(text)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def measure_process_rss_bytes(pid: int, *, timeout_s: float) -> int | None:
    """Measure another process's resident set size, from outside it.

    ``/proc/<pid>/status`` where it exists, ``ps`` everywhere else. Never
    ``resource.getrusage`` or this process's own memory: the number has to
    describe the control plane, and a monitor reporting its own RSS would make a
    leaking control plane look healthy.

    Args:
        pid: The control-plane process.
        timeout_s: Bound on the ``ps`` fallback.

    Returns:
        Resident bytes, or ``None`` when the process cannot be measured.
    """

    status = Path(f"/proc/{pid}/status")
    try:
        text = status.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    for line in text.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        parts = line.split()
        if len(parts) != 3 or parts[2].lower() != "kb":
            return None
        kilobytes = _bounded_count(parts[1])
        return None if kilobytes is None else kilobytes * 1024

    completed = _run_command(["ps", "-o", "rss=", "-p", str(pid)], timeout_s=timeout_s)
    if completed is None or completed.returncode != 0:
        return None
    kilobytes = _bounded_count(completed.stdout)
    return None if kilobytes is None else kilobytes * 1024


def measure_process_fd_count(pid: int, *, timeout_s: float) -> int | None:
    """Count another process's open descriptors, from outside it.

    Args:
        pid: The control-plane process.
        timeout_s: Bound on the ``lsof`` fallback.

    Returns:
        The open descriptor count, or ``None`` when it cannot be counted.
        ``lsof`` exiting non-zero is unknown rather than zero: it reports "no
        matching processes" the same way, and a leaking control plane must never
        be summarised as one holding no files.
    """

    descriptors = Path(f"/proc/{pid}/fd")
    try:
        return sum(1 for _entry in os.scandir(descriptors))
    except OSError:
        pass

    completed = _run_command(["lsof", "-p", str(pid), "-F", "f"], timeout_s=timeout_s)
    if completed is None or completed.returncode != 0:
        return None
    count = 0
    for line in completed.stdout.splitlines():
        # `-F f` prints one field per line; a descriptor line is `f` followed by
        # the number. `fcwd`, `ftxt` and friends are the process's own image and
        # working directory, which are not descriptors it can leak.
        if line.startswith("f") and _bounded_count(line[1:]) is not None:
            count += 1
    return count if count <= MAX_SAFE_COUNTER else None


def measure_local_state_bytes(path: Path) -> int | None:
    """Measure the local control plane's durable document.

    Args:
        path: The state document ``EYVAN_LOCAL_STATE_PATH`` names.

    Returns:
        Its size in bytes, or ``None`` when it cannot be read. A missing file is
        unknown, not zero: "the path is wrong" and "the plane has written
        nothing yet" look identical from here, and only one of them is evidence.
    """

    try:
        size = path.expanduser().stat().st_size
    except OSError:
        return None
    return size if 0 <= size <= MAX_SAFE_COUNTER else None


def measure_postgres_database_bytes(database_url: str, *, timeout_s: float) -> int | None:
    """Ask PostgreSQL how large the control plane's database is.

    One read, on its own connection, disposed before returning: the monitor
    holds no transaction while it does anything else, and nothing it runs can
    take a lock the control plane then waits behind.

    Args:
        database_url: The SQLAlchemy async URL the control plane uses.
        timeout_s: Bound on connecting and querying.

    Returns:
        ``pg_database_size(current_database())``, or ``None`` on any failure.
    """

    import asyncio

    from sqlalchemy import text as sql_text
    from sqlalchemy.ext.asyncio import create_async_engine

    async def _query() -> object:
        engine = create_async_engine(database_url, poolclass=None)
        try:
            async with engine.connect() as connection:
                result = await connection.execute(sql_text("SELECT pg_database_size(current_database())"))
                return result.scalar_one()
        finally:
            await engine.dispose()

    async def _bounded() -> object:
        return await asyncio.wait_for(_query(), timeout=timeout_s)

    try:
        value = asyncio.run(_bounded())
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_SAFE_COUNTER:
        return None
    return value


def measure_provider_inventory(command: str, *, timeout_s: float) -> int | None:
    """Ask the provider how many cells it still has, through the operator's command.

    A provider call and not a control-plane read, deliberately. The inventory
    that matters at the end of a soak is the one that is still being billed,
    and only the provider knows it; a control plane that lost a cell record
    would report an empty fleet with perfect confidence.

    Args:
        command: An argv string, split without a shell. A pipeline belongs in a
            wrapper script: running the string through a shell would make the
            monitor's behaviour depend on the operator's ``$IFS`` and quoting.
        timeout_s: Bound on the command.

    Returns:
        The count the command printed on stdout, or ``None`` when it could not
        be run, exited non-zero, printed too much, or printed anything that is
        not a single bounded non-negative integer.
    """

    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    if not argv:
        return None
    completed = _run_command(argv, timeout_s=timeout_s, max_output_bytes=MAX_INVENTORY_OUTPUT_BYTES)
    if completed is None or completed.returncode != 0:
        return None
    if len(completed.stdout) > MAX_INVENTORY_OUTPUT_BYTES:
        return None
    return _bounded_count(completed.stdout)


def read_spend_reconciled(client: tp.Any) -> bool | None:
    """Decide whether the control plane's own ledger reconciles.

    Two conditions, both read from the ledger the control plane already keeps:

    * invariant ``I5`` -- a budget's spend equals what its settled leases cost
      -- is not being violated; and
    * no cell is contributing zero cost because its pool declares no unit
      prices. A ledger with unpriced cells records only a cost floor, not a
      reconciled total.

    Args:
        client: An object with the SDK client's ``request`` method.

    Returns:
        ``True`` or ``False`` when both conditions could be evaluated, otherwise
        ``None``. A truncated violation list is ``None``, not ``True``: the I5
        finding may be one of the ones that was dropped.
    """

    try:
        invariants = client.request("GET", "/api/v1/system/invariants")
        billing = client.request("GET", "/api/v1/capacity/billing")
    except Exception:
        return None
    if not isinstance(invariants, dict) or not isinstance(billing, dict):
        return None
    checked = invariants.get("checked")
    violations = invariants.get("violations")
    truncated = invariants.get("violations_truncated")
    if not isinstance(checked, list) or SPEND_RECONCILIATION_INVARIANT not in checked:
        return None
    if truncated is not False or not isinstance(violations, list):
        return None
    for violation in violations:
        if not isinstance(violation, dict) or not isinstance(violation.get("invariant"), str):
            return None
        if violation["invariant"] == SPEND_RECONCILIATION_INVARIANT:
            return False
    unpriced = billing.get("unpriced_cells")
    if isinstance(unpriced, bool) or not isinstance(unpriced, int) or unpriced < 0:
        return None
    return unpriced == 0


def read_cleanup_complete(client: tp.Any) -> bool | None:
    """Decide whether the fleet still owns anything.

    Args:
        client: An object with the SDK client's ``request`` method.

    Returns:
        ``True`` when every cell the control plane lists has reached
        ``deleted``; ``False`` when any has not. ``None`` when the list cannot
        be read, is longer than this monitor will judge, or contains a state
        this build does not recognise -- a rolling upgrade must not be able to
        turn a state it cannot read into "cleaned up".
    """

    try:
        cells = client.request("GET", "/api/v1/cells")
    except Exception:
        return None
    if not isinstance(cells, list) or len(cells) > MAX_CLEANUP_CELLS:
        return None
    outstanding = 0
    for cell in cells:
        if not isinstance(cell, dict):
            return None
        state = cell.get("state")
        if not isinstance(state, str) or state not in _KNOWN_CELL_STATES:
            return None
        if state != _RELEASED_CELL_STATE:
            outstanding += 1
    return outstanding == 0


@click.command("soak-monitor")
@click.option("--proof-id", envvar="EYVAN_SOAK_PROOF_ID", required=True, help="Opaque sha256 proof/run identifier.")
@click.option("--control-plane-pid", type=int, help="Pid of the control-plane process to measure.")
@click.option(
    "--control-plane-pid-file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Pid file re-read every tick, so a restarted control plane keeps being measured.",
)
@click.option(
    "--state-path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Local control-plane state document to size.",
)
@click.option("--database-url", help="PostgreSQL URL to size with pg_database_size.")
@click.option(
    "--provider-inventory-command",
    required=True,
    help="Command printing the provider's live cell count; split without a shell.",
)
@click.option("--interval-s", type=float, default=30.0, show_default=True, help="Seconds between samples.")
@click.option("--samples", type=int, default=0, show_default=True, help="Stop after N samples; 0 watches until killed.")
@click.option("--command-timeout-s", type=float, default=30.0, show_default=True, help="Bound on each external probe.")
@click.pass_obj
def soak_monitor(
    session: CliSession,
    proof_id: str,
    control_plane_pid: int | None,
    control_plane_pid_file: Path | None,
    state_path: Path | None,
    database_url: str | None,
    provider_inventory_command: str,
    interval_s: float,
    samples: int,
    command_timeout_s: float,
) -> None:
    """Measure the control plane from outside it and post proof-bound P8 evidence.

    Emits one JSON line per tick on stdout: ``emitted`` with the exact sample
    that was accepted, ``incomplete`` naming the fields that could not be
    measured, or ``not_delivered`` when the control plane did not take a
    complete sample. Exits non-zero if any tick was not an ``emitted``, so a
    bounded ``--samples`` run is usable as a check.
    """

    if not is_opaque_id(proof_id):
        raise click.UsageError("--proof-id must be a lowercase sha256 opaque identifier")
    if (control_plane_pid is None) == (control_plane_pid_file is None):
        raise click.UsageError("pass exactly one of --control-plane-pid or --control-plane-pid-file")
    if control_plane_pid is not None and control_plane_pid <= 0:
        raise click.UsageError("--control-plane-pid must be a positive process identifier")
    if (state_path is None) == (database_url is None):
        raise click.UsageError("pass exactly one of --state-path or --database-url")
    if not math.isfinite(interval_s) or interval_s < 0:
        raise click.UsageError("--interval-s must be finite and non-negative")
    if not math.isfinite(command_timeout_s) or command_timeout_s <= 0:
        raise click.UsageError("--command-timeout-s must be finite and positive")
    if samples < 0:
        raise click.UsageError("--samples must be zero or positive")

    incomplete = 0
    tick = 0
    try:
        while samples == 0 or tick < samples:
            outcome = _observe_once(
                session,
                proof_id=proof_id,
                control_plane_pid=control_plane_pid,
                control_plane_pid_file=control_plane_pid_file,
                state_path=state_path,
                database_url=database_url,
                provider_inventory_command=provider_inventory_command,
                command_timeout_s=command_timeout_s,
            )
            if outcome["outcome"] != "emitted":
                incomplete += 1
            click.echo(json.dumps(outcome, sort_keys=True))
            tick += 1
            if samples == 0 or tick < samples:
                time.sleep(interval_s)
    except KeyboardInterrupt:
        # Stopping the monitor is an operator action, not a measurement failure.
        pass
    if incomplete:
        raise SystemExit(1)


def _observe_once(
    session: CliSession,
    *,
    proof_id: str,
    control_plane_pid: int | None,
    control_plane_pid_file: Path | None,
    state_path: Path | None,
    database_url: str | None,
    provider_inventory_command: str,
    command_timeout_s: float,
) -> dict[str, object]:
    """Take one whole measurement and post it if -- and only if -- it is whole."""

    pid = control_plane_pid if control_plane_pid is not None else read_pid_file(tp.cast(Path, control_plane_pid_file))
    if pid is None:
        rss_bytes: int | None = None
        fd_count: int | None = None
    else:
        rss_bytes = measure_process_rss_bytes(pid, timeout_s=command_timeout_s)
        fd_count = measure_process_fd_count(pid, timeout_s=command_timeout_s)
    if state_path is not None:
        database_bytes = measure_local_state_bytes(state_path)
    else:
        database_bytes = measure_postgres_database_bytes(
            tp.cast(str, database_url),
            timeout_s=command_timeout_s,
        )
    sample, unmeasurable = build_external_evidence_sample(
        proof_id,
        rss_bytes=rss_bytes,
        fd_count=fd_count,
        database_bytes=database_bytes,
        spend_reconciled=read_spend_reconciled(session.client),
        provider_inventory_count=measure_provider_inventory(
            provider_inventory_command,
            timeout_s=command_timeout_s,
        ),
        cleanup_complete=read_cleanup_complete(session.client),
    )
    if sample is None:
        return {"outcome": "incomplete", "unmeasurable": list(unmeasurable)}
    return _deliver(session, sample)


def _deliver(session: CliSession, sample: ExternalEvidenceSample) -> dict[str, object]:
    """Post one complete sample, reporting a refusal without discarding the run."""

    try:
        session.client.request("POST", SOAK_EVIDENCE_PATH, json=tp.cast(tp.Any, sample.as_dict()))
    except Exception:
        # A control plane that is restarting is not evidence about the fleet.
        # The tick is reported as undelivered and the next one is taken; the
        # evaluator sees the gap as missing evidence, which is what it is.
        return {"outcome": "not_delivered", "unmeasurable": []}
    return {"outcome": "emitted", "sample": sample.as_dict()}


def register_soak_monitor_commands(main: click.Group) -> None:
    """Attach the external evidence monitor to the CLI root."""

    main.add_command(soak_monitor)


__all__ = (
    "MAX_CLEANUP_CELLS",
    "MAX_INVENTORY_OUTPUT_BYTES",
    "MAX_PID_FILE_BYTES",
    "SOAK_EVIDENCE_PATH",
    "SPEND_RECONCILIATION_INVARIANT",
    "measure_local_state_bytes",
    "measure_postgres_database_bytes",
    "measure_process_fd_count",
    "measure_process_rss_bytes",
    "measure_provider_inventory",
    "read_cleanup_complete",
    "read_pid_file",
    "read_spend_reconciled",
    "register_soak_monitor_commands",
    "soak_monitor",
)
