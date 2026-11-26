# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Obtaining a real PostgreSQL for the integration suite.

The whole of ``test_postgres_*.py`` exists because nothing in this package had
ever connected to PostgreSQL. The dev extra pins ``aiosqlite``,
``test_migrations`` runs Alembic in offline mode and asserts emitted DDL strings
without executing them, and ``test_persistence_leader`` models advisory locks in
Python. SERIALIZABLE conflict retry, ``pg_try_advisory_lock``, the ``pg_locks``
ownership re-check, ``FOR UPDATE SKIP LOCKED``, BIGSERIAL out-of-order commit
visibility and the migration chain itself are all PostgreSQL semantics, and all
of them were unexecuted on the one backend production uses.

Two ways in, in order:

* ``EYVAN_TEST_POSTGRES_URL`` -- a database someone else is running. This is the
  CI path, pointed at a service container.
* ``EYVAN_TEST_POSTGRES_BINDIR`` or ``initdb`` on ``PATH`` -- start a throwaway
  cluster in a temporary directory for the session and tear it down after.

With neither, the suite skips -- unless ``EYVAN_REQUIRE_POSTGRES`` is set, which
turns the skip into a failure. CI sets it, because an integration suite that can
silently skip is one that silently stops running.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest

_STARTUP_TIMEOUT_S = 30.0


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _binary(name: str) -> str | None:
    """Locate one PostgreSQL binary, preferring an explicitly configured directory."""

    bindir = os.environ.get("EYVAN_TEST_POSTGRES_BINDIR")
    if bindir:
        candidate = Path(bindir) / name
        return str(candidate) if candidate.is_file() else None
    return shutil.which(name)


def _unavailable(reason: str) -> None:
    """Skip, or fail when the caller declared a database mandatory."""

    if os.environ.get("EYVAN_REQUIRE_POSTGRES"):
        raise AssertionError(
            f"EYVAN_REQUIRE_POSTGRES is set but no PostgreSQL is usable: {reason}. "
            "Set EYVAN_TEST_POSTGRES_URL or EYVAN_TEST_POSTGRES_BINDIR."
        )
    pytest.skip(f"no PostgreSQL available: {reason}")


class _TemporaryCluster:
    """A throwaway PostgreSQL cluster owned by one test session."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.data = directory / "data"
        self.port = _free_port()
        self._started = False

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://postgres@127.0.0.1:{self.port}/postgres"

    def start(self) -> None:
        initdb, pg_ctl = _binary("initdb"), _binary("pg_ctl")
        if initdb is None or pg_ctl is None:
            _unavailable("initdb/pg_ctl not found")
        assert initdb is not None and pg_ctl is not None
        password_file = self.directory / "pwfile"
        password_file.write_text("postgres", encoding="utf-8")
        subprocess.run(
            [initdb, "-D", str(self.data), "-U", "postgres", "--auth=trust", "--pwfile", str(password_file)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                pg_ctl,
                "-D",
                str(self.data),
                "-l",
                str(self.directory / "postgres.log"),
                # Loopback only, on a port nobody else claimed: a test cluster
                # must never be reachable from off the machine, and must never
                # collide with a database an operator is actually using.
                "-o",
                f"-p {self.port} -c listen_addresses=127.0.0.1 -c fsync=off -c synchronous_commit=off",
                "-w",
                "-t",
                str(int(_STARTUP_TIMEOUT_S)),
                "start",
            ],
            check=True,
            capture_output=True,
        )
        self._started = True

    def stop(self) -> None:
        pg_ctl = _binary("pg_ctl")
        if self._started and pg_ctl is not None:
            subprocess.run([pg_ctl, "-D", str(self.data), "-m", "immediate", "-w", "stop"], capture_output=True)
        self._started = False


def _wait_for(url: str) -> None:
    """Block until the server accepts connections, so a slow start is not a flake.

    A plain socket probe rather than a SQLAlchemy round trip: this fixture is
    resolved both outside and inside a running event loop -- the second happens
    whenever an async fixture asks for it through ``getfixturevalue`` -- and an
    ``asyncio.run`` in that case raises instead of waiting.
    """

    parsed = urlsplit(url)
    host, port = parsed.hostname or "127.0.0.1", parsed.port or 5432
    deadline = time.monotonic() + _STARTUP_TIMEOUT_S
    last: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError as error:
            last = error
            time.sleep(0.25)
    raise AssertionError(f"PostgreSQL at {host}:{port} never became reachable: {last}")


@pytest.fixture(scope="session")
def postgres_url() -> str:
    """Return a URL for a real PostgreSQL, starting one if nobody supplied it."""

    configured = os.environ.get("EYVAN_TEST_POSTGRES_URL")
    if configured:
        _wait_for(configured)
        yield configured
        return

    if _binary("initdb") is None:
        _unavailable("initdb/pg_ctl not found")

    with tempfile.TemporaryDirectory(prefix="eyvan-pg-") as directory:
        cluster = _TemporaryCluster(Path(directory))
        cluster.start()
        try:
            _wait_for(cluster.url)
            yield cluster.url
        finally:
            cluster.stop()
