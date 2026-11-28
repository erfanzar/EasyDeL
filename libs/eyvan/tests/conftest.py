# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fixtures shared across the suite.

Only the PostgreSQL session fixture lives here. Everything else in this suite
builds its own harness in the module that needs it, which keeps each test file
readable on its own; a real database is an exception because starting one per
module would cost seconds per file. The other exception is daemon teardown:
any test file that isolates HOME and can start a real local daemon stops, in
teardown, everything its isolated profile recorded, using the one helper here
so no file reinvents (or skips) it.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from postgres_fixture import postgres_url

__all__ = ("postgres_url", "stop_local_daemons_recorded_under")

# Bounded smoke coverage, not an alternative to the full integration gate.
# Keep this selection explicit so adding slow integration work elsewhere does
# not silently turn the default feedback loop into a multi-minute run.
_FAST_MODULES = frozenset(
    {
        "test_scheduler.py",
        "test_config_settings.py",
        "test_provider_boundary.py",
        "test_fleet_config.py",
        "test_elastic_packing_preserves_feasible_subsets.py",
        "test_dashboard_session_expiry.py",
        "test_dashboard_cookie_csrf.py",
        "test_dashboard_callback_input.py",
    }
)


@pytest.fixture(autouse=True)
def _committed_records_are_never_edited_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any memory/local-JSON transaction that edits a committed record in place.

    The state repositories share committed records with each transaction rather
    than deep-copying them; that is only sound while nothing mutates a record's
    nested dicts. Every test is therefore also a check of that contract.
    """

    from eyvan.server.persistence.state import StateRepository

    monkeypatch.setattr(StateRepository, "verify_record_immutability", True)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Keep extended soak counts explicit instead of slowing every PR run."""

    parser.addoption("--soak-iterations", type=int, default=10, help="repository soak cycles (1..10000)")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Label the documented smoke subset without skipping any full-suite test."""

    for item in items:
        if item.path.name in _FAST_MODULES:
            item.add_marker(pytest.mark.fast)
        for marker in item.iter_markers(name="quarantine"):
            metadata = marker.kwargs
            try:
                if marker.args or set(metadata) != {"issue", "owner", "until"}:
                    raise ValueError("requires issue, owner and until keyword arguments")
                if not all(isinstance(value, str) and value.strip() for value in metadata.values()):
                    raise ValueError("metadata must contain non-empty strings")
                if not re.fullmatch(r"https://github\.com/[^/]+/[^/]+/issues/[1-9][0-9]*", metadata["issue"]):
                    raise ValueError("issue must name a GitHub tracking issue")
                remaining = (date.fromisoformat(metadata["until"]) - datetime.now(UTC).date()).days
                if not 0 <= remaining <= 14:
                    raise ValueError("expiry must be today through 14 days from today (UTC)")
            except (TypeError, ValueError) as error:
                raise pytest.UsageError(f"{item.nodeid}: invalid quarantine: {error}") from error


def stop_local_daemons_recorded_under(home: Path) -> None:
    """Stop local Eyvan daemons an isolated profile recorded, then name survivors.

    A daemon started under an isolated home outlives the test process, keeps
    its port bound with a credential from a home pytest is about to delete,
    and the next real ``eyvan`` command against that port answers 401 from a
    plane nobody owns. The PID records are owner-only by construction, so
    anything readable here was written during this test run. What the
    supervisors can prove they own, they stop -- control planes and the cell
    agents the native GCP runtime records alike; what they cannot prove (a
    foreign command under a recycled PID) they refuse to signal, and the
    liveness check then fails the test naming the survivor instead of
    silently passing.
    """

    from eyvan.capacity.providers.gcp.runtime.supervisor import LocalAgentSupervisor
    from eyvan.server.local.daemon import (
        LocalDaemonError,
        LocalDaemonSupervisor,
        process_is_alive,
        read_owner_only_record,
    )

    survivors: list[str] = []
    for record_path in sorted(home.glob(".eyvan/*.pid")):
        record = read_owner_only_record(record_path)
        if record is None:
            continue
        try:
            if record_path.name.startswith("cell-agent-"):
                LocalAgentSupervisor(endpoint=record.endpoint).stop()
            else:
                LocalDaemonSupervisor().stop(record.endpoint)
        except LocalDaemonError:
            pass
        if process_is_alive(record.pid):
            survivors.append(f"pid {record.pid}, {record.endpoint.url}")
    if survivors:
        pytest.fail(
            f"local Eyvan daemons ({'; '.join(survivors)}) outlived the test; "
            "record them for teardown or patch their ensure_* helpers out"
        )
