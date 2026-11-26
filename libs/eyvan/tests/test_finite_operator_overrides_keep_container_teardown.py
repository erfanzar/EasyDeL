# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Finite operator overrides keep container teardown and relaunch bounded."""

from __future__ import annotations

import fcntl
import math
from pathlib import Path

import pytest
from test_container_executor import _context, _executor, _record, _spec
from test_stop_survives_a_slow_container_teardown import SlowTeardownDocker

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.worker.runtime import container


@pytest.mark.parametrize(
    ("name", "default"),
    [
        ("EYVAN_CONTAINER_STOP_SETTLE_S", container._CONTAINER_STOP_SETTLE_S),
        ("EYVAN_CONTAINER_STOP_POLL_S", container._CONTAINER_STOP_SETTLE_POLL_S),
    ],
)
@pytest.mark.parametrize("override", ["inf", "+inf", "1e309", "nan", "-inf", "-1", "invalid"])
def test_invalid_timing_override_uses_finite_default(
    monkeypatch: pytest.MonkeyPatch, name: str, default: float, override: str
) -> None:
    monkeypatch.setenv(name, override)

    assert container._env_seconds(name, default) == default


@pytest.mark.parametrize("override", ["0", "0.25", "5"])
def test_valid_timing_override_is_preserved(monkeypatch: pytest.MonkeyPatch, override: str) -> None:
    monkeypatch.setenv("EYVAN_CONTAINER_STOP_SETTLE_S", override)

    assert container._env_seconds("EYVAN_CONTAINER_STOP_SETTLE_S", 30.0) == float(override)


@pytest.mark.parametrize("name", ["EYVAN_CONTAINER_STOP_SETTLE_S", "EYVAN_CONTAINER_STOP_POLL_S"])
@pytest.mark.parametrize("override", ["inf", "+inf", "1e309"])
def test_nonfinite_override_cannot_hold_stop_and_relaunch_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, override: str
) -> None:
    daemon = SlowTeardownDocker(removing_inspections=10_000, images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    monkeypatch.setenv("EYVAN_CONTAINER_STOP_SETTLE_S", "30")
    monkeypatch.setenv("EYVAN_CONTAINER_STOP_POLL_S", "0.5")
    monkeypatch.setenv(name, override)
    elapsed = 0.0
    sleeps: list[float] = []

    def monotonic() -> float:
        return elapsed

    def sleep(seconds: float) -> None:
        nonlocal elapsed
        # Test-only safety guards: broken production must fail, not hang pytest.
        assert math.isfinite(seconds), "production requested a nonfinite sleep"
        assert len(sleeps) < 61, "test safety sentinel: production exceeded its finite deadline"
        sleeps.append(seconds)
        elapsed += seconds

    monkeypatch.setattr(container.time, "monotonic", monotonic)
    monkeypatch.setattr(container.time, "sleep", sleep)

    with pytest.raises(RuntimeError, match="container is still active"):
        executor.stop(remote_id, _context())

    assert elapsed == container._CONTAINER_STOP_SETTLE_S
    assert sleeps == [container._CONTAINER_STOP_SETTLE_POLL_S] * 60
    assert _record(executor, remote_id)["status"] == ExecutionWorkloadStatus.INDETERMINATE.value
    assert any(command[:2] == ("docker", "stop") for command in daemon.commands)

    # The actual stop lock is no longer held, so retry/relaunch can progress.
    lock_path = executor._record_path(remote_id).with_suffix(".retire-stop.lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    daemon._remaining_removing = 0
    executor.stop(remote_id, _context())
    assert executor.status(remote_id).status is ExecutionWorkloadStatus.STOPPED
