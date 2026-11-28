# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A wait outlives the control plane it is watching.

A TPU sits in a provider queue for many minutes, and over that window somebody
restarts the control plane, a proxy drops a connection, or a rolling deploy
moves it. The workload is durable across all three -- it is a row in a database
and a queued resource at the provider, neither of which cares that a client
stopped looking.

Killing the caller's script on the first failed poll therefore ends the
*watching*, not the job: the run continues on hardware with nobody reading it,
which is the worst of the available outcomes. It happened for real, to a v5p-16
that stayed queued while the script it was submitted from died with a traceback.
"""

from __future__ import annotations

import pytest

from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.client.sync import UNREACHABLE_GRACE_S, EyvanClient
from eyvan.sdk.client.transport import EyvanUnavailableError


class _Plane:
    """A plane that is unreachable for the first ``outages`` polls."""

    def __init__(self, outages: int, *, final_state: str = "succeeded") -> None:
        self.outages = outages
        self.final_state = final_state
        self.polls = 0

    def __call__(self, workload_id: str):
        self.polls += 1
        if self.polls <= self.outages:
            raise EyvanUnavailableError(ControlPlaneEndpoint("http://127.0.0.1:8080"), "GET")
        return _Workload(self.final_state)


class _Workload:
    def __init__(self, state: str) -> None:
        self.state = state


class _AsyncPlane(_Plane):
    """The same scripted plane, awaiting like ``AsyncEyvanClient`` does."""

    async def __call__(self, workload_id: str):
        return super().__call__(workload_id)


def _client(plane: _Plane, sleeps: list[float]) -> EyvanClient:
    client = EyvanClient.__new__(EyvanClient)
    client.get_workload = plane  # type: ignore[method-assign]
    return client


def test_a_restart_mid_wait_does_not_kill_the_caller(monkeypatch) -> None:
    """The case that happened: the plane was restarted while a v5p-16 was
    queued, and the script died holding a job that was still running."""

    plane = _Plane(outages=3)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    result = EyvanClient.wait_workload(_client(plane, []), "wl-1", poll_interval=0.01)

    assert result.state == "succeeded"
    assert plane.polls == 4, "three failed polls were forgiven, the fourth answered"


def test_a_plane_that_is_genuinely_gone_is_still_reported(monkeypatch) -> None:
    """Forgiving forever would turn a dead plane into a hung script, which is
    the failure this must not trade for."""

    plane = _Plane(outages=10_000)
    clock = iter([0.0, 0.0, UNREACHABLE_GRACE_S + 1])

    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    monkeypatch.setattr("time.monotonic", lambda: next(clock, UNREACHABLE_GRACE_S * 10))

    with pytest.raises(EyvanUnavailableError):
        EyvanClient.wait_workload(_client(plane, []), "wl-1", poll_interval=0.01)


def test_the_grace_window_restarts_after_a_successful_poll(monkeypatch) -> None:
    """Two separate blips an hour apart are not one long outage, and treating
    them as one would fail a wait that never actually lost the plane."""

    class _Flaky:
        def __init__(self) -> None:
            self.polls = 0

        def __call__(self, workload_id: str):
            self.polls += 1
            if self.polls in {2, 4}:
                raise EyvanUnavailableError(ControlPlaneEndpoint("http://127.0.0.1:8080"), "GET")
            return _Workload("running" if self.polls < 5 else "succeeded")

    plane = _Flaky()
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    result = EyvanClient.wait_workload(_client(plane, []), "wl-1", poll_interval=0.01)  # type: ignore[arg-type]

    assert result.state == "succeeded"


# --------------------------------------------------------------------------- the async twin


async def _instant_sleep(_seconds: float) -> None:
    return None


async def test_the_async_wait_forgives_a_restart_too(monkeypatch) -> None:
    """A notebook or service waiter dies exactly as a script does on the first
    dropped poll; the async client must carry the same grace the sync one does."""

    from eyvan.sdk.client.async_client import AsyncEyvanClient

    plane = _AsyncPlane(outages=3)
    monkeypatch.setattr("eyvan.sdk.client.async_client.asyncio.sleep", _instant_sleep)

    client = AsyncEyvanClient.__new__(AsyncEyvanClient)
    client.get_workload = plane  # type: ignore[method-assign]

    result = await AsyncEyvanClient.wait_workload(client, "wl-1", poll_interval=0.01)

    assert result.state == "succeeded"
    assert plane.polls == 4, "three failed polls were forgiven, the fourth answered"


async def test_the_async_wait_still_reports_a_plane_that_is_gone(monkeypatch) -> None:
    """The grace is bounded, or a dead plane trades a traceback for a hang."""

    from eyvan.sdk.client.async_client import AsyncEyvanClient

    plane = _AsyncPlane(outages=10_000)
    clock = iter([0.0, 0.0, UNREACHABLE_GRACE_S + 1])

    monkeypatch.setattr("eyvan.sdk.client.async_client.asyncio.sleep", _instant_sleep)
    monkeypatch.setattr("time.monotonic", lambda: next(clock, UNREACHABLE_GRACE_S * 10))

    client = AsyncEyvanClient.__new__(AsyncEyvanClient)
    client.get_workload = plane  # type: ignore[method-assign]

    with pytest.raises(EyvanUnavailableError):
        await AsyncEyvanClient.wait_workload(client, "wl-1", poll_interval=0.01)
