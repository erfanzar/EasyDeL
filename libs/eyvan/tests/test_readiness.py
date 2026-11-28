# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Telling "restart me" apart from "do not send me traffic".

The deployment probed ``/healthz`` for *both*, and ``/healthz`` returned a value
computed at startup. So a replica whose database connection had died reported
200 forever and stayed in the Service's endpoints, absorbing traffic it could
only fail.

The obvious fix -- make ``/healthz`` check the database -- would have been
worse. Liveness failure means *restart me*, and a liveness probe that depends on
a shared database restarts every replica the moment that database blips: a
recoverable outage becomes a total one, and then stays that way because every
replica reconnects at once, forever.

So the two questions get two endpoints, and the tests below assert the
separation rather than only the checks.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.server.api.app import create_app
from eyvan.server.persistence.memory import InMemoryRepository


class _DeadStore(InMemoryRepository):
    """A repository that opened fine and has since lost its database."""

    async def ping(self) -> None:
        raise ConnectionError("connection to server was lost")


def _client(tmp_path, repository=None) -> TestClient:
    return TestClient(
        create_app(
            settings=EyvanSettings(
                dev_auth_enabled=True,
                artifact_token_key="k" * 32,
                artifact_store_uri=str(tmp_path / "artifacts"),
            ),
            repository=repository,
        )
    )


# --------------------------------------------------------------------------- readiness


def test_a_healthy_replica_is_ready(tmp_path) -> None:
    """The probe must not fail the configuration it exists to admit."""

    response = _client(tmp_path).get("/readyz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_a_replica_that_cannot_reach_its_database_reports_unready(tmp_path) -> None:
    """The gap: it used to report 200 forever and keep absorbing traffic.

    Nothing about the process is wrong -- it is running, it answers, it simply
    cannot serve -- which is exactly the state readiness exists to describe.
    """

    response = _client(tmp_path, _DeadStore()).get("/readyz")

    assert response.status_code == 503


def test_an_unready_replica_still_names_itself(tmp_path) -> None:
    """An operator reading a failing probe wants to know *which* replica this is.

    Returning a bare 503 with no body makes the probe output useless for
    anything except the scheduler that reads the status code.
    """

    body = _client(tmp_path, _DeadStore()).get("/readyz").json()

    assert body["status"] == "unavailable"
    assert body["controller"]


# --------------------------------------------------------------------------- liveness


def test_liveness_stays_up_when_the_database_is_down(tmp_path) -> None:
    """The property worth more than the readiness check itself.

    A liveness probe that consulted the database would restart every replica
    simultaneously on a database blip, then keep them cycling as each restart
    stampedes reconnections into a server already struggling. The blast radius
    of getting this backwards is the entire control plane.
    """

    response = _client(tmp_path, _DeadStore()).get("/healthz")

    assert response.status_code == 200


def test_liveness_never_touches_the_store(tmp_path) -> None:
    """Asserted by observation, not by reading the handler.

    A future change that adds a database read to ``/healthz`` compiles, passes
    every other test, and reintroduces the failure mode silently.
    """

    pinged: list[int] = []

    class _CountingStore(InMemoryRepository):
        async def ping(self) -> None:
            pinged.append(1)

    client = _client(tmp_path, _CountingStore())
    client.get("/healthz")
    client.get("/api/v1/health")

    assert pinged == []

    client.get("/readyz")

    assert pinged == [1]


# --------------------------------------------------------------------------- the store


async def test_the_in_memory_store_is_always_reachable() -> None:
    """It has no remote to lose, so unreadiness would be a lie."""

    await InMemoryRepository().ping()
