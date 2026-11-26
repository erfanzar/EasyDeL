# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""`eyvan status` with no argument finds the fleet, or says which ones exist.

Requiring the name made the first command a new operator runs exit with a usage
error, and the two most likely reasons to run it are that a fleet is misbehaving
and that the caller is not certain which fleets exist.
"""

from __future__ import annotations

import typing as tp

import pytest
from click.testing import CliRunner

from eyvan.sdk.cli import main


class _Client:
    """Answers the discovery read and the fleet read, recording both."""

    def __init__(self, fleets: list[str] | Exception, status: dict[str, tp.Any] | None = None) -> None:
        self._fleets = fleets
        self._status = status or {"fleet": "research", "pools": [], "cells": []}
        self.requested: list[str] = []

    def request(self, method: str, path: str, **kwargs: tp.Any) -> tp.Any:
        del method, kwargs
        self.requested.append(path)
        if path == "/api/v1/fleets":
            if isinstance(self._fleets, Exception):
                raise self._fleets
            return self._fleets
        return self._status


class _Scope:
    org_id = "acme"
    project_id = "research"


class _Session:
    def __init__(self, client: _Client) -> None:
        self.client = client
        self.autoscale_cache_dir = None

    def workload_scope(self) -> _Scope:
        return _Scope()


def _run(client: _Client, *args: str):
    return CliRunner().invoke(main.commands["status"], list(args), obj=_Session(client))


def test_one_running_fleet_is_reported_without_being_named() -> None:
    """The whole point: the common case needs no argument."""

    client = _Client(["research"])

    result = _run(client)

    assert result.exit_code == 0, result.output
    assert "/api/v1/fleets/research" in client.requested


def test_several_running_fleets_are_listed_rather_than_guessed_between() -> None:
    """Choosing for the caller reports on a fleet they did not mean."""

    client = _Client(["alpha", "beta", "gamma"])

    result = _run(client)

    assert result.exit_code != 0
    for name in ("alpha", "beta", "gamma"):
        assert name in result.output
    assert "eyvan status alpha" in result.output
    assert not [path for path in client.requested if path.startswith("/api/v1/fleets/")]


def test_no_running_fleet_says_how_to_make_one() -> None:
    """An empty fleet list is an answer, and it has an obvious next step."""

    result = _run(_Client([]))

    assert result.exit_code != 0
    assert "eyvan up" in result.output


def test_an_unreachable_control_plane_does_not_become_a_traceback() -> None:
    """Discovery is best effort: failing it must degrade to the old requirement."""

    result = _run(_Client(RuntimeError("connection refused")))

    assert result.exit_code != 0
    assert "no running fleet" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("named", ["research", "acme-fleet"])
def test_naming_a_fleet_still_skips_discovery_entirely(named: str) -> None:
    """An explicit name must not pay for a read it does not need."""

    client = _Client(["research", "other"])

    _run(client, named)

    assert "/api/v1/fleets" not in client.requested
    assert f"/api/v1/fleets/{named}" in client.requested
