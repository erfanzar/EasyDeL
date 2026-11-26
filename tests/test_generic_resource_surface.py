# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""`eyvan list` and `eyvan get`: one grammar for every kind of record.

Each noun had its own command with its own flags, which is fine to read and
miserable to script. The risk in adding a uniform surface is that it becomes a
second, drifting map of the API -- a resource naming a route nobody serves, or
a filter the server ignores. These tests hold it against the real routes.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from eyvan.sdk import cli
from eyvan.sdk.commands.resources import RESOURCES, resolve


def _app():
    """The real application, so the routes compared against are the served ones."""

    from eyvan.config import EyvanSettings
    from eyvan.server.api.app import create_app

    return create_app(settings=EyvanSettings(dev_auth_enabled=True))


class _Plane:
    """Records what the CLI asked the control plane for."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method, path, *, params=None, json=None, headers=None):
        del json, headers
        self.calls.append((method, path, dict(params or {})))
        return []

    def request_text(self, method, path, *, params=None, headers=None):
        del method, path, params, headers
        return ""

    def replace_identity_headers(self, headers) -> None:
        del headers


def _invoke(monkeypatch, plane: _Plane, *args: str):
    monkeypatch.setattr(cli, "EyvanClient", lambda *_a, **_k: plane)
    return CliRunner().invoke(cli.main, ["--user", "alice", "--org", "acme", "--project", "research", *args])


# --------------------------------------------------------------------------- the vocabulary


def test_every_declared_resource_names_a_route_the_server_serves() -> None:
    """The failure this surface invites: a noun pointing at nothing.

    A vocabulary maintained beside the API drifts the moment a route is renamed,
    and the symptom is a 404 from a command that looks correct.
    """

    app = _app()
    served = {route.path for route in app.routes if hasattr(route, "path")}

    for resource in RESOURCES:
        assert resource.list_path in served, f"{resource.name} lists a route the server does not serve"
        if resource.get_path is not None:
            expected = resource.get_path.replace("{id}", "{")
            assert any(path.startswith(expected.rsplit("{", 1)[0]) and path.count("{") == 1 for path in served), (
                f"{resource.name} reads one record from a route the server does not serve"
            )


def test_every_declared_filter_is_a_query_parameter_the_route_accepts() -> None:
    """A filter the server ignores narrows nothing and says it did."""

    app = _app()
    accepted: dict[str, set[str]] = {}
    for route in app.routes:
        if not hasattr(route, "path") or not hasattr(route, "dependant"):
            continue
        accepted.setdefault(route.path, set()).update(param.alias for param in route.dependant.query_params)

    for resource in RESOURCES:
        unknown = set(resource.filters) - accepted.get(resource.list_path, set())
        assert not unknown, f"{resource.name} declares filters {sorted(unknown)} its route does not accept"


def test_listing_with_no_kind_prints_the_vocabulary(monkeypatch) -> None:
    """The first thing anyone types should tell them what else they can type."""

    plane = _Plane()

    result = _invoke(monkeypatch, plane, "list")

    assert result.exit_code == 0, result.output
    for resource in RESOURCES:
        assert resource.name in result.output
    assert plane.calls == [], "printing the vocabulary must not call the control plane"


def test_an_unknown_kind_is_refused_with_the_vocabulary(monkeypatch) -> None:
    """A bare "unknown resource" leaves the next guess no better informed."""

    result = _invoke(monkeypatch, _Plane(), "list", "gpus")

    assert result.exit_code != 0
    assert "workloads" in result.output and "cells" in result.output


@pytest.mark.parametrize("spelling", ["workload", "workloads", "job", "jobs", "WORKLOADS"])
def test_a_resource_answers_to_the_names_people_actually_type(monkeypatch, spelling: str) -> None:
    """Singular, plural and the word Ray uses all reach the same records."""

    assert resolve(spelling).name == "workloads"


# --------------------------------------------------------------------------- reading


def test_listing_a_scoped_resource_carries_the_session_scope(monkeypatch) -> None:
    """Without the scope the route refuses, and the user never named one."""

    plane = _Plane()

    result = _invoke(monkeypatch, plane, "list", "workloads")

    assert result.exit_code == 0, result.output
    method, path, params = plane.calls[0]
    assert (method, path) == ("GET", "/api/v1/workloads")
    assert params["org_id"] == "acme"
    assert params["project_id"] == "research"


def test_listing_an_unscoped_resource_sends_no_scope(monkeypatch) -> None:
    """Cells belong to the fleet, not to a project, and asking scoped is wrong."""

    plane = _Plane()

    result = _invoke(monkeypatch, plane, "list", "cells")

    assert result.exit_code == 0, result.output
    _method, path, params = plane.calls[0]
    assert path == "/api/v1/cells"
    assert params == {}


def test_a_filter_reaches_the_route_as_a_query_parameter(monkeypatch) -> None:
    plane = _Plane()

    result = _invoke(monkeypatch, plane, "list", "workloads", "--filter", "state=running")

    assert result.exit_code == 0, result.output
    assert plane.calls[0][2]["state"] == "running"


def test_a_filter_the_resource_does_not_accept_is_refused_by_name(monkeypatch) -> None:
    """Forwarding it would silently return everything and look filtered."""

    plane = _Plane()

    result = _invoke(monkeypatch, plane, "list", "cells", "--filter", "state=running")

    assert result.exit_code != 0
    assert "state" in result.output
    assert plane.calls == [], "nothing may reach the control plane"


def test_getting_one_record_addresses_it_by_id(monkeypatch) -> None:
    plane = _Plane()

    result = _invoke(monkeypatch, plane, "get", "workloads", "wl-1")

    assert result.exit_code == 0, result.output
    _method, path, _params = plane.calls[0]
    assert path == "/api/v1/workloads/wl-1"


def test_a_resource_with_no_single_record_says_so(monkeypatch) -> None:
    """Events are a stream, not a set of addressable rows."""

    plane = _Plane()

    result = _invoke(monkeypatch, plane, "get", "events", "evt-1")

    assert result.exit_code != 0
    assert "eyvan list events" in result.output
    assert plane.calls == []


def test_output_is_the_same_json_every_other_command_emits(monkeypatch) -> None:
    """The point of a uniform surface is that one parser reads all of it.

    `--json` because the listing renders a table for a person by default. Both
    spellings are one fetch; only this one is meant to be parsed.
    """

    plane = _Plane()

    result = _invoke(monkeypatch, plane, "list", "cells", "--json")

    assert json.loads(result.stdout) == []


def test_an_empty_listing_says_so_rather_than_printing_an_empty_table(monkeypatch) -> None:
    """`[]` is a correct answer and an unreadable one. A blank table reads as a
    rendering bug; a sentence reads as the fleet holding nothing."""

    result = _invoke(monkeypatch, _Plane(), "list", "cells")

    assert "No cells." in result.stdout
    assert "[]" not in result.stdout
