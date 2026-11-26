# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Every route the operations UI calls has to be one the plane serves.

Nothing checked this in either direction. The Python suite does not read
TypeScript, and the dashboard's own tests mock the API -- so a renamed route, or
a page written against an endpoint that was never added, is green on both sides
and 404s only in a browser, on a head an operator has already provisioned.

The UI is served from the control plane's own address, so there is no version
skew to appeal to: these two are deployed as one thing and can be checked as one
thing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from eyvan.config import EyvanSettings
from eyvan.server.api.app import create_app

#: Every request the dashboard makes goes through one client, and that is worth
#: keeping true: a path built anywhere else is a path this test cannot see.
_CLIENT = Path(__file__).resolve().parents[1] / "dashboard" / "src" / "api" / "client.ts"

#: The prefix `ApiClient` puts in front of every path it is given.
_BASE = "/api/v1"

_TEMPLATE = re.compile(r"\$\{[^}]*\}")
_CANDIDATE = re.compile(r"""[`"'](/[a-z][^`"'\s]*)""")

#: One call site builds its last segment from a variable, so no reader of the
#: source can resolve it. The variable is a closed union in the same file, and
#: naming its members here is what keeps them checked -- a fourth action added
#: to that union without a route fails the test that follows this map.
_DYNAMIC_SEGMENTS: dict[str, tuple[str, ...]] = {
    "/workloads/{}/{}": ("cancel", "checkpoint", "kick", "retry"),
}


def _dashboard_paths() -> set[str]:
    """Return each API path the dashboard client requests, parameters collapsed.

    Query strings are dropped, including the ones assembled into a template:
    what is being checked is which routes exist, and a route is its path.
    """

    source = _CLIENT.read_text(encoding="utf-8")
    paths: set[str] = set()
    for match in _CANDIDATE.finditer(source):
        path = _TEMPLATE.sub("{}", match.group(1)).split("?")[0].rstrip("/")
        # A template glued to a literal segment is a query or a suffix rather
        # than a path segment of its own: `/capacity/quota${suffix}`.
        if path.endswith("{}") and not path.endswith("/{}"):
            path = path[: -len("{}")]
        if not path or path.startswith(_BASE):
            continue
        for expansion in _DYNAMIC_SEGMENTS.get(path, ()):
            paths.add(f"{path[: -len('{}')]}{expansion}")
        if path not in _DYNAMIC_SEGMENTS:
            paths.add(path)
    return paths


def _route_shapes() -> set[tuple[str, ...]]:
    """Return every served route as segments, with parameters collapsed."""

    app = create_app(settings=EyvanSettings(dev_auth_enabled=True, artifact_token_key="k" * 32))
    shapes: set[tuple[str, ...]] = set()
    for route in app.routes:
        path = getattr(route, "path", "")
        if not getattr(route, "methods", None) or not path.startswith(_BASE):
            continue
        segments = tuple("{}" if segment.startswith("{") else segment for segment in path.split("/") if segment)
        shapes.add(segments)
    return shapes


def _shape(path: str) -> tuple[str, ...]:
    return tuple(segment for segment in f"{_BASE}{path}".split("/") if segment)


def test_the_client_is_the_only_place_paths_are_built() -> None:
    """The check above is only as good as this assumption.

    A page that assembles its own URL is invisible to it, so the day that
    happens should fail here rather than silently narrow the coverage.
    """

    root = _CLIENT.parents[2]
    strays = {
        source.relative_to(root)
        for source in root.rglob("*.ts*")
        if source != _CLIENT
        and "test" not in source.name
        and "mockData" not in source.name
        and "node_modules" not in source.parts
        and re.search(r"""[`"'](/(?:workloads|cells|leases|events|budgets|policies|capacity)\b)""", source.read_text())
    }

    assert strays == set()


def test_the_dashboard_reads_at_least_the_pages_it_ships() -> None:
    """A guard over an empty set passes forever; this is the floor under it."""

    assert len(_dashboard_paths()) >= 10


@pytest.mark.parametrize("path", sorted(_dashboard_paths()))
def test_every_path_the_dashboard_requests_is_served(path: str) -> None:
    """One case per path, so a failure names the route rather than the count."""

    assert _shape(path) in _route_shapes(), f"the dashboard requests {_BASE}{path}, which this plane does not serve"


def test_the_matcher_rejects_a_route_the_plane_does_not_serve() -> None:
    """Otherwise every assertion above could be passing on a matcher that says yes.

    Both shapes matter: a path that exists nowhere, and one whose parameter count
    differs from a real route by a single segment -- which is what a renamed or
    re-nested endpoint actually looks like.
    """

    shapes = _route_shapes()

    assert _shape("/workloads/{}/there-is-no-such-thing") not in shapes
    assert _shape("/workloads/{}/{}/logs") not in shapes
    assert _shape("/cells/{}/{}") not in shapes
    # And the control: a real one still matches, so the rejections above are not
    # simply a matcher that says no to everything.
    assert _shape("/workloads/{}/logs") in shapes


def test_a_dynamic_segment_that_gains_a_member_must_be_declared() -> None:
    """The map is only honest while it matches the union it stands in for."""

    source = _CLIENT.read_text(encoding="utf-8")
    declared = set(_DYNAMIC_SEGMENTS["/workloads/{}/{}"])
    union = re.search(r"action:\s*((?:\"[a-z]+\"\s*\|?\s*)+)", source)

    assert union is not None, "the workload action union moved; the expansion map cannot be trusted"
    assert set(re.findall(r'"([a-z]+)"', union.group(1))) == declared
