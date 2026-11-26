# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pointing a machine at a control plane somebody else runs.

Every other setup path in Eyvan assumes the plane and the operator are the same
person. This is the team case: one CPU server runs the plane, everybody else
submits from their own checkout, and what they had to type was
``eyvan --url http://host:8080 --token ... use SOME_FLEET`` -- which needs a
fleet name before its owner has ever seen the fleet.

Most of what is pinned here is failure reporting, because the three ways this
goes wrong all look like "it didn't work" and each has a different fix: the
address is wrong, the plane is down, or the plane refused the identity.
"""

from __future__ import annotations

import pytest

from eyvan.sdk.commands.connect import (
    DEFAULT_CONTROL_PLANE_PORT,
    ConnectOutcome,
    ConnectTarget,
    choose_fleet,
    connect,
    render,
    resolve_url,
)

# --------------------------------------------------------------------------- the address


def test_a_bare_host_takes_the_default_port() -> None:
    assert resolve_url("eyvan.internal", None) == f"http://eyvan.internal:{DEFAULT_CONTROL_PLANE_PORT}"


def test_a_port_in_the_address_is_not_overwritten_by_the_default() -> None:
    """Defaulting before reading would silently rewrite host:9000 to 8080 and
    then report the plane unreachable."""

    assert resolve_url("eyvan.internal:9000", None) == "http://eyvan.internal:9000"


def test_an_explicit_port_wins_over_one_in_the_address() -> None:
    assert resolve_url("eyvan.internal:9000", 8080) == "http://eyvan.internal:8080"


def test_a_scheme_is_kept_rather_than_rewritten() -> None:
    """A team behind TLS types https://eyvan.internal, and rewriting that to
    http with a port bolted on sends their credential in the clear."""

    assert resolve_url("https://eyvan.internal", None) == "https://eyvan.internal:8080"
    assert resolve_url("https://eyvan.internal:443", None) == "https://eyvan.internal:443"


def test_an_empty_host_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="host is required"):
        resolve_url("   ", None)


# --------------------------------------------------------------------------- which fleet


def _target(*fleets: str) -> ConnectTarget:
    return ConnectTarget(url="http://head:8080", reachable=True, fleets=fleets, pools=len(fleets))


def test_one_fleet_needs_no_argument() -> None:
    """The whole point: somebody connecting for the first time does not know
    the name yet."""

    assert choose_fleet(_target("research"), None) == ("research", None)


def test_several_fleets_are_named_rather_than_guessed_between() -> None:
    chosen, note = choose_fleet(_target("a", "b"), None)

    assert chosen is None
    assert note is not None and "a, b" in note


def test_a_named_fleet_is_saved_even_before_it_is_applied() -> None:
    """Two people setting up in either order must both work; refusing here
    would make one of them wait on the other."""

    chosen, note = choose_fleet(_target("other"), "research")

    assert chosen == "research"
    assert note is not None and "not applied" in note


def test_a_plane_with_no_fleet_says_what_to_do_next() -> None:
    chosen, note = choose_fleet(_target(), None)

    assert chosen is None
    assert note is not None and "--fleet" in note


# --------------------------------------------------------------------------- what it reports


def test_an_unreachable_plane_names_the_address_and_the_cause() -> None:
    """ "Connection refused" and "refused this identity" are different fixes."""

    rendered = render(ConnectOutcome(target=ConnectTarget("http://head:8080", False, "Connection refused")))

    assert "http://head:8080" in rendered
    assert "Connection refused" in rendered


def test_a_successful_connect_ends_with_what_to_run() -> None:
    outcome = ConnectOutcome(
        target=ConnectTarget(
            "http://head:8080",
            True,
            fleets=("research",),
            pools=6,
            subject="dana",
            roles=("project_submitter",),
        ),
        saved_context="research",
        scope=("acme", "training"),
    )

    rendered = render(outcome)

    assert "you are dana" in rendered
    assert "6 capacity pool" in rendered
    assert "acme/training" in rendered
    assert "eyvan run" in rendered


# --------------------------------------------------------------------------- what it writes


def test_nothing_is_saved_when_the_plane_cannot_be_reached(tmp_path) -> None:
    """A saved context pointing at an address that never answered is a trap:
    every later command fails with no hint that the address is the problem."""

    from eyvan.sdk.commands.state import CliSession, ContextOverrides
    from eyvan.sdk.contexts import FileContextStore

    store = FileContextStore(tmp_path / "contexts.json")
    session = CliSession(
        client=None,  # type: ignore[arg-type]
        identity=type("I", (), {"headers": lambda self: {}})(),  # type: ignore[arg-type]
        context_store=store,
        context_config=store.load(),
        base_url="http://127.0.0.1:8080",
        org_id=None,
        project_id=None,
        overrides=ContextOverrides(),
    )

    outcome = connect(
        session,
        host="down.example.com",
        port=None,
        fleet="research",
        token=None,
        org_id=None,
        project_id=None,
        probe_plane=lambda url, **_: ConnectTarget(url, False, "Connection refused"),
    )

    assert outcome.saved_context is None
    assert store.load().active_context is None
