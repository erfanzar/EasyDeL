# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A process on a control-plane host finds its plane without being told where it is."""

from __future__ import annotations

from pathlib import Path

import pytest

from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.execution import init
from eyvan.sdk.node import (
    NodeContextError,
    discover_node_context,
    discover_node_token,
    discover_node_token_or_warn,
    write_node_context,
)

_INTERNAL_URL = "http://10.128.0.7:8080"


def _head_environment(directory: Path, *, public_url: str = _INTERNAL_URL, token: str | None = "head-token") -> Path:
    """Write the systemd EnvironmentFile shape the installer produces."""

    lines = [
        "EYVAN_AGENT_SHARED_SECRET=agent-secret",
        "EYVAN_METRICS_TOKEN=metrics-token",
        f"EYVAN_PUBLIC_URL={public_url}",
        "EYVAN_REPOSITORY_BACKEND=postgres",
    ]
    if token is not None:
        lines.append(f"EYVAN_BOOTSTRAP_TOKEN={token}")
    path = directory / "head.env"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _node(directory: Path, **kwargs: object) -> dict[str, str]:
    """Write a node discovery file and return the environment that selects it."""

    environment_path = _head_environment(directory, **kwargs)  # type: ignore[arg-type]
    node_path = directory / "node.json"
    write_node_context(environment_path, node_path, fleet="research")
    return {
        "EYVAN_NODE_FILE": str(node_path),
        # Pinned away from the real ``~/.config/eyvan/config.json``. Without it
        # these tests read whatever context the machine running them happens to
        # have selected -- which is a faithful demonstration that an operator's
        # own context outranks node discovery, and useless as a test.
        "EYVAN_CONFIG_FILE": str(directory / "operator-config.json"),
    }


def _wire(context) -> tuple[str, str | None]:  # type: ignore[no-untyped-def]
    """Return the endpoint and Authorization header the SDK would actually use.

    Reaches the underlying httpx client because that is the only place the two
    resolved values meet; asserting on the inputs instead would let a wiring
    mistake between them pass.
    """

    http = context.submitter.client._client
    return str(http.base_url), http.headers.get("Authorization")


def test_a_process_on_a_head_finds_its_control_plane_with_no_environment(tmp_path: Path) -> None:
    """The Ray property: the thing that stood the node up wrote the answer down."""

    environ = _node(tmp_path)

    context = init(environ=environ)

    assert context.fleet == "research"
    url, authorization = _wire(context)
    assert url.rstrip("/") == _INTERNAL_URL
    # And the credential, from the same discovery, so `python train.py` on a head
    # needs neither EYVAN_URL nor EYVAN_TOKEN.
    assert authorization == "Bearer head-token"


def test_node_discovery_never_starts_a_second_control_plane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trap this file exists to disarm, asserted rather than argued.

    Without a node file the endpoint falls back to ``http://127.0.0.1:8080``,
    and loopback is the signal that means "local development, start a plane".
    On a head that would mean starting a second control plane on the machine
    already running one -- today it is saved only by a 250ms health probe
    against the real server, which is a race, not a design.

    A node file carries the internal VPC address, so the branch is not entered
    at all.
    """

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the SDK tried to start a local control plane on a head")

    monkeypatch.setattr("eyvan.sdk.execution.ensure_loopback_control_plane", refuse)

    init(environ=_node(tmp_path))


def test_a_node_context_naming_loopback_is_refused_when_written(tmp_path: Path) -> None:
    """A node file saying 127.0.0.1 would re-arm the trap while looking like the fix."""

    environment_path = _head_environment(tmp_path, public_url="http://127.0.0.1:8080")

    with pytest.raises(NodeContextError, match="loopback"):
        write_node_context(environment_path, tmp_path / "node.json", fleet="research")


def test_a_head_environment_naming_no_public_url_is_refused(tmp_path: Path) -> None:
    """Guessing an address for a control plane is worse than declining to."""

    path = tmp_path / "head.env"
    path.write_text("EYVAN_REPOSITORY_BACKEND=postgres\n", encoding="utf-8")

    with pytest.raises(NodeContextError, match="EYVAN_PUBLIC_URL"):
        write_node_context(path, tmp_path / "node.json", fleet="research")


def test_the_node_context_round_trips_through_its_own_reader(tmp_path: Path) -> None:
    """One module writes and reads it, so the format cannot drift between them."""

    environ = _node(tmp_path)

    discovered = discover_node_context(environ)

    assert discovered is not None
    assert (discovered.fleet, discovered.base_url) == ("research", _INTERNAL_URL)
    # World-readable on purpose: a workload runs as another account and must be
    # able to answer the same question. It holds no secret to protect.
    assert (tmp_path / "node.json").stat().st_mode & 0o777 == 0o644


def test_an_ordinary_machine_discovers_nothing(tmp_path: Path) -> None:
    """None for a laptop is the point: discovery fills a vacuum, it never overrules."""

    assert discover_node_context({"EYVAN_NODE_FILE": str(tmp_path / "absent.json")}) is None


def test_an_explicit_choice_outranks_node_discovery(tmp_path: Path) -> None:
    """A URL someone typed on this machine beats the one the machine inferred."""

    environ = {**_node(tmp_path), "EYVAN_URL": "http://10.128.0.99:8080"}

    context = init(environ=environ)

    url, authorization = _wire(context)
    assert url.rstrip("/") == "http://10.128.0.99:8080"
    # A token discovered for a different plane must not follow the URL over.
    assert authorization is None


def test_the_bootstrap_token_is_read_in_place_and_copied_nowhere(tmp_path: Path) -> None:
    """The credential already exists on this machine; a second copy is the risk."""

    environ = _node(tmp_path)
    before = {path.name for path in tmp_path.iterdir()}

    token = discover_node_token(ControlPlaneEndpoint.parse(_INTERNAL_URL), environ)

    assert token == "head-token"
    assert {path.name for path in tmp_path.iterdir()} == before


def test_a_node_token_is_never_sent_to_a_plane_that_did_not_issue_it(tmp_path: Path) -> None:
    """A bearer presented to the wrong control plane is a disclosure, not a retry."""

    environ = _node(tmp_path)

    assert discover_node_token(ControlPlaneEndpoint.parse("http://10.128.0.9:8080"), environ) is None


def test_an_issuer_head_keeps_no_bootstrap_token_and_that_is_ordinary(tmp_path: Path) -> None:
    """Absent is not broken: an OIDC head mints no bootstrap credential."""

    environ = _node(tmp_path, token=None)

    assert discover_node_token(ControlPlaneEndpoint.parse(_INTERNAL_URL), environ) is None


def test_an_unreadable_credential_names_the_group_instead_of_401ing_later(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning None here surfaces as a 401 naming neither the file nor the group."""

    environ = _node(tmp_path)
    original = Path.open

    def deny(self: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if self.name == "head.env":
            raise PermissionError(13, "Permission denied")
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", deny)

    with pytest.raises(NodeContextError, match="eyvan-head"):
        discover_node_token(ControlPlaneEndpoint.parse(_INTERNAL_URL), environ)


def test_an_unreadable_credential_does_not_break_commands_that_need_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Found on the first real head: this broke every command, including `eyvan context`.

    Raising is right for a caller that needs the credential -- silence there
    becomes an unexplained 401 much later. It is wrong as a startup step, which
    is where both the CLI and `init()` call it: most commands need no credential,
    and one unreadable file turned a degraded capability into a CLI that could
    not run at all, with a traceback instead of the message.

    The explanation is still delivered, once, where it is discovered.
    """

    environ = _node(tmp_path)
    original = Path.open

    def deny(self: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if self.name == "head.env":
            raise PermissionError(13, "Permission denied")
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", deny)
    reported: list[str] = []

    token = discover_node_token_or_warn(
        ControlPlaneEndpoint.parse(_INTERNAL_URL),
        environ,
        warn=reported.append,
    )

    assert token is None
    assert len(reported) == 1
    assert "eyvan-head" in reported[0]

    # And `init()` still produces a usable client -- pointed at the right plane,
    # simply without a credential it could not read.
    context = init(environ=environ)
    url, authorization = _wire(context)
    assert url.rstrip("/") == _INTERNAL_URL
    assert authorization is None
