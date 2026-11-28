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


# --------------------------------------------------------------------------- probe against a real plane


def _pool(pool_id: str, fleet: str, *, enabled: bool = True, max_cells: int = 2):
    from eyvan.records import CapacityPoolRecord

    return CapacityPoolRecord(
        id=pool_id,
        fleet_name=fleet,
        node_type=pool_id.rsplit("/", 1)[-1],
        provider="gcp",
        capacity_backend="gcp-compute",
        provider_project="project-a",
        region="us-central1",
        zone="us-central1-a",
        capacity={"cpu": 4, "memory_gib": 16, "tpu": 8},
        allocatable={"cpu": 4, "memory_gib": 16, "tpu": 8},
        executor="native-local",
        agent_endpoint="http://agent:8090",
        provider_options={"kind": "cpu", "machine_type": "n2-standard-4"},
        min_cells=0,
        max_cells=max_cells,
        enabled=enabled,
    )


def test_probe_discovers_ready_fleets_for_non_admin_team_roles(tmp_path, monkeypatch) -> None:
    """Non-admin team users get 403 on /capacity-pools and must discover ready fleets via /capacity."""

    import asyncio

    import httpx
    from starlette.testclient import TestClient

    from eyvan.config import EyvanSettings
    from eyvan.records import Principal, Role, RoleGrant
    from eyvan.sdk.commands.connect import probe
    from eyvan.server.api.app import create_app

    monkeypatch.setenv("HOME", str(tmp_path))
    base = "http://127.0.0.1:8080"
    app = create_app(
        settings=EyvanSettings(
            dev_auth_enabled=True,
            team_mode=True,
            controller_enabled=False,
            artifact_token_key="k" * 32,
        )
    )
    admin_principal = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))

    async def seed() -> None:
        svc = app.state.service
        await svc.upsert_capacity_pool(admin_principal, _pool("research/v5p-8", "research"), idempotency_key="p1")
        await svc.upsert_capacity_pool(admin_principal, _pool("research/v6e-8", "research"), idempotency_key="p2")
        await svc.upsert_capacity_pool(admin_principal, _pool("training/v5p-16", "training"), idempotency_key="p3")
        await svc.upsert_capacity_pool(
            admin_principal,
            _pool("retired/v5p-8", "retired", enabled=False, max_cells=0),
            idempotency_key="p4",
        )

    asyncio.run(seed())
    http = TestClient(app, base_url=base)

    def _onboard_token(subject: str, grants: list[dict[str, str]], *, tier: str | None = None) -> str:
        admin_headers = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
        invite_body: dict[str, object] = {"email": f"{subject}@example.com", "subject": subject}
        if tier is not None:
            invite_body["tier"] = tier
            if tier != "admin":
                invite_body["org_id"] = grants[0]["org_id"]
                invite_body["project_id"] = grants[0]["project_id"]
        else:
            invite_body.update(grants[0])
        invite = http.post("/api/v1/team/invites", json=invite_body, headers=admin_headers).json()
        assert (
            http.post(
                "/api/v1/team/invites/accept",
                json={"token": invite["token"], "password": "correct horse battery"},
            ).status_code
            == 200
        )
        http.cookies.clear()
        for idx, extra in enumerate(grants[1:], start=1):
            resp = http.post(
                "/api/v1/grants",
                json={"subject": subject, "grant": extra},
                headers={**admin_headers, "Idempotency-Key": f"grant-{subject}-{idx}"},
            )
            assert resp.status_code in (200, 201), resp.text
        login = http.post(
            "/api/v1/team/login",
            json={"email": f"{subject}@example.com", "password": "correct horse battery", "token_name": "cli"},
        )
        assert login.status_code == 200, login.text
        return login.json()["token"]

    admin_token = _onboard_token("chief", [], tier="admin")
    trusted_token = _onboard_token("carol", [{"org_id": "acme", "project_id": "research"}], tier="trusted")
    submitter_token = _onboard_token("alice", [{"org_id": "acme", "project_id": "research"}], tier="user")
    multi_project_token = _onboard_token(
        "dana",
        [
            {"org_id": "acme", "project_id": "research"},
            {"role": "project_submitter", "org_id": "acme", "project_id": "training"},
        ],
        tier="user",
    )
    viewer_token = _onboard_token("bob", [{"role": "project_viewer", "org_id": "acme", "project_id": "research"}])

    real_client = httpx.Client

    def handle(request: httpx.Request) -> httpx.Response:
        resp = http.request(request.method, str(request.url), headers=dict(request.headers), content=request.content)
        return httpx.Response(resp.status_code, headers=resp.headers, content=resp.content)

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))

    # Fleet admin still discovers every configured pool via /capacity-pools.
    admin = probe(base, token=admin_token)
    assert admin.reachable is True
    assert admin.fleets == ("research", "retired", "training")
    assert admin.pools == 4
    assert admin.subject == "chief"

    # Trusted user with explicit scope discovers enabled fleets & pools (skipping disabled fleet).
    trusted = probe(base, token=trusted_token, org_id="acme", project_id="research")
    assert trusted.reachable is True
    assert trusted.fleets == ("research", "training")
    assert trusted.pools == 3
    assert trusted.subject == "carol"

    # Plain submitter (--tier user) discovers the same ready fleets.
    submitter = probe(base, token=submitter_token, org_id="acme", project_id="research")
    assert submitter.reachable is True
    assert submitter.fleets == ("research", "training")
    assert submitter.pools == 3
    assert submitter.subject == "alice"

    # Multi-project submitter probing with default/default falls back to /session submit scopes.
    multi = probe(base, token=multi_project_token)
    assert multi.reachable is True
    assert multi.fleets == ("research", "training")
    assert multi.pools == 3

    # Viewer lacks SUBMIT and gets empty fleets cleanly without crashing.
    viewer = probe(base, token=viewer_token, org_id="acme", project_id="research")
    assert viewer.reachable is True
    assert viewer.fleets == ()
    assert viewer.pools == 0


def test_probe_falls_back_to_fleets_when_capacity_endpoint_is_unavailable(tmp_path, monkeypatch) -> None:
    import httpx

    from eyvan.sdk.commands.connect import probe

    monkeypatch.setenv("HOME", str(tmp_path))
    base = "http://127.0.0.1:8080"
    seen: list[tuple[str, dict[str, str]]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.url.params)))
        if request.url.path == "/api/v1/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/api/v1/session":
            return httpx.Response(
                200,
                json={
                    "subject": "alice",
                    "grants": [{"role": "project_submitter", "org_id": "acme", "project_id": "r"}],
                    "scopes": [{"org_id": "acme", "project_id": "r", "permissions": ["submit"]}],
                },
            )
        if request.url.path == "/api/v1/capacity-pools":
            return httpx.Response(403, json={"detail": "lacks manage_fleet permission"})
        if request.url.path == "/api/v1/capacity":
            return httpx.Response(404, json={"detail": "not found"})
        if request.url.path == "/api/v1/fleets":
            assert request.url.params["org_id"] == "acme"
            assert request.url.params["project_id"] == "r"
            return httpx.Response(200, json=["beta", "alpha"])
        raise AssertionError(request.url)

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))

    target = probe(base, token="eyv_test", org_id="acme", project_id="r")
    assert target.reachable is True
    assert target.fleets == ("alpha", "beta")
    assert target.pools == 0
    assert [path for path, _ in seen] == [
        "/api/v1/health",
        "/api/v1/session",
        "/api/v1/capacity-pools",
        "/api/v1/capacity",
        "/api/v1/fleets",
    ]
