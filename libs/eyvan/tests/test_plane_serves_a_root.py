# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What answers when somebody opens the URL every command prints.

Eyvan states its control-plane URL constantly -- on every local command, in
`eyvan start`, in `doctor`, in the notice above a development session -- so
sooner or later somebody puts it in a browser. What they got was FastAPI's
`{"detail":"Not Found"}`, which reads as a plane that is broken rather than one
that serves an API and no UI.

Two things are pinned here. A plane that *has* a bundle serves it at "/", which
is what makes a development plane behave like the head it stands in for. A plane
that has none says so, in a page that needs nothing it cannot serve.
"""

from __future__ import annotations

from pathlib import Path

from eyvan.server.api import app as app_module


def test_a_checkout_serves_the_dashboard_beside_the_package(monkeypatch) -> None:
    """The bundle sits next to the package in a source tree; a head sets a
    variable to find its own. Without this fallback the development plane is the
    only one of the three that answers its own printed URL with a 404."""

    monkeypatch.delenv("EYVAN_DASHBOARD_DIR", raising=False)

    bundle = app_module.dashboard_bundle()

    assert bundle is not None, "this checkout has dashboard/dist built"
    assert bundle.name == "dist"
    assert (bundle / "index.html").is_file()


def test_the_declared_directory_wins_over_the_checkout(monkeypatch, tmp_path) -> None:
    """A head and the container image both point at their own built bundle, and
    a checkout that happens to be on the same disk must not displace it."""

    declared = tmp_path / "bundle"
    declared.mkdir()
    monkeypatch.setenv("EYVAN_DASHBOARD_DIR", str(declared))

    assert app_module.dashboard_bundle() == declared


def test_a_declared_directory_that_is_not_there_is_not_substituted(monkeypatch, tmp_path) -> None:
    """Silently serving a different bundle than the one an operator named is a
    worse answer than serving none: the UI would be from another build."""

    monkeypatch.setenv("EYVAN_DASHBOARD_DIR", str(tmp_path / "absent"))

    assert app_module.dashboard_bundle() is None


def test_a_wheel_install_has_no_bundle_to_find(monkeypatch, tmp_path) -> None:
    """A wheel built without a staged ``eyvan/_dashboard`` carries no UI, so the
    fallback has to come up empty rather than guess."""

    monkeypatch.delenv("EYVAN_DASHBOARD_DIR", raising=False)
    monkeypatch.setattr(app_module, "__file__", str(tmp_path / "eyvan/server/api/app.py"))

    assert app_module.dashboard_bundle() is None


def test_a_plane_without_a_dashboard_explains_itself_at_the_root() -> None:
    """The page replaces `{"detail":"Not Found"}`, which sends people to debug
    the plane instead of reading that it simply hosts no UI."""

    page = app_module.NO_DASHBOARD_PAGE

    assert "Eyvan control plane" in page
    assert "eyvan dashboard-dev" in page
    assert "/healthz" in page


def test_the_root_page_needs_nothing_it_cannot_serve() -> None:
    """A plane hosting no bundle hosts no stylesheet, font or script either, so
    a page with one link renders unstyled -- the failure it exists to avoid."""

    page = app_module.NO_DASHBOARD_PAGE

    assert "<link" not in page
    assert "<script" not in page
    assert "http://" not in page and "https://" not in page


def test_the_root_page_is_registered_only_when_no_bundle_is_mounted() -> None:
    """A route at "/" is matched before a mount at "/", so registering both
    would shadow the dashboard's index with the page that says there is none."""

    source = Path(app_module.__file__).read_text(encoding="utf-8")
    mount = source.index('app.mount("/"')
    fallback = source.index("async def dashboard_absent")

    assert mount < fallback, "the fallback belongs in the else branch of the mount"
    assert source.count("async def dashboard_absent") == 1


def _isolated_loopback_app(tmp_path: Path):
    """Build a loopback app that owns its own state, not the caller's."""

    from eyvan.config import EyvanSettings, RepositoryBackend
    from eyvan.server.api.app import create_app

    return create_app(
        settings=EyvanSettings(
            dev_auth_enabled=True,
            repository_backend=RepositoryBackend.LOCAL,
            local_state_path=tmp_path / "control-plane.json",
            local_artifact_token_key_path=tmp_path / "artifact-token.key",
            artifact_store_uri=str(tmp_path / "managed-artifacts"),
            controller_enabled=False,
        )
    )


def test_the_dashboard_config_names_the_planes_own_auth_posture(monkeypatch, tmp_path) -> None:
    """The browser could see "no OIDC" and not "and none is expected here".

    Those two render identically and only one of them means something is wrong,
    so a laptop plane -- which has no OIDC by design -- was greeted as broken.
    """

    from fastapi.testclient import TestClient

    monkeypatch.setenv("HOME", str(tmp_path))

    with TestClient(_isolated_loopback_app(tmp_path)) as client:
        payload = client.get("/api/v1/dashboard/config").json()

    # Compared against the enum rather than a literal: the browser branches on
    # this string, so a renamed member has to fail here and not in a browser.
    from eyvan.server.api.server_startup import DevelopmentAuthMode

    assert payload["development_auth"] in {mode.value for mode in DevelopmentAuthMode}
    assert payload["development_auth"] != DevelopmentAuthMode.DISABLED.value
    assert payload["issuer"] is None


def test_the_dashboard_config_stays_readable_without_a_credential(monkeypatch, tmp_path) -> None:
    """Requiring a token to learn how to obtain one is a loop, and this route is
    the first thing the page loads."""

    from fastapi.testclient import TestClient

    monkeypatch.setenv("HOME", str(tmp_path))

    with TestClient(_isolated_loopback_app(tmp_path)) as client:
        response = client.get("/api/v1/dashboard/config")

    assert response.status_code == 200


def test_the_dashboard_config_reports_the_scope_the_plane_puts_a_session_in(monkeypatch, tmp_path) -> None:
    """The dashboard compiled in "acme"/"research" while the CLI defaults to
    "default"/"default", so a job submitted with `eyvan run` never appeared on
    the page -- both are free-form scopes a plane answers for, so nothing
    failed and the page simply asked about a different project."""

    from fastapi.testclient import TestClient

    monkeypatch.setenv("HOME", str(tmp_path))

    with TestClient(_isolated_loopback_app(tmp_path)) as client:
        payload = client.get("/api/v1/dashboard/config").json()

    assert payload["default_org"] == payload["default_project"]
    assert payload["default_org"]


def test_the_reported_scope_is_the_one_the_cli_submits_into() -> None:
    """Two constants for one fact is how they drift apart. Checked here because
    the SDK and the server hold their own, and a page that trusts the server's
    answer has to land where `eyvan run` put the work.
    """

    from eyvan.sdk.contexts import DEFAULT_ORG_ID, DEFAULT_PROJECT_ID
    from eyvan.server.api.auth import DEFAULT_DEVELOPMENT_SCOPE

    assert DEFAULT_DEVELOPMENT_SCOPE == DEFAULT_ORG_ID == DEFAULT_PROJECT_ID
