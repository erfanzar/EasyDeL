# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Malformed callback input must be rejected before any provider I/O."""

from fastapi import FastAPI
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.server.api.dashboard_sessions import (
    TRANSACTION_COOKIE,
    DashboardSessionStore,
    register_dashboard_auth_routes,
)


def test_non_ascii_callback_state_is_a_bad_request_not_a_server_error() -> None:
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=10)
    transaction, _, _, _ = store.begin_transaction()
    app = FastAPI()
    register_dashboard_auth_routes(
        app,
        settings=EyvanSettings(
            public_url="https://plane.example.com",
            oidc_issuer="https://issuer.example.com",
            oidc_dashboard_client_id="dashboard",
        ),
        store=store,
    )
    with TestClient(app, base_url="https://plane.example.com", raise_server_exceptions=False) as client:
        client.cookies.set(TRANSACTION_COOKIE, transaction)
        response = client.get("/api/v1/dashboard/auth/callback", params={"code": "test", "state": "نامعتبر"})
    assert response.status_code == 400
    assert "mismatched" in response.json()["detail"]


def test_provider_error_callbacks_still_require_the_initiating_transaction() -> None:
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=10)
    transaction, state, _, _ = store.begin_transaction()
    app = FastAPI()
    register_dashboard_auth_routes(
        app,
        settings=EyvanSettings(
            public_url="https://plane.example.com",
            oidc_issuer="https://issuer.example.com",
            oidc_dashboard_client_id="dashboard",
        ),
        store=store,
    )
    with TestClient(app, base_url="https://plane.example.com") as client:
        forged = client.get(
            "/api/v1/dashboard/auth/callback", params={"error": "access_denied", "state": state}, follow_redirects=False
        )
        assert forged.status_code == 400
        client.cookies.set(TRANSACTION_COOKIE, transaction)
        legitimate = client.get(
            "/api/v1/dashboard/auth/callback", params={"error": "access_denied", "state": state}, follow_redirects=False
        )
        assert legitimate.status_code in {303, 307}
        assert legitimate.headers["location"].startswith("/?auth_error=")
        replay = client.get(
            "/api/v1/dashboard/auth/callback", params={"error": "access_denied", "state": state}, follow_redirects=False
        )
        assert replay.status_code == 400


def test_provider_error_details_are_not_reflected_into_the_dashboard_url() -> None:
    for provider_error in (
        {"error": "access_denied", "error_description": "PRIVATE_PROVIDER_DETAIL"},
        {"error": "PRIVATE_PROVIDER_DETAIL"},
    ):
        store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=10)
        transaction, state, _, _ = store.begin_transaction()
        app = FastAPI()
        register_dashboard_auth_routes(
            app,
            settings=EyvanSettings(
                public_url="https://plane.example.com",
                oidc_issuer="https://issuer.example.com",
                oidc_dashboard_client_id="dashboard",
            ),
            store=store,
        )
        with TestClient(app, base_url="https://plane.example.com") as client:
            client.cookies.set(TRANSACTION_COOKIE, transaction)
            response = client.get(
                "/api/v1/dashboard/auth/callback",
                params={**provider_error, "state": state},
                follow_redirects=False,
            )
        assert response.status_code in {303, 307}
        assert response.headers["location"].startswith("/?auth_error=")
        assert "PRIVATE_PROVIDER_DETAIL" not in response.headers["location"]
        assert "PRIVATE_PROVIDER_DETAIL" not in response.text
        assert store.pop_transaction(transaction) is None
