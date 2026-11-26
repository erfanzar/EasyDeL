# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The server, not merely the browser cookie, bounds session lifetime."""

import pytest

from eyvan.server.api.dashboard_sessions import DashboardSessionStore


@pytest.mark.parametrize("provider_lifetime", [None, 0, 3600])
def test_a_session_cannot_outlive_the_configured_server_limit(monkeypatch, provider_lifetime) -> None:
    clock = [100.0]
    monkeypatch.setattr("eyvan.server.api.dashboard_sessions.time.monotonic", lambda: clock[0])
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=10)
    token = store.create_session(access_token="test-token", id_token=None, expires_in=provider_lifetime)
    clock[0] = 159.0
    assert store.resolve(token) is not None
    clock[0] = 160.0
    assert store.resolve(token) is None


def test_a_shorter_provider_expiry_still_limits_the_session(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("eyvan.server.api.dashboard_sessions.time.monotonic", lambda: clock[0])
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=10)
    token = store.create_session(access_token="test-token", id_token=None, expires_in=5)
    clock[0] = 105.0
    assert store.resolve(token) is None
