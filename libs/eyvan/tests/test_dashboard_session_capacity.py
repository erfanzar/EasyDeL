# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Login floods must not grow the in-process session store without bound."""

import pytest
from fastapi import HTTPException

from eyvan.server.api.dashboard_sessions import DashboardSessionStore


@pytest.mark.parametrize("kind", ["transaction", "session"])
def test_the_default_store_refuses_overflow_without_evicting_live_entries(monkeypatch, kind) -> None:
    clock = [100.0]
    monkeypatch.setattr("eyvan.server.api.dashboard_sessions.time.monotonic", lambda: clock[0])
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=60)

    def create():
        if kind == "transaction":
            return store.begin_transaction()[0]
        return store.create_session(access_token="fixture", id_token=None, expires_in=60)

    tokens = [create() for _ in range(1024)]
    with pytest.raises(HTTPException) as refusal:
        create()
    assert refusal.value.status_code == 503
    assert refusal.value.headers["Retry-After"]
    if kind == "transaction":
        assert store.pop_transaction(tokens[0]) is not None
    else:
        assert store.resolve(tokens[0]) is not None
        store.drop(tokens[0])
    assert create()
    clock[0] = 161.0
    assert create(), "expired entries must release capacity"
