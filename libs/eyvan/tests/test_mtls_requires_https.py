# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Configuring certificates must never silently select a plaintext transport."""

import httpx
import pytest
from test_identity_mtls import pki as _pki_fixture

from eyvan.worker.agent.client import CellAgentClient
from eyvan.worker.runtime.native_remote import NativeRemoteExecutor

pki = _pki_fixture


@pytest.mark.parametrize("kind", ["agent", "worker"])
@pytest.mark.parametrize("endpoint", ["http://127.0.0.1:8100", "http://worker.example:8100"])
def test_configured_mtls_refuses_plaintext_before_creating_a_transport(kind, endpoint, pki):
    with pytest.raises(ValueError, match="HTTPS"):
        if kind == "agent":
            driver = CellAgentClient(endpoint, token="fixture", mtls=pki["client"])
        else:
            driver = NativeRemoteExecutor(endpoint, secret="fixture", mtls=pki["client"])
        driver.client.close()


@pytest.mark.parametrize("kind", ["agent", "worker"])
def test_disabled_mtls_preserves_the_existing_http_transport(kind):
    endpoint = "http://127.0.0.1:8100"
    if kind == "agent":
        driver = CellAgentClient(endpoint, token="fixture")
    else:
        driver = NativeRemoteExecutor(endpoint, secret="fixture")
    try:
        assert driver.client.base_url.scheme == "http"
    finally:
        driver.client.close()


@pytest.mark.parametrize("kind", ["agent", "worker"])
def test_explicit_client_retains_ownership_of_its_transport_policy(kind, pki):
    endpoint = "http://127.0.0.1:8100"
    with httpx.Client(base_url=endpoint) as client:
        if kind == "agent":
            driver = CellAgentClient(endpoint, token="fixture", mtls=pki["client"], client=client)
        else:
            driver = NativeRemoteExecutor(endpoint, secret="fixture", mtls=pki["client"], client=client)
        assert driver.client is client
