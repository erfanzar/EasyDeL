# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Untrusted Kubernetes denial bodies must not cross the diagnostic boundary."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from starlette.requests import Request

from eyvan.capacity.drivers.base import CapacityProviderError, CapacityProviderFailureCode
from eyvan.capacity.providers.kubernetes.api import KubernetesApi
from eyvan.capacity.providers.kubernetes.workers import secret_manifest
from eyvan.config import EyvanSettings
from eyvan.worker.agent.app import create_agent_app

_SYNTHETIC_TOKEN = "synthetic-worker-token-writer-041-not-a-real-credential"


@pytest.mark.parametrize("surface", ["diagnostic", "agent-response"])
@pytest.mark.parametrize("status", [401, 403, 422, 429, 500, 503])
def test_secret_denial_never_exposes_worker_token(tmp_path, status: int, surface: str) -> None:
    """Exercise the submitted Secret and the agent's actual JSON exception handler."""

    manifest = secret_manifest("cell-writer-041", "eyvan", _SYNTHETIC_TOKEN)

    def deny(request: httpx.Request) -> httpx.Response:
        submitted = json.loads(request.content)
        assert request.method == "POST"
        assert submitted == manifest
        return httpx.Response(
            status,
            json={
                "message": f"admission denied Secret: {submitted}",
                "reason": _SYNTHETIC_TOKEN,
            },
        )

    token_path = tmp_path / "service-account-token"
    token_path.write_text("synthetic-service-account-token", encoding="utf-8")
    with httpx.Client(transport=httpx.MockTransport(deny)) as client:
        api = KubernetesApi(token_path=str(token_path), client=client)
        with pytest.raises(CapacityProviderError) as raised:
            api.create("/api/v1/namespaces/eyvan/secrets", manifest)

    error = raised.value
    retryable = status == 429 or status >= 500
    expected_code = (
        CapacityProviderFailureCode.PROVIDER_UNAVAILABLE if retryable else CapacityProviderFailureCode.CONFIGURATION
    )
    assert error.code is expected_code
    assert error.retryable is retryable
    if surface == "diagnostic":
        emitted = error.diagnostic
    else:
        # The factory registers the real handler; no service operation or provider
        # is needed to serialize an already-produced CapacityProviderError.
        app = create_agent_app(
            service=object(),  # type: ignore[arg-type]
            settings=EyvanSettings(agent_shared_secret="synthetic-agent-secret"),
        )
        handler = app.exception_handlers[CapacityProviderError]
        response = asyncio.run(handler(Request({"type": "http"}), error))
        assert response.status_code == 502
        emitted = response.body.decode("utf-8")
        assert json.loads(emitted)["code"] == expected_code.value
    assert str(status) in emitted
    assert _SYNTHETIC_TOKEN not in emitted
    assert "admission denied" not in emitted
    assert len(emitted) < 300


@pytest.mark.parametrize("operation", ["get", "observe"])
def test_error_reads_do_not_expose_untrusted_response_body(tmp_path, operation: str) -> None:
    """Both raising and observational reads omit arbitrary upstream text."""

    def deny(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text=f"denied: {_SYNTHETIC_TOKEN}")

    token_path = tmp_path / "service-account-token"
    token_path.write_text("synthetic-service-account-token", encoding="utf-8")
    with httpx.Client(transport=httpx.MockTransport(deny)) as client:
        api = KubernetesApi(token_path=str(token_path), client=client)
        if operation == "get":
            with pytest.raises(CapacityProviderError) as raised:
                api.get("/api/v1/namespaces/eyvan/secrets/cell-writer-041")
            emitted = raised.value.diagnostic
        else:
            observation = api.observe("/api/v1/nodes")
            assert observation.unreadable
            assert observation.payload is None
            emitted = observation.reason
    assert "403" in emitted
    assert _SYNTHETIC_TOKEN not in emitted
    assert "denied:" not in emitted
