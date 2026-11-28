# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``/metrics`` was the one route with no authentication at all.

Every other route in this package resolves a principal and asks the authorizer.
``/metrics`` had no ``Depends`` -- and it exports ``eyvan_project_spend_usd``
labeled ``(org, project, kind)``, so anyone who could reach the port read every
tenant's spend. The shipped manifests put no NetworkPolicy in front of it, so
"reach the port" meant any pod in the cluster.

A bearer token rather than an OIDC identity, because a Prometheus scraper has no
identity to present and the realistic alternative to a token is leaving the
endpoint open.

The part that makes this a fix rather than an option is the startup refusal: a
network-bound server will not start without a token. A setting nobody is
required to set is a setting most deployments will not have.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.server.api.app import create_app
from eyvan.server.api.server_bind import ServerBindPolicy
from eyvan.server.api.server_startup import ServerStartupSettings


def _network_settings(**overrides) -> EyvanSettings:
    base = {
        "repository_backend": RepositoryBackend.MEMORY,
        "metrics_token": "scrape-token",
        "artifact_token_key": "k" * 32,
    }
    base.update(overrides)
    return EyvanSettings(**base)  # type: ignore[arg-type]


def _client(settings: EyvanSettings, host: str = "127.0.0.1") -> TestClient:
    return TestClient(create_app(settings=settings, server_bind=ServerBindPolicy.from_host(host)))


def test_a_network_bound_server_refuses_to_start_without_a_metrics_token() -> None:
    """The refusal is the fix. Everything else here is what the refusal protects.

    Documented-but-optional would leave the exposure in place for every
    deployment that did not read the paragraph, which is most of them.
    """

    with pytest.raises(ValueError, match="EYVAN_METRICS_TOKEN"):
        ServerStartupSettings.from_settings(
            ServerBindPolicy.from_host("0.0.0.0"),
            EyvanSettings(repository_backend=RepositoryBackend.POSTGRES),
        )


def test_a_configured_token_is_required_at_the_endpoint() -> None:
    """A refusal at startup is worth nothing if the route then serves anyone."""

    client = _client(_network_settings())

    assert client.get("/metrics").status_code == 401


@pytest.mark.parametrize(
    "header",
    [
        {"Authorization": "Bearer wrong-token"},
        {"Authorization": "scrape-token"},
        {"Authorization": "Basic scrape-token"},
        {"Authorization": "Bearer "},
        {"X-Metrics-Token": "scrape-token"},
    ],
    ids=["wrong-value", "no-scheme", "wrong-scheme", "empty", "wrong-header"],
)
def test_a_near_miss_credential_is_still_refused(header: dict[str, str]) -> None:
    """Each of these is a plausible misconfiguration that must not half-work.

    A scraper configured with the right secret in the wrong place should fail
    loudly at setup rather than succeed for one of five spellings.
    """

    assert _client(_network_settings()).get("/metrics", headers=header).status_code == 401


def test_the_right_token_serves_the_exposition() -> None:
    """The endpoint has to keep working, or operators will remove the token."""

    response = _client(_network_settings()).get("/metrics", headers={"Authorization": "Bearer scrape-token"})

    assert response.status_code == 200
    assert "eyvan_api_requests_total" in response.text


def test_spend_is_not_readable_without_the_token() -> None:
    """Naming the concrete leak: per-tenant cost, by organization and project.

    This is the series that made an unauthenticated endpoint a tenancy problem
    rather than an operational preference.
    """

    settings = _network_settings()
    client = _client(settings)
    app_metrics = client.app.state.metrics  # type: ignore[attr-defined]
    app_metrics.project_spend.labels(org="acme", project="research", kind="spent").set(12345.0)

    denied = client.get("/metrics")
    allowed = client.get("/metrics", headers={"Authorization": "Bearer scrape-token"})

    assert denied.status_code == 401
    assert "12345" not in denied.text
    assert 'eyvan_project_spend_usd{kind="spent",org="acme",project="research"} 12345.0' in allowed.text


def test_a_loopback_server_without_a_token_still_serves_metrics() -> None:
    """Local development must not need a credential it has nowhere to put.

    Leaving it open is safe exactly here and nowhere else: the startup refusal
    means an unconfigured server can only be bound to loopback.
    """

    client = _client(EyvanSettings(dev_auth_enabled=True))

    assert client.get("/metrics").status_code == 200
