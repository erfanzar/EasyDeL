# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lazy public entry point for the Eyvan FastAPI application."""

from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
    from fastapi import FastAPI
    from prometheus_client import CollectorRegistry

    from eyvan._internal.observability import EyvanMetrics
    from eyvan._internal.platform import EyvanPlatformConfig
    from eyvan.artifacts.transfers import ManagedArtifactAccess
    from eyvan.capacity.autoscale.service import TpuQuotaDiscovery
    from eyvan.config import EyvanSettings
    from eyvan.server.api.auth import TokenVerifier
    from eyvan.server.api.server_bind import ServerBindPolicy
    from eyvan.server.api.server_startup import ServerStartupSettings
    from eyvan.server.persistence.base import Repository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.soak_evidence import SoakEvidenceStore


def create_app(
    *,
    settings: EyvanSettings | None = None,
    repository: Repository | None = None,
    service: ControlPlaneService | None = None,
    verifier: TokenVerifier | None = None,
    metrics: EyvanMetrics | None = None,
    metrics_registry: CollectorRegistry | None = None,
    artifact_access: ManagedArtifactAccess | None = None,
    server_bind: ServerBindPolicy | None = None,
    automatic_loopback_auth: bool = False,
    platform_config: EyvanPlatformConfig | None = None,
    soak_evidence: SoakEvidenceStore | None = None,
    quota_discovery: TpuQuotaDiscovery | None = None,
    server_startup: ServerStartupSettings | None = None,
) -> FastAPI:
    """Import and build the control-plane application on demand.

    This signature must stay identical to :func:`eyvan.server.api.app.create_app`; it is
    the only way the factory is reachable without importing FastAPI eagerly.
    """

    from eyvan.server.api.app import create_app as build_app

    return build_app(
        settings=settings,
        repository=repository,
        service=service,
        verifier=verifier,
        metrics=metrics,
        metrics_registry=metrics_registry,
        artifact_access=artifact_access,
        server_bind=server_bind,
        automatic_loopback_auth=automatic_loopback_auth,
        platform_config=platform_config,
        soak_evidence=soak_evidence,
        quota_discovery=quota_discovery,
        server_startup=server_startup,
    )


__all__ = ("create_app",)
