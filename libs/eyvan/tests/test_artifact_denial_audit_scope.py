# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Artifact denial audits must follow the workload's durable scope.

The artifact transfer routes are reachable without any principal: their only
credential is the capability token. When that token is refused, the denial
used to be recorded under the *caller's* query-string org/project -- an
unauthenticated write into any tenant's audit ledger. Both the read and the
write refusal paths resolve the workload's real scope instead, and record
nothing for an unresolvable workload.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eyvan.artifacts.managed import LocalManagedArtifactStore
from eyvan.artifacts.transfers import ManagedArtifactAccess
from eyvan.records import Principal, Role, RoleGrant
from eyvan.server.api.app import ArtifactTokenSigner, create_app
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService


@pytest.mark.asyncio
async def test_a_refused_result_write_names_no_foreign_ledger(tmp_path) -> None:
    repository = InMemoryRepository()
    await repository.open()
    store = LocalManagedArtifactStore(tmp_path / "artifacts")
    service = ControlPlaneService(repository=repository, artifact_store=store)
    settings = EyvanSettings(dev_auth_enabled=True, artifact_token_key="a" * 32)
    artifact_access = ManagedArtifactAccess(
        store=store,
        signer=ArtifactTokenSigner(
            "a" * 32,
            default_ttl_seconds=300,
            max_ttl_seconds=300,
            clock_skew_seconds=30,
        ),
        public_url=settings.public_url,
        token_ttl_seconds=300,
    )
    app = create_app(settings=settings, repository=repository, service=service, artifact_access=artifact_access)
    client = TestClient(app)

    response = client.put(
        "/api/v1/artifacts/results/run-1/content",
        params={"org_id": "foreign", "project_id": "spy", "workload_id": "wl_doesnotexist"},
        headers={"X-Eyvan-Artifact-Token": "bogus"},
    )
    assert response.status_code >= 400

    fleet_admin = Principal(subject="system", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    foreign = await service.list_events(fleet_admin, org_id="foreign", project_id="spy", limit=100)
    denial_actions = {event.action for event in foreign if event.action.startswith("artifact.")}
    assert not denial_actions, f"the caller's claimed scope must not receive denial events: {denial_actions}"


from eyvan.config import EyvanSettings  # noqa: E402  (kept beside the test that parametrizes it)
