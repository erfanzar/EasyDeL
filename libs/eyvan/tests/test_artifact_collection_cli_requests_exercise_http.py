# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Artifact collection CLI requests exercise the real HTTP and deletion contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from eyvan.artifacts.managed import callable_uri
from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.records import ManagedArtifactState, utcnow
from eyvan.sdk.client.transport import SyncApiTransport
from eyvan.sdk.commands.artifacts import artifacts
from eyvan.sdk.commands.http import EyvanClient
from eyvan.server.api.app import create_app
from eyvan.server.service.control_plane import DEFAULT_ARTIFACT_GRACE_PERIOD_S


@dataclass(frozen=True)
class _Scope:
    org_id: str
    project_id: str


@dataclass
class _Session:
    client: EyvanClient

    def workload_scope(self, *, org_id: str | None, project_id: str | None) -> _Scope:
        assert org_id is not None and project_id is not None
        return _Scope(org_id, project_id)


@pytest.mark.parametrize(
    ("options", "age_s", "expected_dry_run", "expected_count", "expected_grace", "expected_capped"),
    [
        ([], DEFAULT_ARTIFACT_GRACE_PERIOD_S + 3600, True, 2, 0, False),
        (["--apply", "--limit", "1", "--grace-period-s", "12"], 60, False, 1, 0, True),
        (["--dry-run", "--limit", "2", "--grace-period-s", "120"], 60, True, 0, 2, False),
    ],
    ids=["default-dry-run", "apply-bounded-custom-grace", "explicit-dry-run-protects-young-artifacts"],
)
def test_collect_cli_reaches_api_and_observes_collection_controls(
    tmp_path: Path,
    options: list[str],
    age_s: float,
    expected_dry_run: bool,
    expected_count: int,
    expected_grace: int,
    expected_capped: bool,
) -> None:
    app = create_app(
        settings=EyvanSettings(
            dev_auth_enabled=True,
            repository_backend=RepositoryBackend.MEMORY,
            artifact_store_uri=str(tmp_path / "managed"),
            artifact_token_key="k" * 32,
        )
    )
    service = app.state.service
    uris = []

    async def seed() -> None:
        for index in range(2):
            payload = f"collect-cli-payload-{index}".encode()
            uri = callable_uri(hashlib.sha256(payload).hexdigest())
            uris.append(uri)
            for project_id in ("research", "other"):
                await service.upload_managed_artifact(org_id="acme", project_id=project_id, uri=uri, payload=payload)
                async with service.repository.transaction() as tx:
                    record = await tx.get_managed_artifact("acme", project_id, uri)
                    assert record is not None
                    aged = utcnow() - timedelta(seconds=age_s)
                    await tx.put_managed_artifact(
                        record.model_copy(
                            update={
                                "created_at": aged,
                                "activated_at": aged,
                                "upload_started_at": aged,
                                "updated_at": aged,
                            }
                        )
                    )

    asyncio.run(seed())
    # Inject in-process HTTP at the existing transport seam. Both the CLI JSON
    # adapter and shared mutation transport remain real, with no socket traffic.
    http_client = TestClient(app)
    cli_client = EyvanClient.__new__(EyvanClient)
    cli_client._transport = SyncApiTransport(
        endpoint="http://testserver",
        client=http_client,
        headers={
            "X-Eyvan-User": "alice",
            "X-Eyvan-Roles": "admin",
            "X-Eyvan-Org": "acme",
            "X-Eyvan-Project": "research",
        },
    )
    try:
        result = CliRunner().invoke(
            artifacts,
            ["collect", "--org", "acme", "--project", "research", *options],
            obj=_Session(cli_client),
        )
    finally:
        http_client.close()

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["dry_run"] is expected_dry_run
    assert len(report["collected_uris"]) == expected_count
    assert report["within_grace"] == expected_grace
    assert report["capped"] is expected_capped
    assert set(report["collected_uris"]) <= set(uris)
    removed = set(report["collected_uris"]) if not expected_dry_run else set()

    async def verify_records() -> None:
        async with service.repository.transaction() as tx:
            for uri in uris:
                for project_id in ("research", "other"):
                    record = await tx.get_managed_artifact("acme", project_id, uri)
                    assert record is not None
                    deleted = project_id == "research" and uri in removed
                    assert record.state is (ManagedArtifactState.DELETED if deleted else ManagedArtifactState.ACTIVE)
                    assert service.artifact_store.exists(uri, org_id="acme", project_id=project_id) is not deleted

    asyncio.run(verify_records())
