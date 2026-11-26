# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Local repository readiness and credential-safe decoding regressions."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import httpx
import pytest

from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.records import CommandExecution, WebhookRecord, WorkloadDefinition, WorkloadRecord
from eyvan.server.api.app import create_app
from eyvan.server.api.server_bind import ServerBindPolicy
from eyvan.server.persistence.local import (
    LocalControlPlaneAmbiguousCommitError,
    LocalControlPlaneError,
    LocalControlPlaneSchemaError,
    LocalControlPlaneStore,
    LocalJsonRepository,
)
from eyvan.server.persistence.state import ControlPlaneState


def _webhook(secret: str = "dummy-key-1234567") -> WebhookRecord:
    return WebhookRecord(
        id="webhook-1",
        org_id="acme",
        project_id="research",
        url="https://receiver.example.invalid/events",
        secret=secret,
        created_by="alice",
    )


@pytest.mark.asyncio
async def test_local_readiness_stays_unavailable_after_ambiguous_commit_until_explicit_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = EyvanSettings(
        repository_backend=RepositoryBackend.LOCAL,
        local_state_path=tmp_path / "control-plane.json",
        local_artifact_token_key_path=tmp_path / "artifact-token.key",
        artifact_store_uri=str(tmp_path / "managed-artifacts"),
        controller_enabled=False,
        dev_auth_enabled=True,
    )
    repository = LocalJsonRepository(settings.local_state_path)
    app = create_app(settings=settings, repository=repository, server_bind=ServerBindPolicy.from_host("127.0.0.1"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/readyz")).status_code == 200

            def fail_directory_fsync() -> None:
                raise OSError("simulated state directory fsync failure")

            webhook = _webhook()
            with monkeypatch.context() as context:
                context.setattr(repository.store, "_fsync_parent", fail_directory_fsync)
                with pytest.raises(LocalControlPlaneAmbiguousCommitError, match="durability is unknown"):
                    async with repository.transaction() as transaction:
                        await transaction.put_webhook(webhook)

            assert repository.store.commit_status_ambiguous is True
            for _ in range(2):
                response = await client.get("/readyz")
                assert response.status_code == 503
                assert response.json()["status"] == "unavailable"
                assert repository.store.commit_status_ambiguous is True
            with pytest.raises(LocalControlPlaneAmbiguousCommitError, match="ambiguous prior commit"):
                async with repository.transaction():
                    pytest.fail("read transactions must remain refused after readiness probes")

            await repository.close()
            assert (await client.get("/readyz")).status_code == 503
            assert repository.store.commit_status_ambiguous is True
            await repository.open()
            assert repository.store.commit_status_ambiguous is False
            assert (await client.get("/readyz")).status_code == 200
            async with repository.transaction() as transaction:
                assert await transaction.get_webhook(webhook.id) == webhook


@pytest.mark.asyncio
async def test_local_ping_does_not_implicitly_open_or_recover_a_closed_repository(tmp_path: Path) -> None:
    repository = LocalJsonRepository(tmp_path / "control-plane.json")
    with pytest.raises(LocalControlPlaneError, match="not open"):
        await repository.ping()
    assert not repository.store.path.exists()
    assert not repository._opened
    try:
        await repository.open()
        await repository.ping()
    finally:
        await repository.close()
    with pytest.raises(LocalControlPlaneError, match="not open"):
        await repository.ping()
    assert not repository._opened


def _assert_private_error(error: Exception, secret: str) -> None:
    diagnostics = (str(error), repr(error), "".join(traceback.format_exception(error)))
    for diagnostic in diagnostics:
        for fragment in (secret, secret[:8], secret[-8:]):
            assert fragment not in diagnostic


@pytest.mark.parametrize("secret", ("dummy-key-1234567", "dummy-prefix-" + "x" * 80 + "-dummy-suffix"))
@pytest.mark.parametrize("corruption", ("missing-creator", "secret-extra-key"))
def test_malformed_webhook_diagnostics_do_not_disclose_signing_keys(
    tmp_path: Path, secret: str, corruption: str
) -> None:
    webhook = _webhook(secret)
    path = tmp_path / "control-plane.json"
    store = LocalControlPlaneStore(path)
    store.save(ControlPlaneState(webhooks={webhook.id: webhook}))
    assert store.load().webhooks[webhook.id] == webhook
    raw = json.loads(path.read_text())
    assert raw["version"] == 6
    if corruption == "missing-creator":
        del raw["webhooks"][0]["created_by"]
        # JSON key order is not a schema constraint; put the key last to
        # exercise the credential-bearing suffix in a truncated input repr.
        raw["webhooks"][0]["secret"] = raw["webhooks"][0].pop("secret")
    else:
        raw["webhooks"][0][secret] = "invalid-extra-field"
    path.write_text(json.dumps(raw))
    path.chmod(0o600)
    with pytest.raises(LocalControlPlaneSchemaError) as caught:
        store.load()
    _assert_private_error(caught.value, secret)
    assert "webhooks" in str(caught.value)
    assert ("missing" if corruption == "missing-creator" else "extra_forbidden") in str(caught.value)


def test_malformed_workload_diagnostics_do_not_disclose_environment_credentials(tmp_path: Path) -> None:
    secret = "dummy-environment-token-1234567"
    workload = WorkloadRecord(
        id="workload-1",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train", execution=CommandExecution(command="echo done"), environment={"EXAMPLE_TOKEN": secret}
        ),
        created_by="alice",
    )
    path = tmp_path / "control-plane.json"
    store = LocalControlPlaneStore(path)
    store.save(ControlPlaneState(workloads={workload.id: workload}))
    assert store.load().workloads[workload.id] == workload
    raw = json.loads(path.read_text())
    raw["workloads"][0]["spec"]["environment"]["EXAMPLE_TOKEN"] = {"invalid-value": secret}
    path.write_text(json.dumps(raw))
    path.chmod(0o600)
    with pytest.raises(LocalControlPlaneSchemaError) as caught:
        store.load()
    _assert_private_error(caught.value, secret)
    assert "workloads" in str(caught.value)
    assert "string_type" in str(caught.value)
