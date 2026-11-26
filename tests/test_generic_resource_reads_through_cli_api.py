# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generic resource reads through the real CLI, API, and memory repository."""

from __future__ import annotations

import asyncio
import json

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import BudgetRecord, CapacityPoolRecord, Principal, QuotaRecord, Role, RoleGrant
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands.resources import list_resource
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.contexts import EyvanContextConfig, FileContextStore
from eyvan.server.api import app as app_module
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService


class _Api:
    def __init__(self, client: TestClient) -> None:
        self.client = client
        self.responses = []

    def request(self, method, path, *, params=None):
        response = self.client.request(method, path, params=params)
        self.responses.append(response)
        response.raise_for_status()
        return response.json()


@pytest.fixture
def resource_api(monkeypatch, tmp_path):
    # Build identity is unrelated to the read contract; skip source-tree hashing.
    # No listener or lifespan is run.
    monkeypatch.setattr(app_module, "build_identity", lambda: None)
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)

    async def seed():
        async with repository.transaction() as tx:
            for org_id in ("inherited-org", "explicit-org", "foreign-org"):
                for project_id in ("research", "training"):
                    await tx.put_budget(BudgetRecord(org_id=org_id, project_id=project_id, limit=100, updated_by="seed"))
                    await tx.put_quota(
                        QuotaRecord(org_id=org_id, project_id=project_id, limits={"cpu": 8}, updated_by="seed")
                    )

    asyncio.run(seed())
    app = app_module.create_app(
        settings=EyvanSettings(
            dev_auth_enabled=True,
            artifact_token_key="writer-059-artifact-test-key-32-bytes",
            artifact_store_uri=str(tmp_path / "artifacts"),
            platform_config_path=str(tmp_path / "platform.yaml"),
        ),
        repository=repository,
        service=service,
    )
    principal = Principal(
        subject="reader",
        grants=tuple(RoleGrant(role=Role.ORG_ADMIN, org_id=org_id) for org_id in ("inherited-org", "explicit-org")),
    )
    app.dependency_overrides[app_module.authenticated_principal] = lambda: principal
    api = _Api(TestClient(app))
    session = CliSession(
        client=api,
        identity=ClientIdentity(),
        context_store=FileContextStore(path=tmp_path / "context.json"),
        context_config=EyvanContextConfig(),
        base_url="http://testserver",
        org_id="inherited-org",
        project_id="research",
        overrides=ContextOverrides(),
    )
    yield session, api, repository
    api.client.close()


@pytest.mark.parametrize("kind", ["quotas", "budgets"])
@pytest.mark.parametrize("explicit_org", [False, True])
def test_org_listing_reaches_current_api_and_returns_all_selected_org_projects(resource_api, kind, explicit_org):
    session, api, _repository = resource_api
    args = [kind, "--json"]
    if explicit_org:
        args.extend(["--org", "explicit-org"])
    result = CliRunner().invoke(list_resource, args, obj=session)

    assert api.responses[-1].status_code == 200, api.responses[-1].text
    assert result.exit_code == 0, result.output
    records = json.loads(result.output)
    assert {record["org_id"] for record in records} == {"explicit-org" if explicit_org else "inherited-org"}
    assert {record["project_id"] for record in records} == {"research", "training"}
    assert dict(api.responses[-1].request.url.params) == {"org_id": "explicit-org" if explicit_org else "inherited-org"}


def test_quotas_accept_an_explicit_optional_project_without_narrowing_the_default(resource_api):
    session, api, _repository = resource_api
    result = CliRunner().invoke(list_resource, ["quotas", "--project", "training", "--json"], obj=session)

    assert api.responses[-1].status_code == 200, api.responses[-1].text
    assert result.exit_code == 0, result.output
    assert [record["project_id"] for record in json.loads(result.output)] == ["training"]
    assert dict(api.responses[-1].request.url.params) == {"org_id": "inherited-org", "project_id": "training"}


def test_budgets_do_not_send_an_unsupported_project_query(resource_api):
    session, api, _repository = resource_api
    session.project_id = None
    result = CliRunner().invoke(list_resource, ["budgets", "--project", "training", "--json"], obj=session)

    assert api.responses[-1].status_code == 200, api.responses[-1].text
    assert result.exit_code == 0, result.output
    assert {record["project_id"] for record in json.loads(result.output)} == {"research", "training"}
    assert dict(api.responses[-1].request.url.params) == {"org_id": "inherited-org"}


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("populated", [False, True])
def test_fleets_render_actual_api_names_and_preserve_json_and_empty_output(resource_api, as_json, populated):
    session, api, repository = resource_api

    async def seed_pools():
        async with repository.transaction() as tx:
            for index, name in enumerate(("training", "research", "disabled", "zero")):
                await tx.put_capacity_pool(
                    CapacityPoolRecord(
                        id=f"pool-{index}",
                        fleet_name=name,
                        node_type=f"workers-{index}",
                        provider="gcp",
                        capacity_backend="gcp-tpu",
                        provider_project="test-project",
                        region="us-east5",
                        zone="us-east5-a",
                        capacity={"tpu": 4},
                        allocatable={"tpu": 4},
                        accelerator_type="v5p-8",
                        capacity_class="spot",
                        executor="eyvan-native",
                        agent_endpoint="http://agent:8090",
                        enabled=name != "disabled",
                        max_cells=0 if name == "zero" else 1,
                    )
                )

    if populated:
        asyncio.run(seed_pools())
    args = ["fleets", *(["--json"] if as_json else [])]
    result = CliRunner().invoke(list_resource, args, obj=session)

    assert api.responses[-1].status_code == 200, api.responses[-1].text
    assert result.exit_code == 0, result.output
    names = ["research", "training"] if populated else []
    assert api.responses[-1].json() == names
    if as_json:
        assert json.loads(result.output) == names
    elif populated:
        assert result.output.splitlines() == ["NAME", "research", "training"]
        assert "No fleets." not in result.output
        assert "ZONE" not in result.output
    else:
        assert result.output == "No fleets.\n"
