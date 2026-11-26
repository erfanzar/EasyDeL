# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""DAG depth refusal must not depend on the request's declaration order."""

from __future__ import annotations

import random

import pytest
from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import CommandExecution, Principal, ResourceGang, ResourceVector, Role, RoleGrant, WorkloadDefinition
from eyvan.server.api.app import create_app
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.dag import MAX_DAG_DEPTH, DagError, DagNode, validate


def _chain(length: int, declaration: str) -> tuple[DagNode, ...]:
    template = WorkloadDefinition(
        name="stage",
        execution=CommandExecution(command="python stage.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
    )
    nodes = [DagNode(f"n{index}", template, depends_on=(f"n{index - 1}",) if index else ()) for index in range(length)]
    if declaration == "reverse":
        nodes.reverse()
    elif declaration == "permuted":
        random.Random(103).shuffle(nodes)
    return tuple(nodes)


@pytest.mark.parametrize("declaration", ("root-first", "reverse", "permuted"))
def test_deep_chain_is_refused_independent_of_declaration_order(declaration: str) -> None:
    with pytest.raises(DagError, match="DAG chain is 500 nodes deep, above the 32-node limit"):
        validate(_chain(500, declaration))


@pytest.mark.parametrize("declaration", ("root-first", "reverse", "permuted"))
def test_chain_at_the_depth_limit_retains_topological_order(declaration: str) -> None:
    assert validate(_chain(MAX_DAG_DEPTH, declaration)) == tuple(f"n{index}" for index in range(MAX_DAG_DEPTH))


@pytest.mark.parametrize("declaration", ("root-first", "reverse", "permuted"))
async def test_service_refuses_deep_chain_without_persisting_workloads(declaration: str) -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    principal = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )

    with pytest.raises(DagError, match="DAG chain is 500 nodes deep, above the 32-node limit"):
        await service.submit_dag(
            principal,
            org_id="acme",
            project_id="research",
            nodes=_chain(500, declaration),
            idempotency_key="writer-103-depth",
        )

    async with repository.transaction() as transaction:
        assert await transaction.list_workloads() == []


@pytest.mark.parametrize("declaration", ("root-first", "reverse", "permuted"))
def test_http_maps_deep_chain_refusal_to_400(declaration: str, tmp_path) -> None:
    repository = InMemoryRepository()
    client = TestClient(
        create_app(
            settings=EyvanSettings(
                dev_auth_enabled=True, controller_enabled=False, artifact_store_uri=str(tmp_path / "managed")
            ),
            repository=repository,
        ),
        raise_server_exceptions=False,
    )
    headers = {
        "X-Eyvan-User": "alice",
        "X-Eyvan-Org": "acme",
        "X-Eyvan-Project": "research",
        "X-Eyvan-Roles": "operator",
    }

    response = client.post(
        "/api/v1/dags",
        headers={**headers, "Idempotency-Key": "writer-103-http-depth"},
        json={
            "org_id": "acme",
            "project_id": "research",
            "nodes": [
                {
                    "name": node.name,
                    "template": node.template.model_dump(mode="json"),
                    "depends_on": list(node.depends_on),
                }
                for node in _chain(500, declaration)
            ],
        },
    )

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "DAG chain is 500 nodes deep, above the 32-node limit"
    listed = client.get("/api/v1/workloads", headers=headers, params={"org_id": "acme", "project_id": "research"})
    assert listed.status_code == 200, listed.text
    assert listed.json() == []
