# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Observable regressions for workload CLI addressing and admission contracts."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from eyvan.records import (
    AuditEvent,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    Principal,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.commands.workloads import WorkloadCommandService, run_workload, workload
from eyvan.sdk.contexts import EyvanContextConfig, FileContextStore
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService

FULL_ID = "wl_" + "a" * 32


def _record(identifier=FULL_ID, name="wl_training", state=WorkloadState.PENDING):
    return WorkloadRecord(
        id=identifier,
        org_id="acme",
        project_id="research",
        created_by="alice",
        spec=WorkloadDefinition(name=name, execution=CommandExecution(command="true")),
        state=state,
    ).model_dump(mode="json")


class Plane:
    """Exact-ID local transport: bad addressing fails rather than succeeding."""

    def __init__(self, *records, denied=False):
        self.records = list(records)
        self.denied = denied
        self.calls = []

    def request(self, method, path, *, params=None, json=None, headers=None):
        self.calls.append((method, path, params, headers))
        if method == "GET" and path == "/api/v1/workloads":
            if self.denied:
                raise PermissionError("listing denied")
            assert params == {"org_id": "acme", "project_id": "research"}
            return self.records
        if method == "POST":
            identifier = path.split("/")[-2]
            return next(item for item in self.records if item["id"] == identifier)
        raise AssertionError((method, path))


def _session(plane, tmp_path: Path):
    return CliSession(
        client=plane,
        identity=ClientIdentity(dev_user="alice"),
        context_store=FileContextStore(tmp_path / "contexts.yaml"),
        context_config=EyvanContextConfig(),
        base_url="http://example.invalid",
        org_id="acme",
        project_id="research",
        overrides=ContextOverrides(),
    )


@pytest.mark.parametrize("reference", ["wl_aaaa", "wl_training"])
def test_prefix_resolves_id_and_wl_name(tmp_path, reference):
    plane = Plane(_record())
    result = CliRunner().invoke(workload, ["cancel", reference], obj=_session(plane, tmp_path))
    assert result.exit_code == 0, result.output
    assert [(method, path) for method, path, _, _ in plane.calls] == [
        ("GET", "/api/v1/workloads"),
        ("POST", f"/api/v1/workloads/{FULL_ID}/cancel"),
    ]


def test_prefix_ambiguous_id_refuses_mutation(tmp_path):
    plane = Plane(_record(), _record("wl_aaaa" + "b" * 28, "other"))
    result = CliRunner().invoke(workload, ["cancel", "wl_aaaa"], obj=_session(plane, tmp_path))
    assert result.exit_code != 0
    assert "matches 2 workloads" in result.output
    assert all(method == "GET" for method, _, _, _ in plane.calls)


def test_prefix_full_id_and_denied_listing_controls(tmp_path):
    plane = Plane(_record(), denied=True)
    session = _session(plane, tmp_path)
    result = CliRunner().invoke(workload, ["cancel", FULL_ID], obj=session)
    assert result.exit_code == 0, result.output
    assert len(plane.calls) == 1
    # Listing genuinely unavailable still leaves exact route judgment to server.
    from eyvan.sdk.commands.workloads import WorkloadCommandService

    assert WorkloadCommandService(session).resolve("wl_aaaa", org_id=None, project_id=None) == "wl_aaaa"


class ServicePlane:
    """Use actual authorization, durable memory records and replay semantics."""

    def __init__(self):
        self.repository = InMemoryRepository()
        self.service = ControlPlaneService(self.repository)
        self.operator = Principal(
            subject="alice", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),)
        )
        self.admin = Principal(subject="system", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
        self.keys = []
        self.replays = []
        self.calls = []

    def request(self, method, path, *, params=None, json=None, headers=None):
        self.calls.append((method, path, params, headers))
        if method == "GET" and path == "/api/v1/events":
            return [
                item.model_dump(mode="json") for item in asyncio.run(self.service.list_events(self.operator, **params))
            ]
        if method == "GET" and path == "/api/v1/leases":
            return [
                item.model_dump(mode="json") for item in asyncio.run(self.service.list_leases(self.operator, **params))
            ]
        if method == "GET" and path.startswith("/api/v1/workloads/"):
            return self.current(path.split("/")[-1]).model_dump(mode="json")
        if method == "GET" and path == "/api/v1/workloads":
            return [
                item.model_dump(mode="json")
                for item in asyncio.run(self.service.list_workloads(self.operator, **params))
            ]
        if method == "POST":
            identifier, action = path.split("/")[-2:]
            key = headers["Idempotency-Key"]
            self.keys.append(key)
            item, replay = asyncio.run(
                getattr(self.service, f"{action}_workload")(self.operator, identifier, idempotency_key=key)
            )
            self.replays.append(replay)
            return item.model_dump(mode="json")
        raise AssertionError((method, path))

    def current(self, identifier):
        return asyncio.run(self.service.get_workload(self.operator, identifier))

    def seed(self, record, lease=None):
        async def seed_records():
            async with self.repository.transaction() as tx:
                await tx.put_workload(record)
                if lease is not None:
                    await tx.put_lease(lease)

        asyncio.run(seed_records())


@pytest.mark.parametrize("action", ["cancel", "retry", "checkpoint"])
@pytest.mark.parametrize("explicit", [False, True])
def test_intent_independent_prefix_actions_mutate_again(tmp_path, action, explicit):
    plane = ServicePlane()
    spec = WorkloadDefinition(
        name="sweep-one",
        execution=CommandExecution(command="true"),
        max_retries=4,
        max_failure_retries=4,
        checkpointable=True,
        checkpoint_hook="true",
        checkpoint_output_uri="gs://test-checkpoints/train",
    )
    item, _ = asyncio.run(
        plane.service.submit_workload(
            plane.operator, org_id="acme", project_id="research", spec=spec, idempotency_key="submit"
        )
    )
    if action == "retry":
        asyncio.run(plane.service.transition_workload(plane.admin, item.id, WorkloadState.FAILED))
    if action == "checkpoint":
        item = item.model_copy(
            update={"state": WorkloadState.RUNNING, "lease_id": "lease-one", "remote_id": "execution"}
        )
        lease = LeaseRecord(
            id="lease-one",
            workload_id=item.id,
            cell_id="cell-one",
            org_id="acme",
            project_id="research",
            state=LeaseState.ACTIVE,
            resources=spec.gang,
            fencing_token="fence",
            expires_at=utcnow() + timedelta(minutes=5),
        )
        plane.seed(WorkloadRecord.model_validate(item.model_dump()), lease)
    session = _session(plane, tmp_path)

    def invoke(key):
        args = [action, "--prefix", "sweep-"]
        if explicit:
            args += ["--idempotency-key", key]
        result = CliRunner().invoke(workload, args, obj=session)
        assert result.exit_code == 0, (result.output, result.exception)

    invoke("intent-A")
    if explicit:
        invoke("intent-A")
        assert plane.replays[-1] is True
        assert plane.keys[-1] == plane.keys[-2]
    if action == "cancel":
        asyncio.run(plane.service.retry_workload(plane.operator, item.id, idempotency_key="manual-retry"))
    elif action == "retry":
        asyncio.run(plane.service.transition_workload(plane.admin, item.id, WorkloadState.FAILED))
    else:
        asyncio.run(plane.service.transition_workload(plane.admin, item.id, WorkloadState.RUNNING))
    invoke("intent-B")
    assert plane.replays[-1] is False, "a new operator intent must not replay the old action"
    assert plane.keys[-1] != plane.keys[0]
    current = plane.current(item.id)
    assert (
        current.state
        is {
            "cancel": WorkloadState.CANCELLED,
            "retry": WorkloadState.PENDING,
            "checkpoint": WorkloadState.CHECKPOINTING,
        }[action]
    )
    if action == "retry":
        assert current.attempt == 3
    if action == "cancel":
        assert current.generation == 2


@pytest.mark.parametrize("selector", [["--cpu"], ["--gpu", "nvidia-tesla-a100"], ["--tpu", "v6e-8"]])
@pytest.mark.parametrize("command", [["hostname"], ["echo", "ok"]])
@pytest.mark.parametrize("separator", [[], ["--"]])
def test_run_hardware_command_words_reach_submission(tmp_path, monkeypatch, selector, command, separator):
    captured = []
    monkeypatch.setattr(WorkloadCommandService, "run", lambda self, **kwargs: captured.append(kwargs) or 0)
    result = CliRunner().invoke(run_workload, [*selector, *separator, *command], obj=_session(Plane(), tmp_path))
    assert result.exit_code == 0, result.output
    assert captured[0]["command"] == tuple(command)
    assert captured[0]["target_value"] == ("cpu" if selector == ["--cpu"] else selector[1])


@pytest.mark.parametrize("selector", [["--cpu"], ["--gpu", "nvidia-tesla-a100"], ["--tpu", "v6e-8"], ["demo/cpu"]])
def test_run_missing_effective_command_refused(tmp_path, monkeypatch, selector):
    captured = []
    monkeypatch.setattr(WorkloadCommandService, "run", lambda self, **kwargs: captured.append(kwargs) or 0)
    result = CliRunner().invoke(run_workload, selector, obj=_session(Plane(), tmp_path))
    assert result.exit_code == 2
    assert "COMMAND" in result.output
    assert captured == []


def test_run_pool_single_executable_control(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(WorkloadCommandService, "run", lambda self, **kwargs: captured.append(kwargs) or 0)
    result = CliRunner().invoke(run_workload, ["demo/cpu", "--", "hostname"], obj=_session(Plane(), tmp_path))
    assert result.exit_code == 0, result.output
    assert captured[0]["target_value"] == "demo/cpu"
    assert captured[0]["command"] == ("hostname",)


@pytest.mark.parametrize(
    "field,value", [("replicas", 1.9), ("gpu", 0.5), ("tpu", -0.5), ("gpu", -0.5), ("replicas", -0.5)]
)
def test_dag_invalid_counts_are_not_truncated(tmp_path, monkeypatch, field, value):
    captured = []
    monkeypatch.setattr(WorkloadCommandService, "submit_dag", lambda self, request, **kwargs: captured.append(request))
    node = {"name": "train", "command": "true", "resources": {"cpu": 1}}
    if field == "replicas":
        node[field] = value
    else:
        node["resources"][field] = value
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps({"nodes": [node]}))
    result = CliRunner().invoke(workload, ["submit-dag", "--file", str(path)], obj=_session(Plane(), tmp_path))
    assert result.exit_code != 0, "malformed counts must fail before submission"
    assert "pipeline node 'train'" in result.output
    assert captured == []


def test_dag_integer_count_control(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(WorkloadCommandService, "submit_dag", lambda self, request, **kwargs: captured.append(request))
    path = tmp_path / "pipeline.json"
    path.write_text(
        json.dumps(
            {"nodes": [{"name": "train", "command": "true", "replicas": 2, "resources": {"cpu": 1, "gpu": 2, "tpu": 0}}]}
        )
    )
    result = CliRunner().invoke(workload, ["submit-dag", "--file", str(path)], obj=_session(Plane(), tmp_path))
    assert result.exit_code == 0, result.output
    gang = captured[0].nodes[0].template.gang
    assert gang.replicas == 2
    assert (gang.per_replica.gpu, gang.per_replica.tpu) == (2, 0)


@pytest.mark.parametrize(
    "reference",
    [
        "my-package @ https://example.invalid/pkg.whl#sha256=" + "b" * 64,
        "my-package @ git+https://example.invalid/repo.git#subdirectory=python",
    ],
)
def test_requirements_fragments_preserved_like_with_flag(tmp_path, monkeypatch, reference):
    captured = []
    monkeypatch.setattr(WorkloadCommandService, "run", lambda self, **kwargs: captured.append(kwargs) or 0)
    path = tmp_path / "requirements.txt"
    path.write_text(f"# full-line comment\n\n{reference} # explanatory comment\n")
    session = _session(Plane(), tmp_path)
    for options in (["--with", reference], ["--with-requirements", str(path)]):
        result = CliRunner().invoke(run_workload, ["--cpu", *options, "--", "echo", "ok"], obj=session)
        assert result.exit_code == 0, result.output
    assert captured[0]["dependencies"] == (reference,)
    assert captured[1]["dependencies"] == captured[0]["dependencies"]


def test_requirements_comments_markers_and_option_refusal(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(WorkloadCommandService, "run", lambda self, **kwargs: captured.append(kwargs) or 0)
    path = tmp_path / "requirements.txt"
    marker = 'numpy>=1; python_version >= "3.11"'
    path.write_text(f"  # comment\n\nnumpy>=1 # explanation\n{marker}\n")
    session = _session(Plane(), tmp_path)
    result = CliRunner().invoke(run_workload, ["--cpu", "--with-requirements", str(path), "echo", "ok"], obj=session)
    assert result.exit_code == 0, result.output
    assert captured[0]["dependencies"] == ("numpy>=1", marker)
    captured.clear()
    path.write_text("--index-url https://example.invalid\n")
    result = CliRunner().invoke(run_workload, ["--cpu", "--with-requirements", str(path), "echo", "ok"], obj=session)
    assert result.exit_code == 2
    assert "installer option" in result.output
    assert captured == []


@pytest.mark.parametrize("new_attempt", [False, True])
def test_timeline_keeps_released_prior_attempt_after_retry(tmp_path, new_attempt):
    plane = ServicePlane()
    spec = WorkloadDefinition(name="history", execution=CommandExecution(command="true"), max_failure_retries=4)
    item, _ = asyncio.run(
        plane.service.submit_workload(
            plane.operator, org_id="acme", project_id="research", spec=spec, idempotency_key="submit"
        )
    )
    item = WorkloadRecord.model_validate(
        item.model_copy(
            update={
                "state": WorkloadState.RUNNING,
                "lease_id": "lease-old",
                "lease_ids": ("lease-old",),
                "remote_id": "execution-old",
            }
        ).model_dump()
    )
    old = LeaseRecord(
        id="lease-old",
        workload_id=item.id,
        cell_id="cell-old",
        org_id="acme",
        project_id="research",
        state=LeaseState.ACTIVE,
        resources=spec.gang,
        fencing_token="old-fence",
        expires_at=utcnow() + timedelta(minutes=5),
    )
    plane.seed(item, old)
    asyncio.run(
        plane.service.settle_terminal_workload(plane.admin, item.id, WorkloadState.FAILED, action="workload.failed")
    )
    session = _session(plane, tmp_path)
    before = CliRunner().invoke(workload, ["timeline", item.id], obj=session)
    assert before.exit_code == 0, (before.output, before.exception)
    assert before.output.count("capacity released") == 1
    retried, _ = asyncio.run(plane.service.retry_workload(plane.operator, item.id, idempotency_key="retry"))
    assert retried.lease_id is None and retried.lease_ids == ()
    persisted = asyncio.run(
        plane.service.list_leases(plane.operator, org_id="acme", project_id="research", workload_id=item.id)
    )
    assert len(persisted) == 1 and persisted[0].settled_at is not None
    if new_attempt:
        current = WorkloadRecord.model_validate(
            retried.model_copy(
                update={
                    "state": WorkloadState.RUNNING,
                    "lease_id": "lease-new",
                    "lease_ids": ("lease-new",),
                    "remote_id": "execution-new",
                }
            ).model_dump()
        )
        lease = old.model_copy(update={"id": "lease-new", "generation": retried.generation, "cell_id": "cell-new"})
        plane.seed(current, LeaseRecord.model_validate(lease.model_dump()))

        async def record_placement_event():
            async with plane.repository.transaction() as tx:
                await tx.append_event(
                    AuditEvent(
                        id="evt-new-placement",
                        actor="system",
                        org_id="acme",
                        project_id="research",
                        entity_type="lease",
                        entity_id="lease-new",
                        action="lease.created",
                    )
                )

        asyncio.run(record_placement_event())
    plane.calls.clear()
    after = CliRunner().invoke(workload, ["timeline", item.id], obj=session)
    assert after.exit_code == 0, (after.output, after.exception)
    assert after.output.count("capacity released") == 1, "prior lease settlement must remain visible after retry"
    if new_attempt:
        assert after.output.count("capacity leased") == 1
    historical_queries = [params for _, path, params, _ in plane.calls if path == "/api/v1/leases"]
    assert historical_queries == [{"org_id": "acme", "project_id": "research", "workload_id": item.id}]
    lease_entities = [
        params["entity_id"]
        for _, path, params, _ in plane.calls
        if path == "/api/v1/events" and params["entity_type"] == "lease"
    ]
    assert len(lease_entities) == len(set(lease_entities))
    assert set(lease_entities) == ({"lease-old", "lease-new"} if new_attempt else {"lease-old"})


@pytest.mark.parametrize("action", ["cancel", "checkpoint", "retry"])
@pytest.mark.parametrize("count", [1, 2])
def test_json_prefix_outputs_one_document_for_batch(tmp_path, action, count):
    records = [_record(FULL_ID, "sweep-one"), _record("wl_" + "b" * 32, "sweep-two")][:count]
    plane = Plane(*records)
    result = CliRunner().invoke(
        workload, [action, "--prefix", "sweep-", "--yes", "--json"], obj=_session(plane, tmp_path)
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, list), "prefix JSON has one consistent array framing, even for a single match"
    assert payload == records
    assert len([call for call in plane.calls if call[0] == "POST"]) == count


@pytest.mark.parametrize("action", ["cancel", "checkpoint", "retry"])
def test_json_direct_id_still_outputs_record(tmp_path, action):
    record = _record()
    result = CliRunner().invoke(workload, [action, FULL_ID, "--json"], obj=_session(Plane(record), tmp_path))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == record
