# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CapacityProviderFailureCode, MutationContext
from eyvan.capacity.drivers.gcp import GcpTpuCapacityDriver
from eyvan.capacity.fleet_state import FleetCellRecord, FleetStateStore, LocalBackend
from eyvan.capacity.providers.gcp.queued_resources import GcloudError, GcloudTpuProvider
from eyvan.worker.runtime.native_bootstrap import derive_native_worker_token

UID = "0123456789abcdef0123456789abcdef"


def cell_record() -> FleetCellRecord:
    return FleetCellRecord(
        name="cell-a",
        project="project-a",
        zone="us-central2-b",
        accelerator_type="v6e-8",
        qr_id="cell-a-r1",
        provider_generation=1,
        metadata={"management_uid": UID, "execution_backend": "eyvan-native"},
    )


def qr_payload(*, marked: bool = False, state: str = "ACTIVE") -> dict:
    payload = {
        "name": "projects/project-a/locations/us-central2-b/queuedResources/cell-a-r1",
        "state": {"state": state},
        "tpu": {"nodeSpec": [{"nodeId": "cell-a", "node": {"acceleratorType": "v6e-8"}}]},
    }
    if marked:
        payload["labels"] = {"eyvan-cell": "cell-a", "eyvan-uid": UID}
    return payload


def node_payload() -> dict:
    return {
        "name": "projects/project-a/locations/us-central2-b/nodes/cell-a",
        "state": "READY",
        "acceleratorType": "v6e-8",
        "labels": None,
        "metadata": {},
        "networkEndpoints": [{"ipAddress": "10.2.0.4"}],
    }


def response(payload: dict | None) -> subprocess.CompletedProcess[str]:
    if payload is None:
        return subprocess.CompletedProcess(["gcloud"], 1, "", "NOT_FOUND")
    return subprocess.CompletedProcess(["gcloud"], 0, json.dumps(payload), "")


def observed_provider(qr: dict, node: dict) -> GcloudTpuProvider:
    def runner(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        assert "describe" in args, tuple(args)
        return response(qr if "queued-resources" in args else node)

    return GcloudTpuProvider(runner=runner)


def test_persisted_qr_membership_authorizes_provider_omitted_node_marks() -> None:
    value = cell_record()
    value.state = CapacityStatus.HEALTHY
    observation = observed_provider(qr_payload(), node_payload()).ensure(value)

    assert observation.state is CapacityStatus.HEALTHY
    assert observation.execution_endpoint == "http://10.2.0.4:8100"
    assert observation.worker_endpoints == ("http://10.2.0.4:8100",)
    assert observation.observed_tpu_vms == 1
    assert value.qr_id == "cell-a-r1"
    assert value.provider_generation == 1


@pytest.mark.parametrize("outcome", ["success", "failed-response", "timeout", "already-visible"])
def test_recorded_create_readback_recovers_omitted_qr_marks(outcome: str) -> None:
    landed = False
    qr_reads = 0
    emitted: list[tuple[str, ...]] = []

    def runner(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        nonlocal landed, qr_reads
        emitted.append(tuple(args))
        if "create" in args:
            landed = True
            if outcome == "timeout":
                raise subprocess.TimeoutExpired("gcloud", 5)
            if outcome == "failed-response":
                return subprocess.CompletedProcess(["gcloud"], 1, "", "response lost")
            return response({})
        assert "describe" in args, tuple(args)
        if "tpu-vm" in args:
            return response(None)
        qr_reads += 1
        if outcome == "already-visible" and qr_reads == 2:
            # The QR appears between ensure's first read and create's replay
            # read; no new command should be necessary to recover it.
            landed = True
        return response(qr_payload(state="WAITING_FOR_RESOURCES") if landed else None)

    observation = GcloudTpuProvider(runner=runner).ensure(cell_record())
    assert observation.state is CapacityStatus.WAITING
    assert observation.detail.startswith("queued resource was created")
    assert sum("create" in args for args in emitted) == (0 if outcome == "already-visible" else 1)


@pytest.mark.parametrize("foreign_field", ["qr-uid", "node-uid", "node-cell", "conflicting-metadata"])
def test_explicit_foreign_ownership_overrides_recorded_membership(foreign_field: str) -> None:
    qr = qr_payload()
    node = node_payload()
    if foreign_field == "qr-uid":
        qr["labels"] = {"eyvan-cell": "cell-a", "eyvan-uid": "foreign"}
    elif foreign_field == "node-uid":
        node["labels"] = {"eyvan-uid": "foreign"}
    elif foreign_field == "node-cell":
        node["labels"] = {"eyvan-uid": UID, "eyvan-cell": "foreign-cell"}
    else:
        node["labels"] = {"eyvan-uid": UID}
        node["metadata"] = {"eyvan-uid": "foreign"}

    with pytest.raises(GcloudError, match="not owned"):
        observed_provider(qr, node).ensure(cell_record())


@pytest.mark.parametrize(
    "invalid", ["unrecorded-qr", "wrong-qr-id", "outside-qr", "wrong-type", "missing-type", "missing-membership"]
)
def test_omitted_marks_do_not_authorize_arbitrary_resources(invalid: str) -> None:
    qr = qr_payload()
    node = node_payload()
    value = cell_record()
    if invalid == "unrecorded-qr":
        value.qr_id = None
    elif invalid == "wrong-qr-id":
        qr["name"] = "projects/project-a/locations/us-central2-b/queuedResources/other-r1"
    elif invalid == "outside-qr":
        qr["tpu"]["nodeSpec"][0]["nodeId"] = "outside-cell"
    elif invalid == "wrong-type":
        node["acceleratorType"] = "v4-8"
    elif invalid == "missing-type":
        node.pop("acceleratorType")
    else:
        qr["tpu"]["nodeSpec"] = []

    with pytest.raises(GcloudError):
        observed_provider(qr, node).ensure(value)


@pytest.mark.parametrize("unexpected_prefix", [False, True])
def test_unmarked_multislice_children_require_validated_qr_prefix_membership(unexpected_prefix: bool) -> None:
    value = cell_record()
    value.metadata["tpu_config"] = {"node_count": 2}
    qr = qr_payload()
    prefix = "foreign-node" if unexpected_prefix else "cell-a-r1-node"
    qr["tpu"]["nodeSpec"] = [{"nodeId": f"{prefix}-{index}", "node": {"acceleratorType": "v6e-8"}} for index in range(2)]

    def runner(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        assert "describe" in args, tuple(args)
        if "queued-resources" in args:
            return response(qr)
        node_id = args[args.index("describe") + 1]
        node = node_payload()
        node["name"] = f"projects/project-a/locations/us-central2-b/nodes/{node_id}"
        node["networkEndpoints"] = [{"ipAddress": "10.2.0.4" if node_id.endswith("-0") else "10.2.0.5"}]
        return response(node)

    provider = GcloudTpuProvider(runner=runner)
    if unexpected_prefix:
        with pytest.raises(GcloudError, match="expected TPU node prefix"):
            provider.ensure(value)
    else:
        observation = provider.ensure(value)
        assert observation.state is CapacityStatus.HEALTHY
        assert observation.worker_endpoints == ("http://10.2.0.4:8100", "http://10.2.0.5:8100")
        assert observation.observed_tpu_vms == 2


def test_direct_create_without_recorded_authority_does_not_adopt_unmarked_qr() -> None:
    provider = observed_provider(qr_payload(), node_payload())
    value = cell_record()
    with pytest.raises(GcloudError, match="not owned"):
        provider.create(provider._spec(value), qr_id="cell-a-r1")


def test_marked_active_ready_allocation_remains_healthy() -> None:
    node = node_payload()
    node["metadata"] = {"eyvan-uid": UID}
    observation = observed_provider(qr_payload(marked=True), node).ensure(cell_record())
    assert observation.state is CapacityStatus.HEALTHY
    assert observation.execution_endpoint == "http://10.2.0.4:8100"


@pytest.mark.parametrize(
    "previous_state",
    [CapacityStatus.HEALTHY, CapacityStatus.ACTIVE, CapacityStatus.WAITING, CapacityStatus.NEEDS_RECREATE],
)
def test_missing_previously_observed_tpu_requires_fenced_recreation(previous_state: CapacityStatus) -> None:
    emitted: list[tuple[str, ...]] = []
    landed = False

    def runner(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        nonlocal landed
        emitted.append(tuple(args))
        if "create" in args:
            landed = True
            return response({})
        return response(qr_payload(marked=True, state="WAITING_FOR_RESOURCES") if landed else None)

    value = cell_record()
    value.state = previous_state
    observation = GcloudTpuProvider(runner=runner).ensure(value)

    assert observation.state is CapacityStatus.NEEDS_RECREATE
    assert observation.execution_endpoint is None
    assert observation.worker_endpoints == ()
    assert not any("create" in args for args in emitted)
    assert (value.qr_id, value.provider_generation) == ("cell-a-r1", 1)


@pytest.mark.parametrize("previous_state", [CapacityStatus.UNOBSERVED, CapacityStatus.RECREATING])
def test_missing_new_or_planned_tpu_incarnation_can_be_created(previous_state: CapacityStatus) -> None:
    created: list[str] = []

    def runner(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if "create" in args:
            created.append(args[args.index("create") + 1])
            return response({})
        if "tpu-vm" in args:
            return response(None)
        return response(qr_payload(state="WAITING_FOR_RESOURCES") if created else None)

    value = cell_record()
    value.state = previous_state
    observation = GcloudTpuProvider(runner=runner).ensure(value)
    assert observation.state is CapacityStatus.WAITING
    assert created == ["cell-a-r1"]


def test_missing_tpu_real_driver_persists_new_incarnation_and_derives_new_token_before_create(tmp_path: Path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    cloud: dict[str, dict] = {}
    created: list[str] = []
    provisioned_generations: list[tuple[int, int]] = []
    tokens: list[str] = []

    class OfflineSecrets:
        def ensure(self, **kwargs) -> str:
            tokens.append(kwargs["value"])
            return f"projects/project-a/secrets/offline-test/versions/{len(tokens)}"

        def delete(self, **kwargs) -> None:
            raise AssertionError("secret deletion is not part of this recovery")

    def runner(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if "create" in args:
            qr_id = args[args.index("create") + 1]
            current = state.get("cell-a")
            assert current is not None
            provisioned_generations.append(
                (current.provider_generation, current.metadata["native_worker_bootstrap_generation"])
            )
            created.append(qr_id)
            payload = qr_payload(state="WAITING_FOR_RESOURCES")
            payload["name"] = f"projects/project-a/locations/us-central2-b/queuedResources/{qr_id}"
            cloud[qr_id] = payload
            return response({})
        assert "describe" in args, tuple(args)
        if "queued-resources" in args:
            return response(cloud.get(args[args.index("describe") + 1]))
        if any(payload["state"]["state"] == "ACTIVE" for payload in cloud.values()):
            return response(node_payload())
        return response(None)

    driver = GcpTpuCapacityDriver(
        state,
        provider=GcloudTpuProvider(runner=runner),
        execution_backend="eyvan-native",
        native_worker_package="gs://offline-test/eyvan.whl",
        native_worker_package_sha256="a" * 64,
        native_worker_shared_secret="synthetic-test-root-not-a-credential",
        native_worker_secrets=OfflineSecrets(),
    )

    def context(key: str) -> MutationContext:
        return MutationContext(idempotency_key=key, generation=1, fencing_token="test-fence")

    initial = driver.apply_capacity(
        "cell-a",
        {
            "project": "project-a",
            "zone": "us-central2-b",
            "accelerator_type": "v6e-8",
            "capacity_class": "spot",
            "executor": "eyvan-native",
            "provider_options": {
                "service_account": "offline-worker@project-a.iam.gserviceaccount.com",
                "bootstrap_egress": "public-internet",
            },
        },
        context("create"),
    )
    assert initial.state is CapacityStatus.WAITING
    cloud["cell-a-r1"]["state"]["state"] = "ACTIVE"
    assert driver.reconcile("cell-a", context("observe")).state is CapacityStatus.HEALTHY
    prior = state.get("cell-a")
    assert prior is not None
    prior_token = derive_native_worker_token(
        "synthetic-test-root-not-a-credential", prior.metadata["management_uid"], provider_generation=1
    )
    assert bool(tokens[0] == prior_token)

    cloud.clear()
    missing = driver.reconcile("cell-a", context("missing"))
    assert missing.state is CapacityStatus.NEEDS_RECREATE
    assert created == ["cell-a-r1"]
    assert len(tokens) == 1

    replacement = driver.reconcile("cell-a", context("replace"))
    persisted = state.get("cell-a")
    assert replacement.state is CapacityStatus.WAITING
    assert persisted is not None
    assert (persisted.qr_id, persisted.provider_generation) == ("cell-a-r2", 2)
    assert created == ["cell-a-r1", "cell-a-r2"]
    assert provisioned_generations == [(1, 1), (2, 2)]
    assert "recreate_intent" not in persisted.metadata
    assert persisted.generation == 1
    assert persisted.fencing_token == "test-fence"
    assert len(set(tokens)) == 2
    next_token = derive_native_worker_token(
        "synthetic-test-root-not-a-credential", persisted.metadata["management_uid"], provider_generation=2
    )
    assert bool(tokens[-1] == next_token)


@pytest.mark.parametrize("qr_state", ["FAILED", "DELETING", "SUSPENDING", "SUSPENDED"])
@pytest.mark.parametrize("node_state", ["READY", "CREATING"])
def test_terminal_qr_overrides_visible_children_and_preserves_failure(qr_state: str, node_state: str) -> None:
    qr = qr_payload(marked=True, state=qr_state)
    if qr_state == "FAILED":
        qr["state"]["failedData"] = {"error": {"message": "Quota exceeded for TPU capacity", "code": 8}}
    node = node_payload()
    node["state"] = node_state
    node["metadata"] = {"eyvan-uid": UID}

    observation = observed_provider(qr, node).ensure(cell_record())

    assert observation.state is CapacityStatus.NEEDS_RECREATE
    assert observation.execution_endpoint is None
    assert observation.worker_endpoints == ()
    assert observation.head_ip is None
    assert observation.observed_tpu_vms is None
    assert observation.runtime_process_count is None
    assert qr_state in observation.detail
    if qr_state == "FAILED":
        assert "Quota exceeded for TPU capacity (code 8)" in observation.detail
        assert observation.failure_code is CapacityProviderFailureCode.QUOTA_EXCEEDED
    else:
        assert observation.failure_code is None


def test_nonterminal_qr_and_creating_child_remain_provisioning() -> None:
    node = node_payload()
    node["state"] = "CREATING"
    node["metadata"] = {"eyvan-uid": UID}
    observation = observed_provider(qr_payload(marked=True, state="PROVISIONING"), node).ensure(cell_record())
    assert observation.state is CapacityStatus.PROVISIONING
    assert observation.execution_endpoint is None
    assert observation.worker_endpoints == ()
