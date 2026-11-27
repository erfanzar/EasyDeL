# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Existing TPU attachment is fenced, truthful, and strictly read-only."""

from __future__ import annotations

import copy
from contextlib import contextmanager

import httpx
import pytest

from eyvan.capacity import CapacityStatus
from eyvan.capacity.conformance import ProviderConformanceHarness, run_conformance
from eyvan.capacity.drivers.base import (
    CELL_ACCELERATOR_METADATA_KEY,
    CapacityProviderError,
    CapacityProviderFailureCode,
    MutationContext,
    StaleMutationError,
    capacity_backend_agent_family,
)
from eyvan.capacity.drivers.gcp import GcpCapacityDriver
from eyvan.capacity.drivers.gcp_attach import GcpAttachCapacityDriver
from eyvan.capacity.drivers.registry import DEFAULT_CAPACITY_BACKENDS
from eyvan.capacity.fleet_config import EyvanFleetConfig
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.capacity.providers.gcp.runtime.supervisor import requires_native_gcp_runtime
from eyvan.capacity.providers.gcp.tpu_nodes import TpuNodeApi, attach_management_uid
from eyvan.config import EyvanSettings
from eyvan.records import CallableExecution, CapacityPoolRecord, Principal, Role, RoleGrant
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.errors import ConflictError


def _context(key="apply", generation=1, token="fence-1"):
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


def _document(*, state="READY", spot=True):
    return {
        "name": "projects/project-a/locations/us-east5-a/nodes/existing-tpu",
        "id": "123456",
        "acceleratorType": "v6e-16",
        "state": state,
        "health": "HEALTHY",
        "schedulingConfig": {"spot": spot},
        "networkEndpoints": [
            {"ipAddress": "10.0.0.2", "accessConfig": {"externalIp": "192.0.2.2"}},
            {"ipAddress": "10.0.0.3", "accessConfig": {"externalIp": "192.0.2.3"}},
        ],
    }


def _desired():
    return {
        "provider": "gcp",
        "capacity_backend": "gcp-attach",
        "executor": "eyvan-native",
        "project": "project-a",
        "zone": "us-east5-a",
        "accelerator_type": "v6e-16",
        "capacity_class": "spot",
        "provider_options": {"tpu_node": "existing-tpu", "tpu_node_id": "123456", "endpoint_mode": "public"},
    }


class Substrate:
    """Mutable provider documents; every single transport call must be a GET."""

    def __init__(self):
        self.document = _document()
        self.unreachable = False
        self.status = 200
        self.requests = []
        self.api = TpuNodeApi(token_source=lambda: "tok", transport=httpx.MockTransport(self.handle))

    def handle(self, request):
        assert request.method == "GET", f"attach attempted cloud mutation: {request.method}"
        assert request.url.path == "/v2/projects/project-a/locations/us-east5-a/nodes/existing-tpu"
        self.requests.append(request)
        if self.unreachable:
            raise httpx.ConnectError("offline", request=request)
        if self.document is None:
            return httpx.Response(404, json={})
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "denied"}})
        return httpx.Response(200, json=self.document)


def _driver(tmp_path, substrate):
    return GcpAttachCapacityDriver(
        FleetStateStore(LocalBackend(tmp_path / "fleet.json")), api=substrate.api, native_worker_port=8111
    )


class AttachHarness(ProviderConformanceHarness):
    def __init__(self, root):
        self.root = root
        self.substrates = {}
        # A counter, not len(substrates): substrates is keyed by id(driver),
        # and CPython reuses the id of a collected driver, which overwrote an
        # entry and handed the next driver an earlier driver's state directory
        # (seen on CI: "generation 3 is older than cell 'cell-a' generation 7").
        self._created = 0

    def new_driver(self):
        substrate = Substrate()
        self._created += 1
        driver = _driver(self.root / str(self._created), substrate)
        self.substrates[id(driver)] = substrate
        return driver

    def desired_capacity(self, cell_id):
        return _desired()

    def provider_incarnations(self, driver, cell_id):
        record = driver.state.get(cell_id)
        return int(record is not None and record.metadata.get("capacity_kind") == "attach")

    def leak_previous_incarnation(self, driver, cell_id):
        self.substrates[id(driver)].document["id"] = "654321"

    @contextmanager
    def provider_unreachable(self, driver):
        substrate = self.substrates[id(driver)]
        substrate.unreachable = True
        try:
            yield
        finally:
            substrate.unreachable = False

    @contextmanager
    def provider_rejects_permanently(self, driver):
        substrate = self.substrates[id(driver)]
        substrate.status = 403
        try:
            yield
        finally:
            substrate.status = 200


class _RecordingIngress:
    """The profile ingress manager's contract: ensure one rule, return its tag."""

    def __init__(self, network_tag: str = "eyvan-nw-profile", error: Exception | None = None) -> None:
        self.network_tag = network_tag
        self.error = error
        self.calls: list[tuple[str, str | None, int]] = []

    def ensure(self, *, project: str, network: str | None, port: int) -> str:
        self.calls.append((project, network, port))
        if self.error is not None:
            raise self.error
        return self.network_tag


@pytest.mark.parametrize(
    ("tagged", "endpoint_mode", "expected_calls"),
    [(True, "public", 1), (False, "public", 0), (True, "internal", 0)],
)
def test_reconcile_re_ensures_the_profile_ingress_rule_of_a_public_claim(
    tmp_path, tagged, endpoint_mode, expected_calls
):
    """Recreate a missing profile ingress rule when reconciling a tagged public attached TPU."""

    substrate = Substrate()
    substrate.document["networkConfig"] = {"network": "projects/project-a/global/networks/default"}
    if tagged:
        substrate.document["tags"] = ["eyvan-nw-profile", "other-tag"]
    ingress = _RecordingIngress()
    driver = GcpAttachCapacityDriver(
        FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        api=substrate.api,
        native_worker_port=8111,
        native_worker_public_ingress=ingress,
    )
    desired = _desired()
    desired["provider_options"] = {**desired["provider_options"], "endpoint_mode": endpoint_mode}
    driver.apply_capacity("cell", desired, _context())
    assert ingress.calls == []  # apply never touches the firewall; connect already did

    observation = driver.reconcile("cell", _context("reconcile"))

    assert observation.state is CapacityStatus.HEALTHY
    assert len(ingress.calls) == expected_calls
    if expected_calls:
        assert ingress.calls[0] == ("project-a", "projects/project-a/global/networks/default", 8111)
    # Still strictly read-only toward the TPU API.
    assert all(request.method == "GET" for request in substrate.requests)


def test_a_failing_ingress_refresh_does_not_fail_the_observation(tmp_path):
    substrate = Substrate()
    substrate.document["tags"] = ["eyvan-nw-profile"]
    ingress = _RecordingIngress(
        error=CapacityProviderError("gcloud timed out", code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE)
    )
    driver = GcpAttachCapacityDriver(
        FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        api=substrate.api,
        native_worker_port=8111,
        native_worker_public_ingress=ingress,
    )
    driver.apply_capacity("cell", _desired(), _context())

    assert driver.reconcile("cell", _context("reconcile")).state is CapacityStatus.HEALTHY
    assert len(ingress.calls) == 1


def test_full_provider_conformance(tmp_path):
    harness = AttachHarness(tmp_path)
    report = run_conformance(harness)
    assert len(report.results) == 12
    report.require()
    assert all(request.method == "GET" for substrate in harness.substrates.values() for request in substrate.requests)


@pytest.mark.parametrize(
    ("state", "spot", "expected", "failure"),
    [
        ("READY", True, CapacityStatus.HEALTHY, None),
        *[
            (state, True, CapacityStatus.PROVISIONING, None)
            for state in ("CREATING", "STARTING", "RESTARTING", "REPAIRING", "REIMAGING", "UNHIDING")
        ],
        ("PREEMPTED", True, CapacityStatus.NEEDS_RECREATE, CapacityProviderFailureCode.PREEMPTED),
        ("DELETING", True, CapacityStatus.NEEDS_RECREATE, CapacityProviderFailureCode.PREEMPTED),
        ("DELETING", False, CapacityStatus.NEEDS_RECREATE, None),
        *[
            (state, True, CapacityStatus.NEEDS_RECREATE, None)
            for state in ("STOPPED", "STOPPING", "TERMINATED", "HIDING", "HIDDEN")
        ],
        ("UNKNOWN", True, CapacityStatus.UNAVAILABLE, None),
        ("STATE_UNSPECIFIED", True, CapacityStatus.UNAVAILABLE, None),
    ],
)
def test_every_provider_state(tmp_path, state, spot, expected, failure):
    substrate = Substrate()
    substrate.document = _document(state=state, spot=spot)
    desired = _desired()
    desired["capacity_class"] = "spot" if spot else "on_demand"
    driver = _driver(tmp_path, substrate)
    observation = driver.apply_capacity("cell", desired, _context())
    assert observation.state is expected
    assert observation.failure_code is failure
    assert driver.observe("cell").state is expected
    assert driver.reconcile("cell", _context("reconcile")).state is expected
    if expected is not CapacityStatus.HEALTHY:
        assert observation.execution_endpoint is None
        assert not observation.metadata.get("native_worker_endpoints")


@pytest.mark.parametrize("health", [None, "HEALTHY", "UNHEALTHY", "TIMEOUT", "UNKNOWN"])
def test_ready_health(tmp_path, health):
    substrate = Substrate()
    substrate.document["health"] = health
    observation = _driver(tmp_path, substrate).apply_capacity("cell", _desired(), _context())
    assert observation.state is (CapacityStatus.HEALTHY if health in (None, "HEALTHY") else CapacityStatus.UNAVAILABLE)
    if health not in (None, "HEALTHY"):
        assert observation.execution_endpoint is None


@pytest.mark.parametrize("mode", ["public", "internal"])
def test_complete_ordered_endpoints_and_runtime_metadata(tmp_path, mode):
    substrate = Substrate()
    desired = _desired()
    desired["provider_options"]["endpoint_mode"] = mode
    driver = _driver(tmp_path, substrate)
    observation = driver.apply_capacity("cell", desired, _context())
    prefix = "192.0.2" if mode == "public" else "10.0.0"
    assert observation.metadata["native_worker_endpoints"] == [f"http://{prefix}.2:8111", f"http://{prefix}.3:8111"]
    assert observation.execution_endpoint == f"http://{prefix}.2:8111"
    assert observation.observed_tpu_vms == observation.runtime_process_count == 2
    assert observation.metadata["tpu_config"] == {"node_count": 1, "accelerator_type": "v6e-16"}
    assert observation.metadata["provider"] == "gcp"
    assert observation.metadata["project"] == "project-a"
    assert observation.metadata["region"] == "us-east5"
    assert observation.metadata["zone"] == "us-east5-a"
    assert observation.metadata["resource_id"] == "gcp-attach:project-a/us-east5-a/existing-tpu"
    assert observation.metadata["capacity_kind"] == "attach"
    assert observation.metadata["management_uid"] == attach_management_uid(
        project="project-a", zone="us-east5-a", name="existing-tpu", node_id="123456"
    )
    assert observation.metadata["provider_generation"] == 1
    # This catalog deliberately leaves v6e-16's per-host count unknown.
    assert CELL_ACCELERATOR_METADATA_KEY not in observation.metadata
    replayed = driver.apply_capacity("cell", desired, _context())
    newer = driver.reconcile("cell", _context("new", generation=2))
    assert (
        replayed.metadata["management_uid"] == newer.metadata["management_uid"] == observation.metadata["management_uid"]
    )
    assert replayed.metadata["provider_generation"] == newer.metadata["provider_generation"] == 1


@pytest.mark.parametrize("mode", ["public", "internal"])
def test_missing_host_address_never_publishes_a_partial_slice(tmp_path, mode):
    substrate = Substrate()
    driver = _driver(tmp_path, substrate)
    desired = _desired()
    desired["provider_options"]["endpoint_mode"] = mode
    driver.apply_capacity("cell", desired, _context())
    host = substrate.document["networkEndpoints"][1]
    host.pop("accessConfig" if mode == "public" else "ipAddress")
    observation = driver.observe("cell")
    assert observation.state is CapacityStatus.PROVISIONING
    assert observation.execution_endpoint is None
    assert not observation.metadata.get("native_worker_endpoints")


@pytest.mark.parametrize("spot", [True, False])
def test_missing_node_and_recreated_node_never_publish_endpoints(tmp_path, spot):
    substrate = Substrate()
    substrate.document = _document(spot=spot)
    desired = _desired()
    desired["capacity_class"] = "spot" if spot else "on-demand"
    driver = _driver(tmp_path, substrate)
    original = driver.apply_capacity("cell", desired, _context())
    substrate.document = None
    missing = driver.observe("cell")
    assert missing.state is CapacityStatus.NEEDS_RECREATE
    assert missing.failure_code is (
        CapacityProviderFailureCode.PREEMPTED if spot else CapacityProviderFailureCode.PROVIDER_ERROR
    )
    assert missing.execution_endpoint is None
    with pytest.raises(CapacityProviderError, match="no longer exists") as error:
        driver.apply_capacity("cell", desired, _context("retry", generation=2))
    assert error.value.retryable is False
    substrate.document = _document(spot=spot)
    substrate.document["id"] = "654321"
    leaked = driver.reconcile("cell", _context("reconcile", generation=2))
    assert leaked.state is CapacityStatus.NEEDS_RECREATE
    assert leaked.execution_endpoint is None
    assert not leaked.metadata.get("native_worker_endpoints")
    assert leaked.metadata["management_uid"] == original.metadata["management_uid"]
    with pytest.raises(CapacityProviderError, match="recreated under the same name"):
        driver.apply_capacity("cell", desired, _context("retry", generation=3))


@pytest.mark.parametrize("status", [400, 403, 429, 500, 503])
def test_classified_api_failures(tmp_path, status):
    substrate = Substrate()
    driver = _driver(tmp_path, substrate)
    driver.apply_capacity("cell", _desired(), _context())
    substrate.status = status
    with pytest.raises(CapacityProviderError) as error:
        driver.apply_capacity("other-cell", _desired(), _context())
    assert error.value.retryable is (status >= 429)
    assert error.value.code is (
        CapacityProviderFailureCode.PROVIDER_UNAVAILABLE if status >= 429 else CapacityProviderFailureCode.CONFIGURATION
    )
    observation = driver.observe("cell")
    assert observation.state in {CapacityStatus.UNAVAILABLE, CapacityStatus.UNOBSERVED}
    assert observation.failure_code is error.value.code
    assert observation.execution_endpoint is None
    assert not observation.metadata.get("native_worker_endpoints")
    assert driver.state.get("other-cell") is None


def test_malformed_node_is_conclusive_and_does_not_claim_capacity(tmp_path):
    substrate = Substrate()
    substrate.document.pop("id")
    driver = _driver(tmp_path, substrate)
    with pytest.raises(CapacityProviderError) as error:
        driver.apply_capacity("cell", _desired(), _context())
    assert error.value.code is CapacityProviderFailureCode.CONFIGURATION
    assert error.value.retryable is False
    assert driver.state.get("cell") is None


@pytest.mark.parametrize("options_key", ["provider_options", "provider_config"])
def test_nested_options_supported(tmp_path, options_key):
    substrate = Substrate()
    desired = _desired()
    desired[options_key] = desired.pop("provider_options")
    assert _driver(tmp_path, substrate).apply_capacity("cell", desired, _context()).state is CapacityStatus.HEALTHY


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", "manual"),
        ("capacity_backend", "gcp-tpu"),
        ("executor", "native-local"),
        ("project", ""),
        ("zone", ""),
        ("accelerator_type", ""),
        ("capacity_class", "nonsense"),
    ],
)
def test_invalid_desired_document_is_refused_before_http(tmp_path, field, value):
    substrate = Substrate()
    desired = _desired()
    desired[field] = value
    with pytest.raises((ValueError, CapacityProviderError)):
        _driver(tmp_path, substrate).apply_capacity("cell", desired, _context())
    assert substrate.requests == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("tpu_node", "../other"), ("tpu_node_id", ""), ("endpoint_mode", "other"), ("node_count", 2)],
)
def test_invalid_provider_options_are_refused_before_http(tmp_path, field, value):
    substrate = Substrate()
    desired = _desired()
    desired["provider_options"][field] = value
    with pytest.raises((ValueError, CapacityProviderError)):
        _driver(tmp_path, substrate).apply_capacity("cell", desired, _context())
    assert substrate.requests == []


@pytest.mark.parametrize("field,value", [("accelerator_type", "v6e-8"), ("capacity_class", "on_demand")])
def test_provider_truth_must_match_the_claim(tmp_path, field, value):
    substrate = Substrate()
    desired = _desired()
    desired[field] = value
    driver = _driver(tmp_path, substrate)
    with pytest.raises(CapacityProviderError) as error:
        driver.apply_capacity("cell", desired, _context())
    assert error.value.code is CapacityProviderFailureCode.CONFIGURATION
    assert not error.value.retryable
    assert driver.state.get("cell") is None


@pytest.mark.parametrize(
    "scheduling,expected", [({"spot": True}, "spot"), ({}, "on-demand"), ({"reserved": True}, "reserved")]
)
def test_observed_capacity_class_is_the_nodes_class(tmp_path, scheduling, expected):
    substrate = Substrate()
    substrate.document["schedulingConfig"] = scheduling
    desired = _desired()
    desired["capacity_class"] = expected
    driver = _driver(tmp_path, substrate)
    observation = driver.apply_capacity("cell", desired, _context())
    assert observation.metadata["capacity_class"] == expected
    capabilities = driver.capabilities()
    assert capabilities.provider == "gcp"
    assert capabilities.executor == "eyvan-native"
    assert capabilities.capacity_classes == ("spot", "on-demand", "reserved")
    assert capabilities.capacity_backends == ("gcp-attach",)
    assert capabilities.supports_callables and capabilities.supports_checkpoint
    assert capabilities.supports_exec is False


def test_drain_release_are_local_even_while_provider_is_unreachable(tmp_path):
    substrate = Substrate()
    driver = _driver(tmp_path, substrate)
    driver.apply_capacity("cell", _desired(), _context())
    document = copy.deepcopy(substrate.document)
    count = len(substrate.requests)
    substrate.unreachable = True
    assert driver.drain("cell", _context("drain")).state is CapacityStatus.DRAINING
    assert driver.observe("cell").state is CapacityStatus.DRAINING
    assert driver.release("cell", _context("release", generation=2)).state is CapacityStatus.RELEASED
    assert driver.observe("cell").state is CapacityStatus.RELEASED
    assert len(substrate.requests) == count
    assert substrate.document == document
    with pytest.raises(StaleMutationError):
        driver.apply_capacity("cell", _desired(), _context())


def test_router_routes_new_and_persisted_attach_cells_without_changing_existing_modes(tmp_path):
    substrate = Substrate()
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    router = GcpCapacityDriver(state, mode="auto", execution_backend="eyvan-native", attach_api=substrate.api)
    assert router._desired_kind(_desired()) == "attach"
    assert router._desired_kind({"capacity_backend": "gcp-tpu"}) == "tpu"
    assert router._desired_kind({"capacity_backend": "gcp-compute"}) == "compute"
    assert router._desired_kind({"capacity_backend": "gcp", "accelerator_type": "v6e-16"}) == "tpu"
    assert router._desired_kind({"capacity_backend": "gcp", "accelerator_type": "cpu"}) == "compute"
    assert "gcp-attach" in router.capabilities().capacity_backends
    observation = router.apply_capacity("cell", _desired(), _context())
    assert observation.state is CapacityStatus.HEALTHY
    assert router._record_kind("cell") == "attach"
    assert router.observe("cell").metadata["capacity_kind"] == "attach"
    assert router.reconcile("cell", _context("reconcile")).state is CapacityStatus.HEALTHY
    assert router.release("cell", _context("release", generation=2)).state is CapacityStatus.RELEASED


def test_registry_and_agent_family(tmp_path):
    settings = EyvanSettings(execution_backend="eyvan-native", native_worker_port=8123)
    driver = DEFAULT_CAPACITY_BACKENDS.create(
        "gcp-attach", state=FleetStateStore(LocalBackend(tmp_path / "fleet.json")), settings=settings
    )
    assert driver.mode == "attach"
    assert driver.attach.native_worker_port == 8123
    assert driver.capabilities().capacity_backends == ("gcp-attach",)
    assert capacity_backend_agent_family("gcp-attach") == "gcp"


def _fleet():
    return {
        "version": 1,
        "name": "attached-fleet",
        "provider": {"type": "gcp", "project": "project-a", "region": "us-east5", "zone": "us-east5-a"},
        "executor": "eyvan-native",
        "capacity_class": "spot",
        "pools": {
            "v6e-16": {
                "kind": "tpu",
                "accelerator_type": "v6e-16",
                "capacity_backend": "gcp-attach",
                "min_workers": 1,
                "max_workers": 1,
                "provider_options": _desired()["provider_options"],
            }
        },
    }


def test_contract_fleet_yaml_and_runtime(tmp_path):
    import yaml

    path = tmp_path / "attach.yaml"
    path.write_text(yaml.safe_dump(_fleet()))
    config = EyvanFleetConfig.from_yaml(path)
    assert config.pool_payloads()[0]["capacity_backend"] == "gcp-attach"
    assert requires_native_gcp_runtime(config)
    assert (
        CapacityPoolRecord.model_validate({**config.pool_payloads()[0], "agent_endpoint": "http://agent:8090"})
        .make_cell("attached-cell")
        .worker_count
        == 1
    )


def test_attach_and_provisioned_tpu_pools_can_share_one_agent():
    document = _fleet()
    document["agent_endpoint"] = "http://agent:8090"
    document["pools"]["provisioned"] = {
        "kind": "tpu",
        "accelerator_type": "v6e-16",
        "capacity_backend": "gcp-tpu",
        "provider_options": {"bootstrap_egress": "public-internet"},
    }
    assert len(EyvanFleetConfig.model_validate(document).pool_payloads()) == 2


@pytest.mark.parametrize("labels", [None, {"eyvan_resource_kind": "tpu"}])
async def test_callable_admission_targets_attached_slice_and_refuses_multislice(labels):
    config = EyvanFleetConfig.model_validate(_fleet())
    payload = config.pool_payloads()[0]
    payload["agent_endpoint"] = "http://agent:8090"
    if labels is None:
        payload["labels"] = {"eyvan_fleet": config.name}
    else:
        payload["labels"].update(labels)
    pool = CapacityPoolRecord.model_validate(payload)
    service = ControlPlaneService(
        InMemoryRepository(), callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"})
    )
    admin = Principal(subject="system", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    operator = Principal(
        subject="operator", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),)
    )
    await service.upsert_capacity_pool(admin, pool, idempotency_key="pool")
    arguments = {
        "fleet_name": config.name,
        "org_id": "acme",
        "project_id": "research",
        "accelerator_type": "v6e-16",
        "execution": CallableExecution(artifact_uri="gs://trusted/train.pkl", sha256="a" * 64, python="3.12"),
    }
    workload, replayed = await service.submit_pool_callable(operator, **arguments, pods=1, idempotency_key="one")
    assert not replayed
    assert workload.spec.required_labels["eyvan_pool"] == pool.id
    assert workload.spec.worker_count == 1
    with pytest.raises(ConflictError, match=r"pods=2.*gcp-tpu"):
        await service.submit_pool_callable(operator, **arguments, pods=2, idempotency_key="two")


def test_attached_observation_resolves_native_slice_with_gce_tpu_discovery(tmp_path):
    from eyvan.worker.runtime.native_slice import JaxClusterDetectionMode, NativeSliceExecutor
    from eyvan.worker.runtime.registry import ConfiguredExecutionResolver

    observation = _driver(tmp_path, Substrate()).apply_capacity("cell", _desired(), _context())
    resolver = ConfiguredExecutionResolver("eyvan-native", native_worker_token="test-root-secret")
    try:
        executor = resolver.resolve("cell", observation)
        assert isinstance(executor, NativeSliceExecutor)
        assert executor.jax_cluster_detection is JaxClusterDetectionMode.GCE_TPU
    finally:
        resolver.close()


def test_known_host_accelerator_descriptor_is_published(tmp_path):
    substrate = Substrate()
    substrate.document["acceleratorType"] = "v5p-16"
    desired = _desired()
    desired["accelerator_type"] = "v5p-16"
    observation = _driver(tmp_path, substrate).apply_capacity("cell", desired, _context())
    assert observation.metadata[CELL_ACCELERATOR_METADATA_KEY]["accelerator_type"] == "v5p-16"
    assert observation.metadata[CELL_ACCELERATOR_METADATA_KEY]["kind"] == "tpu"


def test_attach_driver_supports_exec_follows_settings_and_override(tmp_path, monkeypatch):
    monkeypatch.delenv("EYVAN_CAPACITY_SUPPORTS_EXEC", raising=False)
    assert _driver(tmp_path / "default", Substrate()).capabilities().supports_exec is False

    monkeypatch.setenv("EYVAN_CAPACITY_SUPPORTS_EXEC", "true")
    assert _driver(tmp_path / "env-on", Substrate()).capabilities().supports_exec is True

    monkeypatch.setenv("EYVAN_CAPACITY_SUPPORTS_EXEC", "false")
    state = FleetStateStore(LocalBackend(tmp_path / "override" / "fleet.json"))
    driver = GcpAttachCapacityDriver(state, api=Substrate().api, supports_exec=True)
    assert driver.capabilities().supports_exec is True


def test_apply_capacity_performs_provider_lookup_outside_mutation_lock(tmp_path):
    substrate = Substrate()
    lock_free_during_lookup: list[bool] = []
    driver = _driver(tmp_path, substrate)
    original_get = substrate.api.get_node

    def observed_get(project: str, zone: str, name: str):
        acquired = driver._mutation_lock.acquire(blocking=False)
        if acquired:
            driver._mutation_lock.release()
        lock_free_during_lookup.append(acquired)
        return original_get(project, zone, name)

    substrate.api.get_node = observed_get  # type: ignore[method-assign]
    observation = driver.apply_capacity("cell", _desired(), _context())
    assert observation.state is CapacityStatus.HEALTHY
    assert lock_free_during_lookup == [True]


def test_no_endpoints_is_provisioning(tmp_path):
    substrate = Substrate()
    substrate.document["networkEndpoints"] = []
    observation = _driver(tmp_path, substrate).apply_capacity("cell", _desired(), _context())
    assert observation.state is CapacityStatus.PROVISIONING
    assert observation.execution_endpoint is None
    assert observation.metadata["native_worker_endpoints"] == []


def test_ipv6_endpoint_is_a_valid_http_origin(tmp_path):
    substrate = Substrate()
    substrate.document["networkEndpoints"] = [{"accessConfig": {"externalIp": "2001:db8::1"}}]
    observation = _driver(tmp_path, substrate).apply_capacity("cell", _desired(), _context())
    assert observation.execution_endpoint == "http://[2001:db8::1]:8111"


def test_class_drift_reports_provider_truth_without_publishing_capacity(tmp_path):
    substrate = Substrate()
    driver = _driver(tmp_path, substrate)
    driver.apply_capacity("cell", _desired(), _context())
    substrate.document["schedulingConfig"] = {}
    observation = driver.observe("cell")
    assert observation.metadata["capacity_class"] == "on-demand"
    assert observation.state is CapacityStatus.NEEDS_RECREATE
    assert observation.execution_endpoint is None


def test_released_router_claim_retains_its_identity(tmp_path):
    substrate = Substrate()
    router = GcpCapacityDriver(
        FleetStateStore(LocalBackend(tmp_path / "fleet.json")), mode="auto", attach_api=substrate.api
    )
    original = router.apply_capacity("cell", _desired(), _context())
    released = router.release("cell", _context("release", generation=2))
    observed = router.observe("cell")
    assert observed.state is released.state is CapacityStatus.RELEASED
    assert released.metadata["management_uid"] == original.metadata["management_uid"]
    assert released.metadata["provider_generation"] == 1
    assert observed.metadata["capacity_kind"] == "attach"


async def _attached_control_plane(tmp_path, substrate):
    from test_controller import FakeAgent, principal

    from eyvan.server.controller import EyvanController
    from eyvan.server.service.scheduling import CoreSchedulingService
    from eyvan.worker.agent.client import CellAgentCapacityProviderError
    from eyvan.worker.agent.models import AgentObservation

    driver = _driver(tmp_path, substrate)

    class AttachedAgent(FakeAgent):
        def _capacity(self, cell_id, request, *, apply):
            context = MutationContext(**request.mutation.model_dump())
            try:
                observation = (
                    driver.apply_capacity(cell_id, request.desired, context)
                    if apply
                    else driver.reconcile(cell_id, context)
                )
            except CapacityProviderError as exc:
                raise CellAgentCapacityProviderError(exc.diagnostic, code=exc.code) from exc
            return AgentObservation(
                cell_id=cell_id,
                state=observation.state,
                generation=observation.generation,
                execution_endpoint=observation.execution_endpoint,
                metadata=observation.metadata,
                failure_code=observation.failure_code,
                observed_tpu_vms=observation.observed_tpu_vms,
                runtime_process_count=observation.runtime_process_count,
            )

        def apply_capacity(self, cell_id, request):
            return self._capacity(cell_id, request, apply=True)

        def reconcile(self, cell_id, request):
            return self._capacity(cell_id, request, apply=False)

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    agent = AttachedAgent()
    controller = EyvanController(repository, service, settings=EyvanSettings(), agent_factory=lambda _: agent)
    payload = EyvanFleetConfig.model_validate(_fleet()).pool_payloads()[0]
    pool = CapacityPoolRecord.model_validate({**payload, "agent_endpoint": "http://agent:8090"})
    cell = pool.make_cell("cell").model_copy(update={"pool_id": None})
    await service.upsert_cell(principal(Role.FLEET_ADMIN), cell, idempotency_key="cell")
    return repository, service, controller


@pytest.mark.parametrize("state", ["PREEMPTED", "DELETING", None])
async def test_attached_spot_reclamation_retries_using_preemption_budget(tmp_path, state):
    from test_controller import principal

    from eyvan.records import CommandExecution, WorkloadDefinition, WorkloadState

    substrate = Substrate()
    repository, service, controller = await _attached_control_plane(tmp_path, substrate)
    assert await controller.dispatch_outbox_once() == 1
    workload, _ = await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            capacity_class="spot",
            checkpointable=True,
            checkpoint_hook="python -m save",
            checkpoint_output_uri="gs://ckpt",
            max_retries=0,
            max_preemption_retries=1,
        ),
        idempotency_key="train",
    )
    await controller.reconcile_once()
    assert await controller.dispatch_outbox_once() == 1
    async with repository.transaction() as tx:
        stored = await tx.get_workload(workload.id)
        await tx.put_workload(
            stored.model_copy(update={"state": WorkloadState.RUNNING, "checkpoint_uri": "gs://ckpt/1"})
        )
    if state is None:
        substrate.document = None
    else:
        substrate.document["state"] = state
    await controller.observe_cells_once()
    await controller.observe_workloads_once()
    async with repository.transaction() as tx:
        recovered = await tx.get_workload(workload.id)
        lost_cell = await tx.get_cell("cell")
    assert lost_cell.provider_failure_code is CapacityProviderFailureCode.PREEMPTED
    assert recovered.state is WorkloadState.PENDING
    assert recovered.preemption_count == 1
    assert recovered.capacity_loss_count == 0
    assert recovered.failure_count == 0


async def test_refused_initial_attach_apply_is_not_reissued_forever(tmp_path):
    from eyvan.records import CellState, OutboxState, utcnow

    substrate = Substrate()
    substrate.document = None
    repository, _service, controller = await _attached_control_plane(tmp_path, substrate)
    assert await controller.dispatch_outbox_once() == 1
    async with repository.transaction() as tx:
        cell = await tx.get_cell("cell")
        command = (await tx.list_outbox())[0]
        assert cell.state is CellState.UNAVAILABLE
        assert cell.provider_failure_code is CapacityProviderFailureCode.CONFIGURATION
        assert command.state is OutboxState.DISCARDED
        assert "no longer exists" in command.last_error
        await tx.put_outbox(command.model_copy(update={"available_at": utcnow()}))
    assert await controller.dispatch_outbox_once() == 0
    async with repository.transaction() as tx:
        settled = await tx.get_outbox(command.id)
    assert settled.state is OutboxState.DISCARDED
    assert len(substrate.requests) == 1


async def test_initial_apply_preserves_preemption_reason_and_does_not_score_stockout():
    from test_controller import FakeAgent, principal

    from eyvan.records import CellRecord, CellState, ResourceVector
    from eyvan.server.controller import EyvanController
    from eyvan.worker.agent.models import AgentObservation

    class PreemptedAgent(FakeAgent):
        def apply_capacity(self, cell_id, request):
            return AgentObservation(
                cell_id=cell_id,
                state=CapacityStatus.NEEDS_RECREATE,
                generation=request.mutation.generation,
                failure_code=CapacityProviderFailureCode.PREEMPTED,
            )

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service, settings=EyvanSettings(), agent_factory=lambda _: PreemptedAgent())
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell",
            provider="local",
            region="local",
            state=CellState.REGISTERING,
            capacity=ResourceVector(cpu=8, memory_gib=64),
            allocatable=ResourceVector(cpu=8, memory_gib=64),
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell",
    )
    assert await controller.dispatch_outbox_once() == 1
    async with repository.transaction() as tx:
        cell = await tx.get_cell("cell")
    assert cell.provider_failure_code is CapacityProviderFailureCode.PREEMPTED
    assert cell.availability.consecutive_failures == 0
    assert cell.availability.interruptions == 1
    assert cell.availability.cooldown_until is None


class FakeSshReader:
    """A node read through one of its own hosts; records which entry was used."""

    def __init__(self, entries):
        self.entries = entries
        self.document = _document(spot=False)

    def __call__(self, entry):
        self.entries.append(entry.target)
        return self

    def get_node(self, project, zone, name):
        from eyvan.capacity.providers.gcp.tpu_nodes import parse_tpu_node

        return None if self.document is None else parse_tpu_node(self.document, project=project)


class FakeTunnels:
    def __init__(self):
        self.ensured = []
        self.stopped = []
        self.fail = None

    def ensure(self, spec):
        if self.fail is not None:
            raise self.fail
        self.ensured.append(spec)
        return True

    def stop(self, key):
        self.stopped.append(key)
        return True


def _tunnel_desired(**overrides):
    desired = _desired()
    desired["capacity_class"] = "on-demand"
    desired["provider_options"] = {
        "tpu_node": "existing-tpu",
        "tpu_node_id": "123456",
        "endpoint_mode": "tunnel",
        "ssh_entry": "me@192.0.2.2",
        "tunnel_ports": [4100, 4101],
        "worker_port": 8100,
        **overrides,
    }
    return desired


def _tunnel_driver(tmp_path):
    entries: list[str] = []
    reader = FakeSshReader(entries)
    tunnels = FakeTunnels()
    substrate = Substrate()
    driver = GcpAttachCapacityDriver(
        FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        api=substrate.api,
        native_worker_port=8111,
        ssh_reader_factory=reader,
        tunnels=tunnels,
    )
    return driver, reader, tunnels, substrate, entries


def test_tunnel_mode_reads_over_ssh_and_publishes_local_forwards(tmp_path):
    driver, _, tunnels, substrate, entries = _tunnel_driver(tmp_path)
    observation = driver.apply_capacity("cell", _tunnel_desired(), _context())
    assert observation.state is CapacityStatus.HEALTHY
    assert observation.metadata["native_worker_endpoints"] == ["http://127.0.0.1:4100", "http://127.0.0.1:4101"]
    assert observation.execution_endpoint == "http://127.0.0.1:4100"
    assert observation.metadata["capacity_class"] == "on-demand"
    assert not substrate.requests, "a node read over SSH must not touch the operator's TPU API"
    assert set(entries) == {"me@192.0.2.2"}
    spec = tunnels.ensured[-1]
    uid = attach_management_uid(project="project-a", zone="us-east5-a", name="existing-tpu", node_id="123456")
    assert spec.key == f"attach-{uid}"
    # Forwards go to internal addresses on the recorded worker port, not the agent's default.
    assert [(f.local_port, f.remote_host, f.remote_port) for f in spec.forwards] == [
        (4100, "10.0.0.2", 8100),
        (4101, "10.0.0.3", 8100),
    ]
    assert driver.observe("cell").state is CapacityStatus.HEALTHY


def test_dead_tunnel_is_unobserved_not_healthy(tmp_path):
    from eyvan._internal.ssh import SshError

    driver, _, tunnels, _, _ = _tunnel_driver(tmp_path)
    driver.apply_capacity("cell", _tunnel_desired(), _context())
    tunnels.fail = SshError("ssh: connect to host 192.0.2.2 port 22: timed out")
    observation = driver.observe("cell")
    assert observation.state is CapacityStatus.UNOBSERVED
    assert observation.failure_code is CapacityProviderFailureCode.PROVIDER_UNAVAILABLE
    assert observation.metadata["native_worker_endpoints"] == []
    assert "tunnel" in observation.detail


def test_tunnel_port_count_mismatch_publishes_nothing(tmp_path):
    driver, reader, _, _, _ = _tunnel_driver(tmp_path)
    driver.apply_capacity("cell", _tunnel_desired(), _context())
    reader.document["networkEndpoints"].append({"ipAddress": "10.0.0.4"})
    observation = driver.observe("cell")
    assert observation.state is CapacityStatus.UNAVAILABLE
    assert observation.metadata["native_worker_endpoints"] == []


def test_release_stops_the_tunnel_and_never_reads_the_node(tmp_path):
    driver, _, tunnels, _, _ = _tunnel_driver(tmp_path)
    driver.apply_capacity("cell", _tunnel_desired(), _context())
    uid = attach_management_uid(project="project-a", zone="us-east5-a", name="existing-tpu", node_id="123456")
    assert driver.release("cell", _context("release", generation=2)).state is CapacityStatus.RELEASED
    assert tunnels.stopped == [f"attach-{uid}"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"ssh_entry": None},
        {"tunnel_ports": []},
        {"tunnel_ports": [4100, 4100]},
        {"tunnel_ports": [0]},
        {"ssh_entry": "-oProxyCommand=x"},
        {"worker_port": 70000},
    ],
)
def test_invalid_tunnel_options_are_refused_before_any_read(tmp_path, overrides):
    driver, _, tunnels, substrate, entries = _tunnel_driver(tmp_path)
    desired = _tunnel_desired(**overrides)
    desired["provider_options"] = {k: v for k, v in desired["provider_options"].items() if v is not None}
    with pytest.raises(ValueError):
        driver.apply_capacity("cell", desired, _context())
    assert not entries and not substrate.requests and not tunnels.ensured


def test_tunnel_ports_outside_tunnel_mode_are_refused(tmp_path):
    driver, _, _, _, _ = _tunnel_driver(tmp_path)
    with pytest.raises(ValueError, match="tunnel_ports"):
        driver.apply_capacity("cell", _tunnel_desired(endpoint_mode="public"), _context())
