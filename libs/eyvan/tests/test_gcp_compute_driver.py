# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import typing as tp

import pytest

from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan.capacity.drivers.base import CapacityStatus, MutationContext, StaleMutationError
from eyvan.capacity.drivers.gcp import GcpCapacityDriver, GcpComputeCapacityDriver, GcpTpuCapacityDriver
from eyvan.capacity.fleet_state import FleetCellRecord, FleetStateStore, LocalBackend
from eyvan.capacity.providers.gcp import GcsArtifactBucket, ProviderObservation, compute_instance_name
from eyvan.capacity.providers.gcp.queued_resources import GcloudError
from eyvan.config import EyvanSettings
from eyvan.worker.agent.service import CellAgentService
from eyvan.worker.runtime.native_bootstrap import derive_native_worker_token

_SHA256 = "a" * 64


def metadata_object(record: FleetCellRecord, name: str) -> JsonObject:
    value = record.metadata[name]
    assert is_json_object(value)
    return value


def context(key: str, generation: int = 1, token: str = "leader-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


class FakeProvider:
    def __init__(self, reports: list[ProviderObservation] | None = None, *, events: list[str] | None = None):
        self.reports = reports or [ProviderObservation(CapacityStatus.WAITING, "capacity requested")]
        self.ensured: list[FleetCellRecord] = []
        self.deleted: list[tuple[str, str, str, str | None]] = []
        self.events = events

    def ensure(self, record: FleetCellRecord) -> ProviderObservation:
        if self.events is not None:
            self.events.append("provider")
        self.ensured.append(record)
        return self.reports.pop(0) if len(self.reports) > 1 else self.reports[0]

    def delete(
        self,
        resource_id: str,
        *,
        project: str,
        zone: str,
        management_uid: str | None = None,
    ) -> None:
        self.deleted.append((resource_id, project, zone, management_uid))


class FakeNativeWorkerSecrets:
    def __init__(self, *, on_ensure: tp.Callable[[], None] | None = None) -> None:
        self.ensured: list[dict] = []
        self.deleted: list[dict] = []
        self.on_ensure = on_ensure

    def ensure(self, **kwargs) -> str:
        self.ensured.append(kwargs)
        callback = self.on_ensure
        self.on_ensure = None
        if callback is not None:
            callback()
        return f"projects/{kwargs['project']}/secrets/eyvan-native-{kwargs['management_uid']}/versions/latest"

    def delete(self, **kwargs) -> None:
        self.deleted.append(kwargs)


class FakeNativeWorkerServiceAccountResolver:
    def __init__(self, service_account: str = "123456789012-compute@developer.gserviceaccount.com") -> None:
        self.service_account = service_account
        self.projects: list[str] = []

    def resolve(self, *, project: str) -> str:
        self.projects.append(project)
        return self.service_account


class FakeNativeWorkerArtifactAccess:
    def __init__(self, *, events: list[str] | None = None, error: GcloudError | None = None) -> None:
        self.events = events
        self.error = error
        self.ensured: list[tuple[str, GcsArtifactBucket, str]] = []

    def ensure(
        self,
        *,
        project: str,
        artifact_bucket: GcsArtifactBucket,
        reader_service_account: str,
    ) -> None:
        if self.events is not None:
            self.events.append("artifact-access")
        if self.error is not None:
            raise self.error
        self.ensured.append((project, artifact_bucket, reader_service_account))


class FakeNativeWorkerPublicIngress:
    def __init__(self, *, events: list[str] | None = None) -> None:
        self.events = events
        self.calls: list[tuple[str, str | None, int]] = []

    def ensure(self, *, project: str, network: str | None, port: int) -> str:
        if self.events is not None:
            self.events.append("ingress")
        self.calls.append((project, network, port))
        return "eyvan-nw-profile-managed"


def compute_desired(*, gpu: bool = True) -> dict:
    options: dict = {
        "machine_type": "g2-standard-8" if gpu else "c4-standard-8",
        "service_account": "eyvan-worker@project-a.iam.gserviceaccount.com",
        "bootstrap_egress": "private-google-access",
    }
    if gpu:
        options.update({"gpu_type": "nvidia-l4", "gpu_count": 1})
    return {
        "capacity_backend": "gcp-compute",
        "project": "project-a",
        "zone": "us-central1-a",
        "accelerator_type": "nvidia-l4" if gpu else "c4-standard-8",
        "capacity_class": "spot" if gpu else "on-demand",
        "executor": "native-local",
        "provider_options": options,
    }


def tpu_desired() -> dict:
    return {
        "capacity_backend": "gcp-tpu",
        "project": "project-a",
        "zone": "us-central2-b",
        "accelerator_type": "v6e-8",
        "capacity_class": "spot",
        "executor": "native-local",
    }


def compute_tpu_desired() -> dict:
    return {
        "capacity_backend": "gcp-compute",
        "project": "project-a",
        "zone": "us-east5-a",
        "accelerator_type": "v5p-8",
        "capacity_class": "spot",
        "executor": "ray-jobs",
        "provider_options": {
            "machine_type": "ct5p-hightpu-4t",
            "tpu_type": "ct5p",
            "tpu_count": 4,
            "image_family": "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e",
            "image_project": "ubuntu-os-accelerator-images",
            "service_account": "eyvan-worker@project-a.iam.gserviceaccount.com",
            "bootstrap_egress": "private-google-access",
        },
    }


def remove_cell_record(state: FleetStateStore, cell_id: str) -> None:
    """Simulate an external state writer removing one registered cell."""

    def erase(document: JsonObject) -> None:
        cells = document["cells"]
        assert is_json_object(cells)
        cells.pop(cell_id)

    state.backend.update(erase)


def test_compute_apply_persists_owned_provider_config_and_is_idempotent(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = FakeProvider()
    driver = GcpComputeCapacityDriver(state, provider=provider, execution_backend="native-local")

    first = driver.apply_capacity("GPU_Cell", compute_desired(), context("create"))
    second = driver.apply_capacity("GPU_Cell", compute_desired(), context("create"))

    record = state.get("GPU_Cell")
    assert first.state is CapacityStatus.WAITING
    assert second.state is CapacityStatus.WAITING
    assert len(provider.ensured) == 2
    assert record is not None
    assert record.qr_id == compute_instance_name("GPU_Cell", 1)
    assert record.metadata["capacity_kind"] == "compute"
    assert record.metadata["capacity_backend"] == "gcp-compute"
    config = metadata_object(record, "compute_config")
    assert config["machine_type"] == "g2-standard-8"
    assert config["gpu_type"] == "nvidia-l4"
    assert isinstance(record.metadata["management_uid"], str)
    assert len(record.metadata["management_uid"]) == 32


def test_compute_apply_preserves_public_tpu_shape_and_integrated_provider_config(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = FakeProvider()
    driver = GcpComputeCapacityDriver(state, provider=provider, execution_backend="ray-jobs")

    observation = driver.apply_capacity("v5p-cell", compute_tpu_desired(), context("create"))

    record = state.get("v5p-cell")
    assert observation.state is CapacityStatus.WAITING
    assert record is not None
    assert record.accelerator_type == "v5p-8"
    assert record.metadata["capacity_kind"] == "compute"
    config = metadata_object(record, "compute_config")
    assert config["machine_type"] == "ct5p-hightpu-4t"
    assert config["tpu_type"] == "ct5p"
    assert config["tpu_count"] == 4


def test_compute_native_worker_bootstrap_is_cell_scoped_and_uses_the_selected_wheel(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    secrets = FakeNativeWorkerSecrets()
    accounts = FakeNativeWorkerServiceAccountResolver()
    driver = GcpComputeCapacityDriver(
        state,
        provider=FakeProvider(),
        execution_backend="eyvan-native",
        native_worker_port=8123,
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_wheelhouse="gs://artifacts/eyvan/wheelhouse.zip",
        native_worker_wheelhouse_sha256=_SHA256,
        native_worker_state_dir="/var/lib/eyvan/worker",
        native_worker_shared_secret="agent-global-secret",
        native_worker_secrets=secrets,
        native_worker_service_account_resolver=accounts,
    )

    driver.apply_capacity("v5p-cell", {**compute_tpu_desired(), "executor": "eyvan-native"}, context("create"))

    record = state.get("v5p-cell")
    assert record is not None
    script = metadata_object(record, "compute_config")["startup_script"]
    assert isinstance(script, str)
    token = derive_native_worker_token(
        "agent-global-secret",
        record.metadata["management_uid"],
        provider_generation=record.provider_generation,
    )
    assert record.metadata["execution_backend"] == "eyvan-native"
    assert record.metadata["native_worker_port"] == 8123
    assert "agent-global-secret" not in script
    assert token not in script
    assert record.metadata["native_worker_secret_id"] in script
    assert "gcloud storage cp gs://artifacts/eyvan/eyvan.whl /var/lib/eyvan/bootstrap/eyvan.whl" in script
    assert "/var/lib/eyvan/bootstrap/eyvan.whl" in script
    assert "eyvan.worker.runtime.native_worker --host 0.0.0.0 --port 8123" in script
    assert secrets.ensured == [
        {
            "project": "project-a",
            "management_uid": record.metadata["management_uid"],
            "value": token,
            "reader_service_account": "eyvan-worker@project-a.iam.gserviceaccount.com",
        }
    ]
    assert accounts.projects == []


def test_compute_bootstrap_fences_a_missing_record_before_provider_capacity(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))

    def remove_cell() -> None:
        def erase(document: JsonObject) -> None:
            cells = document["cells"]
            assert is_json_object(cells)
            cells.pop("compute-cell")

        state.backend.update(erase)

    provider = FakeProvider()
    secrets = FakeNativeWorkerSecrets(on_ensure=remove_cell)
    driver = GcpComputeCapacityDriver(
        state,
        provider=provider,
        execution_backend="eyvan-native",
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_shared_secret="agent-global-secret",
        native_worker_secrets=secrets,
        native_worker_service_account_resolver=FakeNativeWorkerServiceAccountResolver(),
    )
    request = compute_desired()
    request["executor"] = "eyvan-native"
    request["provider_options"] = {
        **request["provider_options"],
        "bootstrap_egress": "public-internet",
        "internal_ip_only": False,
    }

    with pytest.raises(StaleMutationError, match="removed before native worker bootstrap could be persisted"):
        driver.apply_capacity("compute-cell", request, context("create"))

    assert provider.ensured == []
    assert state.get("compute-cell") is None
    assert state.cell_tombstone("compute-cell") == {"generation": 1, "fencing_token": "leader-1"}
    assert secrets.deleted == [{"project": "project-a", "management_uid": secrets.ensured[0]["management_uid"]}]

    replacement = driver.apply_capacity("compute-cell", request, context("replacement", generation=2, token="leader-2"))

    assert replacement.state is CapacityStatus.WAITING
    assert len(provider.ensured) == 1


def test_compute_recreation_fences_record_loss_and_cleans_up_owned_resources(tmp_path) -> None:
    class RemoveCellOnSecondSecret(FakeNativeWorkerSecrets):
        def __init__(self) -> None:
            super().__init__()
            self.ensure_count = 0

        def ensure(self, **kwargs) -> str:
            secret = super().ensure(**kwargs)
            self.ensure_count += 1
            if self.ensure_count == 2:
                remove_cell_record(state, "compute-cell")
            return secret

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = FakeProvider(
        [
            ProviderObservation(CapacityStatus.WAITING),
            ProviderObservation(CapacityStatus.NEEDS_RECREATE, "instance failed"),
            ProviderObservation(CapacityStatus.NEEDS_RECREATE, "instance remains failed"),
        ]
    )
    secrets = RemoveCellOnSecondSecret()
    driver = GcpComputeCapacityDriver(
        state,
        provider=provider,
        execution_backend="eyvan-native",
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_wheelhouse="gs://artifacts/eyvan/wheelhouse.zip",
        native_worker_wheelhouse_sha256=_SHA256,
        native_worker_shared_secret="agent-global-secret",
        native_worker_secrets=secrets,
        native_worker_service_account_resolver=FakeNativeWorkerServiceAccountResolver(),
    )
    request = compute_desired()
    request["executor"] = "eyvan-native"
    driver.apply_capacity("compute-cell", request, context("create"))
    driver.reconcile("compute-cell", context("reconcile"))

    with pytest.raises(StaleMutationError, match="native worker recreation could be persisted"):
        driver.reconcile("compute-cell", context("reconcile"))

    assert len(provider.ensured) == 3
    assert [item[0] for item in provider.deleted] == [
        compute_instance_name("compute-cell", 1),
        compute_instance_name("compute-cell", 2),
    ]
    assert state.cell_tombstone("compute-cell") == {"generation": 1, "fencing_token": "leader-1"}
    assert secrets.deleted == [{"project": "project-a", "management_uid": secrets.ensured[0]["management_uid"]}]
    with pytest.raises(StaleMutationError, match="not newer"):
        driver.apply_capacity("compute-cell", request, context("replay"))


def test_compute_native_worker_uses_default_service_account_when_omitted(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    secrets = FakeNativeWorkerSecrets()
    accounts = FakeNativeWorkerServiceAccountResolver()
    driver = GcpComputeCapacityDriver(
        state,
        provider=FakeProvider(),
        execution_backend="eyvan-native",
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_wheelhouse="gs://artifacts/eyvan/wheelhouse.zip",
        native_worker_wheelhouse_sha256=_SHA256,
        native_worker_shared_secret="agent-global-secret",
        native_worker_secrets=secrets,
        native_worker_service_account_resolver=accounts,
    )
    requested = compute_tpu_desired()
    requested["executor"] = "eyvan-native"
    requested["provider_options"] = {
        key: value for key, value in requested["provider_options"].items() if key != "service_account"
    }

    driver.apply_capacity("v5p-cell", requested, context("create"))

    record = state.get("v5p-cell")
    assert record is not None
    assert metadata_object(record, "compute_config")["service_account"] is None
    assert accounts.projects == ["project-a"]
    assert secrets.ensured[0]["reader_service_account"] == accounts.service_account


def test_profile_managed_public_compute_ingress_precedes_capacity_and_adds_its_tag(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    events: list[str] = []
    ingress = FakeNativeWorkerPublicIngress(events=events)
    provider = FakeProvider(events=events)
    driver = GcpComputeCapacityDriver(
        state,
        provider=provider,
        execution_backend="eyvan-native",
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_shared_secret="agent-global-secret",
        native_worker_secrets=FakeNativeWorkerSecrets(),
        native_worker_service_account_resolver=FakeNativeWorkerServiceAccountResolver(),
        native_worker_public_ingress=ingress,
    )
    requested = compute_desired()
    requested["executor"] = "eyvan-native"
    requested["provider_options"] = {
        **requested["provider_options"],
        "bootstrap_egress": "public-internet",
        "internal_ip_only": False,
    }

    driver.apply_capacity("public-compute", requested, context("create"))

    record = state.get("public-compute")
    assert record is not None
    assert events == ["ingress", "provider"]
    assert ingress.calls == [("project-a", None, 8100)]
    assert metadata_object(record, "compute_config")["tags"] == ["eyvan-nw-profile-managed"]
    assert record.metadata["native_worker_endpoint_mode"] == "profile-managed-public"


def test_native_tpu_worker_configures_profile_artifact_access_before_capacity_provisioning(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    events: list[str] = []
    provider = FakeProvider(events=events)
    secrets = FakeNativeWorkerSecrets()
    accounts = FakeNativeWorkerServiceAccountResolver()
    access = FakeNativeWorkerArtifactAccess(events=events)
    driver = GcpTpuCapacityDriver(
        state,
        provider=provider,
        execution_backend="eyvan-native",
        native_worker_package="gs://eyvan-artifacts/eyvan/eyvan-0.1.0-py3-none-any.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_shared_secret="agent-global-secret",
        native_worker_secrets=secrets,
        native_worker_service_account_resolver=accounts,
        native_worker_artifact_root_uri="gs://eyvan-artifacts/eyvan",
        native_worker_artifact_access=access,
    )
    desired = {
        **tpu_desired(),
        "executor": "eyvan-native",
        "provider_options": {"bootstrap_egress": "public-internet"},
    }

    driver.apply_capacity("v6e-cell", desired, context("create"))

    assert events == ["artifact-access", "provider"]
    assert access.ensured == [
        (
            "project-a",
            GcsArtifactBucket.from_artifact_root_uri("gs://eyvan-artifacts/eyvan"),
            accounts.service_account,
        )
    ]
    assert accounts.projects == ["project-a"]
    assert len(secrets.ensured) == 1
    assert len(provider.ensured) == 1


def test_native_tpu_worker_does_not_request_capacity_when_artifact_access_cannot_be_configured(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = FakeProvider()
    secrets = FakeNativeWorkerSecrets()
    driver = GcpTpuCapacityDriver(
        state,
        provider=provider,
        execution_backend="eyvan-native",
        native_worker_package="gs://eyvan-artifacts/eyvan/eyvan-0.1.0-py3-none-any.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_shared_secret="agent-global-secret",
        native_worker_secrets=secrets,
        native_worker_service_account_resolver=FakeNativeWorkerServiceAccountResolver(),
        native_worker_artifact_root_uri="gs://eyvan-artifacts/eyvan",
        native_worker_artifact_access=FakeNativeWorkerArtifactAccess(
            error=GcloudError("could not configure native worker artifact access")
        ),
    )
    desired = {
        **tpu_desired(),
        "executor": "eyvan-native",
        "provider_options": {"bootstrap_egress": "public-internet"},
    }

    with pytest.raises(GcloudError, match="could not configure native worker artifact access"):
        driver.apply_capacity("v6e-cell", desired, context("create"))

    assert provider.ensured == []
    assert secrets.ensured == []


def test_compute_runtime_vm_image_and_setup_are_compiled_without_provider_option_leakage(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    driver = GcpComputeCapacityDriver(
        state,
        provider=FakeProvider(),
        execution_backend="eyvan-native",
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_wheelhouse="gs://artifacts/eyvan/wheelhouse.zip",
        native_worker_wheelhouse_sha256=_SHA256,
        native_worker_shared_secret="agent-global-secret",
        native_worker_secrets=FakeNativeWorkerSecrets(),
    )
    desired = {
        **compute_desired(gpu=False),
        "executor": "eyvan-native",
        "runtime": {
            "vm_image": "projects/project-a/global/images/eyvan-base-v1",
            "env": {"WANDB_MODE": "offline"},
            "init_commands": ["mkdir -p /opt/eyvan-runtime"],
        },
    }

    driver.apply_capacity("runtime-cell", desired, context("create"))

    record = state.get("runtime-cell")
    assert record is not None
    config = metadata_object(record, "compute_config")
    native_runtime = metadata_object(record, "native_runtime")
    assert config["image"] == "projects/project-a/global/images/eyvan-base-v1"
    assert config["image_family"] is None
    assert config["image_project"] is None
    assert native_runtime["env"] == {"WANDB_MODE": "offline"}
    startup_script = config["startup_script"]
    assert isinstance(startup_script, str)
    assert "mkdir -p /opt/eyvan-runtime" in startup_script
    assert "--runtime-env-file /etc/eyvan-native/workload-runtime.json" in startup_script


def test_private_google_access_requires_an_offline_native_worker_wheelhouse(tmp_path) -> None:
    driver = GcpComputeCapacityDriver(
        FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        provider=FakeProvider(),
        execution_backend="eyvan-native",
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_shared_secret="native-worker-root-secret",
        native_worker_secrets=FakeNativeWorkerSecrets(),
    )

    with pytest.raises(ValueError, match="EYVAN_NATIVE_WORKER_WHEELHOUSE"):
        driver.apply_capacity("private-cell", {**compute_tpu_desired(), "executor": "eyvan-native"}, context("create"))


def test_generic_gcp_driver_routes_compute_and_tpu_cells_through_one_agent(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    tpu = FakeProvider()
    compute = FakeProvider()
    driver = GcpCapacityDriver(
        state,
        tpu_provider=tpu,
        compute_provider=compute,
        execution_backend="native-local",
    )

    driver.apply_capacity("cpu-cell", compute_desired(gpu=False), context("cpu"))
    driver.apply_capacity("tpu-cell", tpu_desired(), context("tpu"))
    driver.reconcile("cpu-cell", context("cpu-reconcile"))
    driver.reconcile("tpu-cell", context("tpu-reconcile"))

    assert [record.name for record in compute.ensured] == ["cpu-cell", "cpu-cell"]
    assert [record.name for record in tpu.ensured] == ["tpu-cell", "tpu-cell"]
    assert driver.capabilities().accelerator_types == ("cpu", "gpu", "tpu")


def test_generic_gcp_driver_can_infer_compute_from_machine_type(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    compute = FakeProvider()
    driver = GcpCapacityDriver(
        state,
        tpu_provider=FakeProvider(),
        compute_provider=compute,
        execution_backend="native-local",
    )
    desired = {**compute_desired(gpu=False), "capacity_backend": "gcp"}

    driver.apply_capacity("cpu-cell", desired, context("cpu"))

    assert len(compute.ensured) == 1
    assert state.get("cpu-cell").metadata["capacity_kind"] == "compute"


def test_cell_agent_wires_generic_and_compatibility_gcp_backends(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))

    settings = EyvanSettings(
        agent_shared_secret="agent-global-secret",
        native_worker_root_secret="native-worker-root-secret",
    )
    generic = CellAgentService(state=state, settings=settings)
    compute = CellAgentService(
        state=state,
        settings=EyvanSettings(
            capacity_backend="gcp-compute",
            agent_shared_secret="agent-global-secret",
            native_worker_root_secret="native-worker-root-secret",
        ),
    )
    tpu = CellAgentService(
        state=state,
        settings=EyvanSettings(
            capacity_backend="gcp-tpu",
            agent_shared_secret="agent-global-secret",
            native_worker_root_secret="native-worker-root-secret",
        ),
    )

    assert generic.capabilities().provider == "gcp"
    assert compute.capabilities().provider == "gcp-compute"
    assert compute.capabilities().accelerator_types == ("cpu", "gpu", "tpu")
    assert tpu.capabilities().provider == "gcp-tpu"


def test_compute_reconcile_durably_recreates_terminal_instance(tmp_path) -> None:
    provider = FakeProvider(
        [
            ProviderObservation(CapacityStatus.WAITING),
            ProviderObservation(CapacityStatus.NEEDS_RECREATE, "instance terminated"),
            ProviderObservation(CapacityStatus.NEEDS_RECREATE, "instance still terminated"),
            ProviderObservation(CapacityStatus.WAITING, "replacement requested"),
        ]
    )
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    driver = GcpComputeCapacityDriver(state, provider=provider, execution_backend="native-local")
    driver.apply_capacity("cell", compute_desired(), context("create"))
    first_id = state.get("cell").qr_id

    failed = driver.reconcile("cell", context("reconcile"))
    replacement = driver.reconcile("cell", context("reconcile"))

    record = state.get("cell")
    assert failed.state is CapacityStatus.NEEDS_RECREATE
    assert replacement.state is CapacityStatus.WAITING
    assert record.qr_id == compute_instance_name("cell", 2)
    assert provider.deleted[0][:3] == (first_id, "project-a", "us-central1-a")
    assert provider.deleted[0][3] == record.metadata["management_uid"]
    assert "recreate_intent" not in record.metadata


def test_compute_recreation_clears_a_stale_native_worker_endpoint(tmp_path) -> None:
    provider = FakeProvider(
        [
            ProviderObservation(
                CapacityStatus.HEALTHY,
                head_ip="10.20.0.7",
                execution_endpoint="http://10.20.0.7:8100",
            ),
            ProviderObservation(CapacityStatus.NEEDS_RECREATE, "instance terminated"),
            ProviderObservation(CapacityStatus.NEEDS_RECREATE, "instance remains terminated"),
            ProviderObservation(CapacityStatus.WAITING, "replacement requested"),
        ]
    )
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    secrets = FakeNativeWorkerSecrets()
    driver = GcpComputeCapacityDriver(
        state,
        provider=provider,
        execution_backend="eyvan-native",
        native_worker_shared_secret="agent-global-secret",
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_wheelhouse="gs://artifacts/eyvan/wheelhouse.zip",
        native_worker_wheelhouse_sha256=_SHA256,
        native_worker_secrets=secrets,
    )
    driver.apply_capacity("cell", {**compute_desired(), "executor": "eyvan-native"}, context("create"))

    assert state.get("cell").execution_endpoint == "http://10.20.0.7:8100"
    driver.reconcile("cell", context("reconcile"))
    assert state.get("cell").execution_endpoint is None
    driver.reconcile("cell", context("reconcile"))

    replacement = state.get("cell")
    assert replacement is not None
    assert replacement.execution_endpoint is None
    assert replacement.head_ip is None
    assert len(secrets.ensured) == 2
    assert secrets.ensured[0]["value"] != secrets.ensured[1]["value"]


def test_compute_release_deletes_owned_instance_and_is_replay_safe(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = FakeProvider()
    driver = GcpComputeCapacityDriver(state, provider=provider, execution_backend="native-local")
    driver.apply_capacity("cell", compute_desired(), context("create"))
    record = state.get("cell")
    release = context("release", generation=2, token="leader-2")

    first = driver.release("cell", release)
    second = driver.release("cell", release)

    assert first.state is CapacityStatus.RELEASED
    assert second.state is CapacityStatus.RELEASED
    assert provider.deleted == [(record.qr_id, "project-a", "us-central1-a", record.metadata["management_uid"])]
    assert state.get("cell") is None


def test_compute_release_with_a_consumed_key_still_deletes_live_capacity(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = FakeProvider()
    driver = GcpComputeCapacityDriver(state, provider=provider, execution_backend="native-local")
    driver.apply_capacity("cell", compute_desired(), context("create"))
    record = state.get("cell")
    # A colliding idempotency key is consumed by another mutation while the
    # record still shows live capacity.
    driver.reconcile("cell", context("release", generation=2, token="leader-2"))

    observation = driver.release("cell", context("release", generation=2, token="leader-2"))

    assert observation.state is CapacityStatus.RELEASED
    assert provider.deleted == [(record.qr_id, "project-a", "us-central1-a", record.metadata["management_uid"])]
    assert state.get("cell") is None


def test_compute_observation_metadata_never_shadows_record_fields(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    driver = GcpComputeCapacityDriver(state, provider=FakeProvider(), execution_backend="native-local")
    driver.apply_capacity("cell", compute_desired(), context("create"))
    record = state.get("cell")

    def poison(current: FleetCellRecord) -> None:
        current.metadata["project"] = "poisoned-project"
        current.metadata["zone"] = "poisoned-zone"
        current.metadata["resource_id"] = "poisoned-resource"
        current.metadata["head_ip"] = "203.0.113.99"

    state.mutate_record("cell", poison)

    metadata = driver.observe("cell").metadata
    assert metadata["project"] == "project-a"
    assert metadata["zone"] == "us-central1-a"
    assert metadata["resource_id"] == record.qr_id
    assert metadata["head_ip"] is None


def test_compute_driver_rejects_invalid_shape_and_forced_backend_mismatch(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    driver = GcpComputeCapacityDriver(state, provider=FakeProvider(), execution_backend="native-local")

    with pytest.raises(ValueError, match="machine_type"):
        driver.apply_capacity(
            "cell",
            {**compute_desired(), "provider_options": {"gpu_type": "nvidia-l4", "gpu_count": 1}},
            context("missing-machine"),
        )
    with pytest.raises(ValueError, match="reservation"):
        driver.apply_capacity(
            "cell",
            {**compute_desired(), "capacity_class": "reserved"},
            context("missing-reservation"),
        )

    generic = GcpCapacityDriver(
        state,
        tpu_provider=FakeProvider(),
        compute_provider=FakeProvider(),
        execution_backend="native-local",
        mode="gcp-tpu",
    )
    with pytest.raises(ValueError, match="cannot satisfy 'compute'"):
        generic.apply_capacity("cell", compute_desired(), context("wrong-agent"))


def test_repeated_same_generation_reconciles_are_replays_not_fence_conflicts(tmp_path) -> None:
    """The controller polls with a fixed key; a replay must never read as stale.

    ``_observe_cell_once_locked`` sends ``reconcile:<cell>:<generation>`` on every
    tick, so the second and every later poll of one cell generation arrives with
    an idempotency key the agent has already recorded. If that were treated as a
    fence conflict the controller would mark a healthy cell NEEDS_RECREATE and
    the autoscaler would drain live capacity on the second poll -- which on a
    Spot TPU means losing a slice that took an hour to be granted.

    """

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    tpu = FakeProvider()
    driver = GcpCapacityDriver(
        state, tpu_provider=tpu, compute_provider=FakeProvider(), execution_backend="native-local"
    )
    driver.apply_capacity("tpu-cell", tpu_desired(), context("capacity:tpu-cell:1"))

    observations = [driver.reconcile("tpu-cell", context("reconcile:tpu-cell:1")) for _ in range(4)]

    assert [observation.state for observation in observations] == [CapacityStatus.WAITING] * 4


def _native_driver(tmp_path, package: str) -> GcpTpuCapacityDriver:
    return GcpTpuCapacityDriver(
        FleetStateStore(LocalBackend(tmp_path / "fleet.json")),
        provider=FakeProvider(),
        execution_backend="eyvan-native",
        native_worker_package=package,
        native_worker_package_sha256=_SHA256,
        native_worker_shared_secret="native-worker-root-secret",
        native_worker_secrets=FakeNativeWorkerSecrets(),
    )


def test_restaging_a_wheel_invalidates_the_host_bootstrap(tmp_path) -> None:
    """Invalidate host bootstrap when its staged wheel changes even if the provider generation stays fixed."""

    record = FleetCellRecord(name="tpu-cell", project="project-a", zone="us-central2-b", provider_generation=1)

    before = _native_driver(tmp_path, "gs://artifacts/eyvan/wheels/aaa/eyvan-0.1.0-py3-none-any.whl")
    after = _native_driver(tmp_path, "gs://artifacts/eyvan/wheels/bbb/eyvan-0.1.0-py3-none-any.whl")

    assert before._bootstrap_key(record) != after._bootstrap_key(record), (
        "a re-staged wheel must invalidate the bootstrap, or the host keeps running old code"
    )


def test_an_unchanged_wheel_does_not_churn_the_bootstrap(tmp_path) -> None:
    """The URI is content-addressed, so identical code must produce one key."""

    record = FleetCellRecord(name="tpu-cell", project="project-a", zone="us-central2-b", provider_generation=1)
    package = "gs://artifacts/eyvan/wheels/aaa/eyvan-0.1.0-py3-none-any.whl"

    assert _native_driver(tmp_path, package)._bootstrap_key(record) == _native_driver(tmp_path, package)._bootstrap_key(
        record
    )


def test_a_new_provider_generation_still_invalidates_it(tmp_path) -> None:
    """Recreation must keep forcing a rebuild; the wheel is an addition, not a swap."""

    package = "gs://artifacts/eyvan/wheels/aaa/eyvan-0.1.0-py3-none-any.whl"
    driver = _native_driver(tmp_path, package)
    first = FleetCellRecord(name="tpu-cell", project="p", zone="z", provider_generation=1)
    second = FleetCellRecord(name="tpu-cell", project="p", zone="z", provider_generation=2)

    assert driver._bootstrap_key(first) != driver._bootstrap_key(second)


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow
