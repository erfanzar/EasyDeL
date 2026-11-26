# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import subprocess
import sys
import typing as tp

import pytest

from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan.capacity.drivers.base import CapacityStatus, MutationContext, StaleMutationError
from eyvan.capacity.drivers.gcp import GcpTpuCapacityDriver
from eyvan.capacity.fleet_state import FleetCellRecord, FleetStateStore, LocalBackend
from eyvan.capacity.providers.gcp import ProviderObservation

_SHA256 = "a" * 64


def metadata_object(record: FleetCellRecord, name: str) -> JsonObject:
    value = record.metadata[name]
    assert is_json_object(value)
    return value


def context(key: str, generation: int = 1, token: str = "leader-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


class FakeProvider:
    def __init__(
        self,
        *,
        events: list[str] | None = None,
        on_ensure: tp.Callable[[FleetCellRecord], None] | None = None,
    ):
        self.reports = [ProviderObservation(CapacityStatus.WAITING, "capacity requested")]
        self.ensured: list[FleetCellRecord] = []
        self.deleted: list[tuple[str, str, str]] = []
        self.deleted_ownership_uids: list[str | None] = []
        self.events = events
        self.on_ensure = on_ensure

    def ensure(self, record: FleetCellRecord) -> ProviderObservation:
        if self.events is not None:
            self.events.append("provider")
        self.ensured.append(record)
        callback = self.on_ensure
        self.on_ensure = None
        if callback is not None:
            callback(record)
        return self.reports.pop(0) if len(self.reports) > 1 else self.reports[0]

    def delete(
        self,
        qr_id: str,
        *,
        project: str,
        zone: str,
        management_uid: str | None = None,
    ) -> None:
        self.deleted.append((qr_id, project, zone))
        self.deleted_ownership_uids.append(management_uid)


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


class FakeNativeWorkerPublicIngress:
    def __init__(self, *, events: list[str] | None = None, error: RuntimeError | None = None) -> None:
        self.events = events
        self.error = error
        self.calls: list[tuple[str, str | None, int]] = []

    def ensure(self, *, project: str, network: str | None, port: int) -> str:
        if self.events is not None:
            self.events.append("ingress")
        self.calls.append((project, network, port))
        if self.error is not None:
            raise self.error
        return "eyvan-nw-profile-managed"


def make_driver(tmp_path, provider: FakeProvider | None = None):
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = provider or FakeProvider()
    return state, provider, GcpTpuCapacityDriver(state, provider=provider, execution_backend="native-local")


def desired() -> dict[str, str]:
    return {
        "project": "project-a",
        "zone": "us-central2-b",
        "accelerator_type": "v5p-16",
        "capacity_class": "spot",
        "executor": "native-local",
    }


def remove_cell_record(state: FleetStateStore, cell_id: str) -> None:
    """Simulate an external state writer removing one registered cell."""

    def erase(document: JsonObject) -> None:
        cells = document["cells"]
        assert is_json_object(cells)
        cells.pop(cell_id)

    state.backend.update(erase)


class DropCellOnNextReadState(FleetStateStore):
    """Inject the loss window between durable registration and bootstrap."""

    def __init__(self, backend: LocalBackend, cell_id: str) -> None:
        super().__init__(backend)
        self.cell_id = cell_id
        self.drop_next_read = False

    def get(self, name: str) -> FleetCellRecord | None:
        record = super().get(name)
        if self.drop_next_read and name == self.cell_id:
            self.drop_next_read = False
            remove_cell_record(self, name)
            return None
        return record


def test_apply_capacity_is_eyvan_owned_and_idempotent(tmp_path) -> None:
    state, provider, driver = make_driver(tmp_path)

    first = driver.apply_capacity("cell-a", desired(), context("request-1"))
    second = driver.apply_capacity("cell-a", desired(), context("request-1"))

    record = state.get("cell-a")
    assert first.state is CapacityStatus.WAITING
    assert second.state is CapacityStatus.WAITING
    assert len(provider.ensured) == 2
    assert record is not None
    assert (record.project, record.zone, record.accelerator_type) == (
        "project-a",
        "us-central2-b",
        "v5p-16",
    )
    assert record.qr_id == "cell-a-r1"
    assert record.fencing_token == "leader-1"
    assert metadata_object(record, "tpu_config")["internal_ips"] is False
    management_uid = record.metadata["management_uid"]
    assert isinstance(management_uid, str)
    assert len(management_uid) == 32
    assert all(item.metadata["management_uid"] == management_uid for item in provider.ensured)


def test_apply_capacity_persists_normalized_tpu_provider_options(tmp_path) -> None:
    state, _provider, driver = make_driver(tmp_path)
    configured = {
        **desired(),
        "provider_options": {
            "runtime_version": "v2-alpha-tpuv5",
            "network": "research",
            "subnetwork": "tpu-workers",
            "internal_ips": True,
            "service_account": "tpu-runtime@project-a.iam.gserviceaccount.com",
            "scopes": ["cloud-platform"],
            "tags": ["eyvan-worker"],
            "metadata": {"enable-oslogin": "TRUE"},
        },
    }

    observation = driver.apply_capacity("cell", configured, context("create"))

    record = state.get("cell")
    assert record is not None
    assert record.runtime_version == "v2-alpha-tpuv5"
    assert record.metadata["tpu_config"] == {
        "runtime_version": "v2-alpha-tpuv5",
        "network": "research",
        "subnetwork": "tpu-workers",
        "internal_ips": True,
        "service_account": "tpu-runtime@project-a.iam.gserviceaccount.com",
        "scopes": ["cloud-platform"],
        "tags": ["eyvan-worker"],
        "metadata": {"enable-oslogin": "TRUE"},
    }
    assert observation.metadata["tpu_config"] == record.metadata["tpu_config"]


def test_reconcile_of_an_already_missing_cell_returns_absent_without_provider_side_effects(tmp_path) -> None:
    state, provider, driver = make_driver(tmp_path)

    observation = driver.reconcile("missing", context("reconcile"))

    assert observation.state is CapacityStatus.ABSENT
    assert provider.ensured == []
    assert state.cell_tombstone("missing") is None


def test_native_queued_tpu_bootstraps_every_host_with_one_cell_scoped_secret(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = FakeProvider()
    secrets = FakeNativeWorkerSecrets()
    accounts = FakeNativeWorkerServiceAccountResolver()
    driver = GcpTpuCapacityDriver(
        state,
        provider=provider,
        execution_backend="eyvan-native",
        native_worker_port=8123,
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_wheelhouse="gs://artifacts/eyvan/wheelhouse.zip",
        native_worker_wheelhouse_sha256=_SHA256,
        native_worker_shared_secret="agent-global-secret",
        native_worker_secrets=secrets,
        native_worker_service_account_resolver=accounts,
    )
    native_desired = {
        **desired(),
        "executor": "eyvan-native",
        "provider_options": {
            "node_count": 2,
            "service_account": "eyvan-worker@project-a.iam.gserviceaccount.com",
            "bootstrap_egress": "private-google-access",
        },
        "runtime": {
            "env": {"HF_HOME": "/var/lib/eyvan/workload/hf"},
            "artifacts": [
                {
                    "source": "gs://artifacts/runtime/easydel.whl",
                    "destination": "/opt/eyvan-runtime/easydel.whl",
                }
            ],
            "init_commands": ["/opt/eyvan-native/venv/bin/python -m pip install /opt/eyvan-runtime/easydel.whl"],
            "container": {
                "image": f"us-docker.pkg.dev/project-a/easydel/runtime@sha256:{'a' * 64}",
                "pull_policy": "if-not-present",
                "privileged": True,
            },
        },
    }

    driver.apply_capacity("cell", native_desired, context("create"))

    record = state.get("cell")
    assert record is not None
    tpu_config = metadata_object(record, "tpu_config")
    native_runtime = metadata_object(record, "native_runtime")
    script = tpu_config["startup_script"]
    assert isinstance(script, str)
    assert tpu_config["node_count"] == 2
    assert record.metadata["execution_backend"] == "eyvan-native"
    assert record.metadata["native_worker_port"] == 8123
    assert native_runtime["env"] == {"HF_HOME": "/var/lib/eyvan/workload/hf"}
    assert native_runtime["container"] == native_desired["runtime"]["container"] | {
        "working_dir": None,
        "mounts": [],
    }
    assert "agent-global-secret" not in script
    assert "--systemd-transient" in script
    assert "--runtime-env-file /etc/eyvan-native/workload-runtime.json" in script
    assert "--container-runtime-file /etc/eyvan-native/container-runtime.json" in script
    assert "docker version >/dev/null 2>&1" in script
    assert "gcloud storage cp gs://artifacts/runtime/easydel.whl /opt/eyvan-runtime/easydel.whl" in script
    assert "/opt/eyvan-native/venv/bin/python -m pip install /opt/eyvan-runtime/easydel.whl" in script
    assert secrets.ensured[0]["management_uid"] == record.metadata["management_uid"]
    assert secrets.ensured[0]["reader_service_account"] == "eyvan-worker@project-a.iam.gserviceaccount.com"
    assert accounts.projects == []
    assert metadata_object(provider.ensured[-1], "tpu_config")["startup_script"] == script

    driver.release("cell", context("release", generation=2, token="leader-2"))

    assert secrets.deleted == [{"project": "project-a", "management_uid": record.metadata["management_uid"]}]


def test_native_queued_tpu_uses_the_default_compute_service_account_when_omitted(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    secrets = FakeNativeWorkerSecrets()
    accounts = FakeNativeWorkerServiceAccountResolver()
    driver = GcpTpuCapacityDriver(
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

    driver.apply_capacity(
        "cell",
        {
            **desired(),
            "executor": "eyvan-native",
            "provider_options": {"bootstrap_egress": "private-google-access"},
        },
        context("create"),
    )

    record = state.get("cell")
    assert record is not None
    assert metadata_object(record, "tpu_config")["service_account"] is None
    assert accounts.projects == ["project-a"]
    assert secrets.ensured[0]["reader_service_account"] == accounts.service_account


def test_native_tpu_bootstrap_fences_a_missing_record_before_provider_capacity(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))

    provider = FakeProvider()
    secrets = FakeNativeWorkerSecrets(on_ensure=lambda: remove_cell_record(state, "cell"))
    driver = GcpTpuCapacityDriver(
        state,
        provider=provider,
        execution_backend="eyvan-native",
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_shared_secret="agent-global-secret",
        native_worker_secrets=secrets,
        native_worker_service_account_resolver=FakeNativeWorkerServiceAccountResolver(),
    )
    native_desired = {
        **desired(),
        "executor": "eyvan-native",
        "provider_options": {"bootstrap_egress": "public-internet"},
    }

    with pytest.raises(StaleMutationError, match="removed before native worker bootstrap could be persisted"):
        driver.apply_capacity("cell", native_desired, context("create"))

    assert provider.ensured == []
    assert state.get("cell") is None
    assert state.cell_tombstone("cell") == {"generation": 1, "fencing_token": "leader-1"}
    assert secrets.deleted == [{"project": "project-a", "management_uid": secrets.ensured[0]["management_uid"]}]
    with pytest.raises(StaleMutationError, match="not newer"):
        driver.apply_capacity("cell", native_desired, context("replay"))

    replacement = driver.apply_capacity("cell", native_desired, context("replacement", generation=2, token="leader-2"))

    assert replacement.state is CapacityStatus.WAITING
    assert len(provider.ensured) == 1


def test_native_tpu_bootstrap_fences_record_loss_before_setup_begins(tmp_path) -> None:
    state = DropCellOnNextReadState(LocalBackend(tmp_path / "fleet.json"), "cell")
    state.drop_next_read = True
    provider = FakeProvider()
    secrets = FakeNativeWorkerSecrets()
    driver = GcpTpuCapacityDriver(
        state,
        provider=provider,
        execution_backend="eyvan-native",
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_shared_secret="agent-global-secret",
        native_worker_secrets=secrets,
        native_worker_service_account_resolver=FakeNativeWorkerServiceAccountResolver(),
    )
    native_desired = {
        **desired(),
        "executor": "eyvan-native",
        "provider_options": {"bootstrap_egress": "public-internet"},
    }

    with pytest.raises(StaleMutationError, match="capacity reconciliation could begin"):
        driver.apply_capacity("cell", native_desired, context("create"))

    assert provider.ensured == []
    assert secrets.ensured == []
    assert state.cell_tombstone("cell") == {"generation": 1, "fencing_token": "leader-1"}
    with pytest.raises(StaleMutationError, match="not newer"):
        driver.apply_capacity("cell", native_desired, context("replay"))


def test_provider_observation_fences_record_loss_and_deletes_owned_capacity(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = FakeProvider(on_ensure=lambda _record: remove_cell_record(state, "cell"))
    driver = GcpTpuCapacityDriver(state, provider=provider, execution_backend="native-local")

    with pytest.raises(StaleMutationError, match="provider observation could be persisted"):
        driver.apply_capacity("cell", desired(), context("create"))

    assert len(provider.ensured) == 1
    assert provider.deleted == [("cell-r1", "project-a", "us-central2-b")]
    assert state.cell_tombstone("cell") == {"generation": 1, "fencing_token": "leader-1"}
    with pytest.raises(StaleMutationError, match="not newer"):
        driver.apply_capacity("cell", desired(), context("replay"))


def test_native_tpu_bootstrap_never_resurrects_a_released_record(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))

    def release_cell() -> None:
        state.tombstone_record("cell", 2, "release", lambda _record: True)

    provider = FakeProvider()
    secrets = FakeNativeWorkerSecrets(on_ensure=release_cell)
    driver = GcpTpuCapacityDriver(
        state,
        provider=provider,
        execution_backend="eyvan-native",
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_shared_secret="agent-global-secret",
        native_worker_secrets=secrets,
        native_worker_service_account_resolver=FakeNativeWorkerServiceAccountResolver(),
    )
    native_desired = {
        **desired(),
        "executor": "eyvan-native",
        "provider_options": {"bootstrap_egress": "public-internet"},
    }

    with pytest.raises(StaleMutationError, match="not newer"):
        driver.apply_capacity("cell", native_desired, context("create"))

    assert provider.ensured == []
    assert state.get("cell") is None
    assert state.cell_tombstone("cell") == {"generation": 2, "fencing_token": "release"}
    with pytest.raises(StaleMutationError, match="not newer"):
        driver.apply_capacity("cell", native_desired, context("replay"))

    replacement = driver.apply_capacity("cell", native_desired, context("replacement", generation=3, token="leader-3"))

    assert replacement.state is CapacityStatus.WAITING
    assert len(provider.ensured) == 1


def test_native_tpu_recreation_fences_record_loss_and_cleans_up_owned_resources(tmp_path) -> None:
    class RemoveCellOnSecondSecret(FakeNativeWorkerSecrets):
        def __init__(self) -> None:
            super().__init__()
            self.ensure_count = 0

        def ensure(self, **kwargs) -> str:
            secret = super().ensure(**kwargs)
            self.ensure_count += 1
            if self.ensure_count == 2:
                remove_cell_record(state, "cell")
            return secret

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = FakeProvider()
    provider.reports = [
        ProviderObservation(CapacityStatus.WAITING),
        ProviderObservation(CapacityStatus.NEEDS_RECREATE, "queued resource failed"),
        ProviderObservation(CapacityStatus.NEEDS_RECREATE, "queued resource remains failed"),
    ]
    secrets = RemoveCellOnSecondSecret()
    driver = GcpTpuCapacityDriver(
        state,
        provider=provider,
        execution_backend="eyvan-native",
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_shared_secret="agent-global-secret",
        native_worker_secrets=secrets,
        native_worker_service_account_resolver=FakeNativeWorkerServiceAccountResolver(),
    )
    native_desired = {
        **desired(),
        "executor": "eyvan-native",
        "provider_options": {"bootstrap_egress": "public-internet"},
    }
    driver.apply_capacity("cell", native_desired, context("create"))
    driver.reconcile("cell", context("reconcile"))

    with pytest.raises(StaleMutationError, match="native worker recreation could be persisted"):
        driver.reconcile("cell", context("reconcile"))

    assert len(provider.ensured) == 3
    assert provider.deleted == [
        ("cell-r1", "project-a", "us-central2-b"),
        ("cell-r2", "project-a", "us-central2-b"),
    ]
    assert state.cell_tombstone("cell") == {"generation": 1, "fencing_token": "leader-1"}
    assert secrets.deleted == [{"project": "project-a", "management_uid": secrets.ensured[0]["management_uid"]}]
    with pytest.raises(StaleMutationError, match="not newer"):
        driver.apply_capacity("cell", native_desired, context("replay"))


def test_profile_managed_public_tpu_ingress_precedes_capacity_and_refreshes_on_reconcile(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    events: list[str] = []
    ingress = FakeNativeWorkerPublicIngress(events=events)
    provider = FakeProvider(events=events)
    driver = GcpTpuCapacityDriver(
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

    driver.apply_capacity(
        "public-cell",
        {
            **desired(),
            "executor": "eyvan-native",
            "provider_options": {"bootstrap_egress": "public-internet"},
        },
        context("create"),
    )

    record = state.get("public-cell")
    assert record is not None
    assert events == ["ingress", "provider"]
    assert ingress.calls == [("project-a", None, 8100)]
    assert metadata_object(record, "tpu_config")["tags"] == ["eyvan-nw-profile-managed"]
    assert record.metadata["native_worker_endpoint_mode"] == "profile-managed-public"

    driver.reconcile("public-cell", context("reconcile"))

    assert ingress.calls == [("project-a", None, 8100), ("project-a", None, 8100)]


def test_private_tpu_workers_never_receive_the_profile_managed_public_tag(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    ingress = FakeNativeWorkerPublicIngress()
    driver = GcpTpuCapacityDriver(
        state,
        provider=FakeProvider(),
        execution_backend="eyvan-native",
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_wheelhouse="gs://artifacts/eyvan/wheelhouse.zip",
        native_worker_wheelhouse_sha256=_SHA256,
        native_worker_shared_secret="agent-global-secret",
        native_worker_secrets=FakeNativeWorkerSecrets(),
        native_worker_service_account_resolver=FakeNativeWorkerServiceAccountResolver(),
        native_worker_public_ingress=ingress,
    )

    driver.apply_capacity(
        "private-cell",
        {
            **desired(),
            "executor": "eyvan-native",
            "provider_options": {"internal_ips": True, "bootstrap_egress": "private-google-access"},
        },
        context("create"),
    )

    record = state.get("private-cell")
    assert record is not None
    assert ingress.calls == []
    assert metadata_object(record, "tpu_config")["tags"] == []
    assert "native_worker_endpoint_mode" not in record.metadata


def test_public_tpu_capacity_is_not_requested_when_profile_ingress_cannot_be_configured(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = FakeProvider()
    ingress = FakeNativeWorkerPublicIngress(error=RuntimeError("local public IPv4 unavailable"))
    driver = GcpTpuCapacityDriver(
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

    with pytest.raises(RuntimeError, match="local public IPv4 unavailable"):
        driver.apply_capacity(
            "public-cell",
            {
                **desired(),
                "executor": "eyvan-native",
                "provider_options": {"bootstrap_egress": "public-internet"},
            },
            context("create"),
        )

    assert ingress.calls == [("project-a", None, 8100)]
    assert provider.ensured == []


def test_reconcile_persists_native_worker_endpoints_and_observed_tpu_vm_count(tmp_path) -> None:
    provider = FakeProvider()
    provider.reports = [
        ProviderObservation(CapacityStatus.WAITING),
        ProviderObservation(
            CapacityStatus.HEALTHY,
            head_ip="10.0.0.2",
            execution_endpoint="http://10.0.0.2:8100",
            worker_endpoints=("http://10.0.0.2:8100", "http://10.0.0.3:8100"),
            observed_tpu_vms=2,
            runtime_process_count=2,
        ),
    ]
    _state, _, driver = make_driver(tmp_path, provider)
    driver.apply_capacity("cell", desired(), context("create"))

    observation = driver.reconcile("cell", context("reconcile"))

    assert observation.metadata["native_worker_endpoints"] == [
        "http://10.0.0.2:8100",
        "http://10.0.0.3:8100",
    ]
    assert observation.observed_tpu_vms == 2
    assert observation.metadata["observed_tpu_vms"] == 2
    assert observation.runtime_process_count == 2
    assert observation.metadata["runtime_process_count"] == 2

    provider.reports = [ProviderObservation(CapacityStatus.UNAVAILABLE, "one host disappeared")]
    lost = driver.reconcile("cell", context("reconcile-after-loss"))
    assert lost.runtime_process_count == 2
    assert lost.observed_tpu_vms is None


@pytest.mark.parametrize(
    "provider_options",
    [
        {"unknown": True},
        {"internal_ips": "false"},
        {"metadata": {"eyvan-uid": "foreign"}},
    ],
)
def test_apply_capacity_rejects_invalid_tpu_provider_options_without_persisting(tmp_path, provider_options) -> None:
    state, _provider, driver = make_driver(tmp_path)

    with pytest.raises(ValueError):
        driver.apply_capacity("cell", {**desired(), "provider_options": provider_options}, context("create"))

    assert state.get("cell") is None


def test_apply_capacity_rejects_native_runtime_for_a_non_native_executor(tmp_path) -> None:
    state, _provider, driver = make_driver(tmp_path)

    with pytest.raises(ValueError, match="native runtime configuration requires execution backend eyvan-native"):
        driver.apply_capacity(
            "cell",
            {**desired(), "runtime": {"env": {"HF_HOME": "/var/lib/eyvan/workload/hf"}}},
            context("create"),
        )

    assert state.get("cell") is None


def test_reconcile_persists_provider_observation(tmp_path) -> None:
    provider = FakeProvider()
    provider.reports = [
        ProviderObservation(CapacityStatus.WAITING),
        ProviderObservation(CapacityStatus.HEALTHY, head_ip="10.0.0.2"),
    ]
    state, _, driver = make_driver(tmp_path, provider)
    driver.apply_capacity("cell", desired(), context("create"))

    observation = driver.reconcile("cell", context("reconcile"))

    assert observation.state is CapacityStatus.HEALTHY
    assert observation.metadata["head_ip"] == "10.0.0.2"
    assert state.get("cell").head_ip == "10.0.0.2"


def test_reconcile_never_recreates_capacity_after_the_agent_is_draining(tmp_path) -> None:
    provider = FakeProvider()
    provider.reports = [
        ProviderObservation(CapacityStatus.WAITING),
        ProviderObservation(CapacityStatus.NEEDS_RECREATE, "queued resource failed"),
    ]
    state, _, driver = make_driver(tmp_path, provider)
    driver.apply_capacity("cell", desired(), context("create"))

    def start_drain(record: FleetCellRecord) -> None:
        record.desired_state = "down"
        record.state = CapacityStatus.NEEDS_RECREATE

    state.mutate_record("cell", start_drain)
    observation = driver.reconcile("cell", context("reconcile"))

    assert observation.state is CapacityStatus.NEEDS_RECREATE
    assert observation.detail == "capacity desired state is down"
    assert len(provider.ensured) == 1
    assert provider.deleted == []


def test_reconcile_durably_requeues_terminal_capacity(tmp_path) -> None:
    provider = FakeProvider()
    provider.reports = [
        ProviderObservation(CapacityStatus.WAITING),
        ProviderObservation(CapacityStatus.NEEDS_RECREATE, "queued resource failed"),
        ProviderObservation(CapacityStatus.NEEDS_RECREATE, "queued resource still failed"),
        ProviderObservation(CapacityStatus.WAITING, "replacement requested"),
    ]
    state, _, driver = make_driver(tmp_path, provider)
    driver.apply_capacity("cell", desired(), context("create"))

    failed = driver.reconcile("cell", context("reconcile"))
    observation = driver.reconcile("cell", context("reconcile"))

    record = state.get("cell")
    assert failed.state is CapacityStatus.NEEDS_RECREATE
    assert observation.state is CapacityStatus.WAITING
    assert provider.deleted == [("cell-r1", "project-a", "us-central2-b")]
    assert record.qr_id == "cell-r2"
    assert record.provider_generation == 2
    assert "recreate_intent" not in record.metadata


def test_reconcile_resumes_a_crashed_capacity_recreate_intent(tmp_path) -> None:
    state, provider, driver = make_driver(tmp_path)
    driver.apply_capacity("cell", desired(), context("create"))

    def interrupted(record):
        record.state = CapacityStatus.RECREATING
        record.provider_generation = 2
        record.qr_id = "cell-r2"
        record.metadata["recreate_intent"] = {
            "previous_qr_id": "cell-r1",
            "target_qr_id": "cell-r2",
            "cell_generation": 1,
        }

    state.mutate_record("cell", interrupted)

    observation = driver.reconcile("cell", context("reconcile"))

    assert observation.state is CapacityStatus.WAITING
    assert provider.deleted == [("cell-r1", "project-a", "us-central2-b")]
    assert "recreate_intent" not in state.get("cell").metadata


def test_release_during_recreate_deletes_source_and_target_capacity(tmp_path) -> None:
    state, provider, driver = make_driver(tmp_path)
    driver.apply_capacity("cell", desired(), context("create"))

    def interrupted(record):
        record.state = CapacityStatus.RECREATING
        record.provider_generation = 2
        record.qr_id = "cell-r2"
        record.metadata["recreate_intent"] = {
            "previous_qr_id": "cell-r1",
            "target_qr_id": "cell-r2",
            "cell_generation": 1,
        }

    state.mutate_record("cell", interrupted)

    driver.release("cell", context("release", generation=2, token="leader-2"))

    assert provider.deleted == [
        ("cell-r1", "project-a", "us-central2-b"),
        ("cell-r2", "project-a", "us-central2-b"),
    ]
    assert state.get("cell") is None


def test_apply_capacity_rejects_stale_leader_and_wrong_executor(tmp_path) -> None:
    _, _, driver = make_driver(tmp_path)
    driver.apply_capacity("cell", desired(), context("new", generation=2, token="leader-2"))

    with pytest.raises(StaleMutationError, match="older"):
        driver.drain("cell", context("old", generation=1, token="leader-1"))
    with pytest.raises(StaleMutationError, match="fencing"):
        driver.drain("cell", context("split-brain", generation=2, token="other"))
    with pytest.raises(ValueError, match="provides 'native-local'"):
        driver.apply_capacity("other", {**desired(), "executor": "ray-jobs"}, context("wrong"))


def test_release_deletes_current_queued_resource_and_is_replay_safe(tmp_path) -> None:
    state, provider, driver = make_driver(tmp_path)
    driver.apply_capacity("cell", desired(), context("create"))
    management_uid = state.get("cell").metadata["management_uid"]
    release = context("release", generation=2, token="leader-2")

    first = driver.release("cell", release)
    second = driver.release("cell", release)

    assert first.state is CapacityStatus.RELEASED
    assert second.state is CapacityStatus.RELEASED
    assert provider.deleted == [("cell-r1", "project-a", "us-central2-b")]
    assert provider.deleted_ownership_uids == [management_uid]
    assert state.get("cell") is None
    # The tombstone carries the management UID so a later release can sweep a
    # secret that an interrupted earlier release left behind.
    assert state.cell_tombstone("cell") == {
        "generation": 2,
        "fencing_token": "leader-2",
        "management_uid": management_uid,
    }


def test_release_does_not_remove_newer_owner_after_provider_call(tmp_path) -> None:
    state, provider, driver = make_driver(tmp_path)
    driver.apply_capacity("cell", desired(), context("create"))

    def steal(qr_id, *, project, zone, **_ownership):
        del qr_id, project, zone

        def mutate(record):
            record.generation = 3
            record.fencing_token = "leader-3"

        state.mutate_record("cell", mutate)

    provider.delete = steal

    with pytest.raises(StaleMutationError, match="changed ownership"):
        driver.release("cell", context("release", generation=2, token="leader-2"))

    assert state.get("cell").generation == 3


def test_release_with_a_consumed_key_still_deletes_live_capacity(tmp_path) -> None:
    state, provider, driver = make_driver(tmp_path)
    driver.apply_capacity("cell", desired(), context("create"))
    management_uid = state.get("cell").metadata["management_uid"]
    # A colliding idempotency key is consumed by another mutation while the
    # record still shows live capacity.
    driver.reconcile("cell", context("release", generation=2, token="leader-2"))

    observation = driver.release("cell", context("release", generation=2, token="leader-2"))

    assert observation.state is CapacityStatus.RELEASED
    assert provider.deleted == [("cell-r1", "project-a", "us-central2-b")]
    assert state.get("cell") is None
    assert state.cell_tombstone("cell") == {
        "generation": 2,
        "fencing_token": "leader-2",
        "management_uid": management_uid,
    }


def test_replayed_release_on_a_deleting_record_still_deletes_capacity(tmp_path) -> None:
    state, provider, driver = make_driver(tmp_path)
    driver.apply_capacity("cell", desired(), context("create"))

    def crashed_release(record: FleetCellRecord) -> None:
        record.desired_state = "down"
        record.state = CapacityStatus.DELETING
        record.generation = 2
        record.fencing_token = "leader-2"
        record.idempotency_keys = [*record.idempotency_keys, "release"]

    state.mutate_record("cell", crashed_release)

    observation = driver.release("cell", context("release", generation=2, token="leader-2"))

    assert observation.state is CapacityStatus.RELEASED
    assert provider.deleted == [("cell-r1", "project-a", "us-central2-b")]
    assert state.get("cell") is None


def test_observation_metadata_never_shadows_record_fields(tmp_path) -> None:
    state, _, driver = make_driver(tmp_path)
    driver.apply_capacity("cell", desired(), context("create"))

    def poison(record: FleetCellRecord) -> None:
        record.metadata["project"] = "poisoned-project"
        record.metadata["zone"] = "poisoned-zone"
        record.metadata["resource_id"] = "poisoned-resource"
        record.metadata["head_ip"] = "203.0.113.99"

    state.mutate_record("cell", poison)

    metadata = driver.observe("cell").metadata
    assert metadata["project"] == "project-a"
    assert metadata["zone"] == "us-central2-b"
    assert metadata["resource_id"] == "cell-r1"
    assert metadata["qr_id"] == "cell-r1"
    assert metadata["head_ip"] is None


def test_drain_fences_absent_incarnation_and_newer_generation_can_recreate(tmp_path) -> None:
    _state, _, driver = make_driver(tmp_path)
    mutation = context("drain", generation=1, token="cell-1")

    observation = driver.drain("cell", mutation)

    assert observation.state is CapacityStatus.DRAINING
    with pytest.raises(StaleMutationError, match="not newer"):
        driver.apply_capacity("cell", desired(), context("late", generation=1, token="cell-1"))
    recreated = driver.apply_capacity("cell", desired(), context("new", generation=2, token="cell-2"))
    assert recreated.generation == 2


def test_a_release_does_not_hold_the_fleet_lock_while_it_talks_to_the_provider(tmp_path) -> None:
    """Two cells release concurrently instead of one waiting out the other's deletes.

    Holding ``_mutation_lock`` and the durable fleet-state lease across provider
    deletes serializes releases for all cells sharing the document. Provider I/O
    must run outside those locks so a slow delete cannot delay unrelated cells.

    The property is asserted by making the first release block *inside* the
    provider call and requiring the second to finish anyway. Under the old code
    the second cannot even start, so this deadlocks rather than fails -- hence
    the timeout on the join, which turns "still serialised" into a failure
    instead of a hung suite.
    """

    import threading

    inside_provider = threading.Event()
    may_finish = threading.Event()

    class BlockingProvider(FakeProvider):
        def delete(self, qr_id: str, *, project: str, zone: str, management_uid: str | None = None) -> None:
            if qr_id.startswith("first"):
                inside_provider.set()
                # Held until the other release has completed, so finishing at
                # all proves the lock was not being held here.
                assert may_finish.wait(timeout=10), "the second release never completed"
            super().delete(qr_id, project=project, zone=zone, management_uid=management_uid)

    state, _provider, driver = make_driver(tmp_path, BlockingProvider())
    driver.apply_capacity("first", desired(), context("create-first"))
    driver.apply_capacity("second", desired(), context("create-second"))

    slow = threading.Thread(
        target=driver.release, args=("first", context("release-first", generation=2, token="leader-2"))
    )
    slow.start()
    assert inside_provider.wait(timeout=10), "the first release never reached the provider"

    driver.release("second", context("release-second", generation=2, token="leader-2"))
    assert state.get("second") is None

    may_finish.set()
    slow.join(timeout=10)
    assert not slow.is_alive(), "the first release never finished"
    assert state.get("first") is None


class _TaggedIngress:
    """An ingress manager that reports its profile tag, as the real one does."""

    def __init__(self, network_tag: str = "eyvan-nw-profile") -> None:
        self.network_tag = network_tag

    def ensure(self, *, project: str, network: str | None, port: int) -> str:
        return self.network_tag


def _native_driver(state, provider, secrets, ingress, accounts=None):
    return GcpTpuCapacityDriver(
        state,
        provider=provider,
        execution_backend="eyvan-native",
        native_worker_port=8100,
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256=_SHA256,
        native_worker_shared_secret="agent-global-secret",
        native_worker_secrets=secrets,
        native_worker_public_ingress=ingress,
        native_worker_service_account_resolver=accounts or FakeNativeWorkerServiceAccountResolver(),
    )


def _register_native_cell(state, cell_id, network="default"):
    def create(record: FleetCellRecord) -> None:
        record.project = "project-a"
        record.state = CapacityStatus.HEALTHY
        record.metadata["execution_backend"] = "eyvan-native"
        record.metadata["compute_config"] = {"network": network}

    state.mutate_or_create_record(cell_id, create, generation=1)


def test_release_sweeps_ingress_rule_whose_last_user_it_was(monkeypatch, tmp_path) -> None:
    import eyvan.capacity.providers.gcp.native_worker_ingress as ingress_module

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = FakeProvider()
    driver = _native_driver(
        state, provider, FakeNativeWorkerSecrets(), _TaggedIngress(), FakeNativeWorkerServiceAccountResolver()
    )
    driver.apply_capacity(
        "cell",
        {
            **desired(),
            "executor": "eyvan-native",
            "provider_options": {"bootstrap_egress": "public-internet", "network": "default"},
        },
        context("create"),
    )
    released_rules: list[str] = []
    managed = [
        ingress_module.ManagedIngressRule(
            name="eyvan-nw-rule-8100",
            network=ingress_module.FirewallNetworkReference.from_config(capacity_project="project-a", network=None),
            network_tag="eyvan-nw-profile",
            port=8100,
            source_range="127.0.0.1/32",
        )
    ]
    monkeypatch.setattr(ingress_module, "list_managed_ingress_rules", lambda *, project: managed)
    monkeypatch.setattr(
        ingress_module, "release_managed_ingress_rule", lambda *, project, rule_name: released_rules.append(rule_name)
    )

    driver.release("cell", context("release", generation=2, token="leader-2"))

    assert released_rules == ["eyvan-nw-rule-8100"]


def test_release_keeps_an_ingress_rule_another_native_cell_still_uses(monkeypatch, tmp_path) -> None:
    import eyvan.capacity.providers.gcp.native_worker_ingress as ingress_module

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = FakeProvider()
    driver = _native_driver(
        state, provider, FakeNativeWorkerSecrets(), _TaggedIngress(), FakeNativeWorkerServiceAccountResolver()
    )
    driver.apply_capacity(
        "cell",
        {
            **desired(),
            "executor": "eyvan-native",
            "provider_options": {"bootstrap_egress": "public-internet", "network": "default"},
        },
        context("create"),
    )
    _register_native_cell(state, "sibling", network="default")
    released_rules: list[str] = []
    managed = [
        ingress_module.ManagedIngressRule(
            name="eyvan-nw-rule-8100",
            network=ingress_module.FirewallNetworkReference.from_config(capacity_project="project-a", network=None),
            network_tag="eyvan-nw-profile",
            port=8100,
            source_range="127.0.0.1/32",
        )
    ]
    monkeypatch.setattr(ingress_module, "list_managed_ingress_rules", lambda *, project: managed)
    monkeypatch.setattr(
        ingress_module, "release_managed_ingress_rule", lambda *, project, rule_name: released_rules.append(rule_name)
    )

    driver.release("cell", context("release", generation=2, token="leader-2"))

    assert released_rules == []


@pytest.mark.parametrize(
    ("endpoint_mode", "ssh_entry", "kept"),
    [("public", None, True), ("tunnel", "203.0.113.9", False), ("internal", None, False)],
)
def test_release_keeps_an_ingress_rule_a_public_attached_tpu_still_uses(
    monkeypatch, tmp_path, endpoint_mode, ssh_entry, kept
) -> None:
    """Retain shared ingress rules while a public attached TPU still uses them."""

    import eyvan.capacity.providers.gcp.native_worker_ingress as ingress_module

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    driver = _native_driver(
        state, FakeProvider(), FakeNativeWorkerSecrets(), _TaggedIngress(), FakeNativeWorkerServiceAccountResolver()
    )
    driver.apply_capacity(
        "cell",
        {
            **desired(),
            "executor": "eyvan-native",
            "provider_options": {"bootstrap_egress": "public-internet", "network": "default"},
        },
        context("create"),
    )

    def attach(record: FleetCellRecord) -> None:
        record.project = "project-a"
        record.state = CapacityStatus.HEALTHY
        record.metadata["capacity_kind"] = "attach"
        record.metadata["endpoint_mode"] = endpoint_mode
        if ssh_entry is not None:
            record.metadata["ssh_entry"] = ssh_entry

    state.mutate_or_create_record("attached", attach, generation=1)
    released_rules: list[str] = []
    managed = [
        ingress_module.ManagedIngressRule(
            name="eyvan-nw-rule-8100",
            network=ingress_module.FirewallNetworkReference.from_config(capacity_project="project-a", network=None),
            network_tag="eyvan-nw-profile",
            port=8100,
            source_range="127.0.0.1/32",
        )
    ]
    monkeypatch.setattr(ingress_module, "list_managed_ingress_rules", lambda *, project: managed)
    monkeypatch.setattr(
        ingress_module, "release_managed_ingress_rule", lambda *, project, rule_name: released_rules.append(rule_name)
    )

    driver.release("cell", context("release", generation=2, token="leader-2"))

    assert released_rules == ([] if kept else ["eyvan-nw-rule-8100"])


def test_release_sweeps_secrets_left_behind_by_interrupted_releases(monkeypatch, tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = FakeProvider()
    secrets = FakeNativeWorkerSecrets()
    accounts = FakeNativeWorkerServiceAccountResolver()
    driver = _native_driver(state, provider, secrets, _TaggedIngress(), accounts)
    driver.apply_capacity(
        "cell",
        {
            **desired(),
            "executor": "eyvan-native",
            "provider_options": {"bootstrap_egress": "public-internet", "network": "default"},
        },
        context("create"),
    )
    live_uid = state.get("cell").metadata["management_uid"]
    # A previous release crashed between its secret delete and its tombstone;
    # the tombstone metadata names the uid whose secret survived.
    crashed_uid = "0123456789abcdef0123456789abcdef"

    def tombstone_with_uid(self, name, generation, fencing_token, predicate, *, metadata=None) -> bool:
        state.backend.update(
            lambda doc: doc.setdefault("cell_tombstones", {}).update(
                {"crashed": {"generation": 1, "fencing_token": "old", "management_uid": crashed_uid}}
            )
        )
        return True

    monkeypatch.setattr(type(driver.state), "tombstone_record", tombstone_with_uid, raising=False)
    monkeypatch.setattr(
        secrets,
        "list_managed",
        lambda *, project: ((crashed_uid, "eyvan-native-crashed-x"), (live_uid, "eyvan-native-live-x")),
        raising=False,
    )
    swept: list[str] = []
    original_delete = type(secrets).delete

    def recording_delete(self, **kwargs):
        swept.append(kwargs["management_uid"])
        original_delete(self, **kwargs)

    monkeypatch.setattr(type(secrets), "delete", recording_delete)

    driver.release("cell", context("release", generation=2, token="leader-2"))

    # The normal per-cell delete runs first; the sweep then removes only the
    # interrupted release's leftover.
    assert swept == [live_uid, crashed_uid]


def test_provider_failure_classification_names_quota_and_stockout() -> None:
    from eyvan.capacity.drivers.base import (
        CapacityProviderFailureCode,
        provider_failure_from_diagnostic,
    )

    quota = provider_failure_from_diagnostic("ERROR: (gcloud.alpha.compute.tpus) Quota exceeded for quota metric")
    stockout = provider_failure_from_diagnostic(
        "The zone 'projects/p/zones/us-central2-b' does not have enough resources available"
    )
    unclassified = provider_failure_from_diagnostic("something entirely unexpected happened")

    assert quota is CapacityProviderFailureCode.QUOTA_EXCEEDED
    assert stockout is CapacityProviderFailureCode.STOCKOUT
    assert unclassified is None


def test_router_sends_record_backed_operations_to_the_owning_kind(tmp_path) -> None:
    from eyvan.capacity.drivers.gcp import GcpCapacityDriver

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    tpu_provider = FakeProvider()
    compute_provider = FakeProvider()
    router = GcpCapacityDriver(
        state,
        tpu_provider=tpu_provider,
        compute_provider=compute_provider,
        execution_backend="native-local",
    )

    router.apply_capacity(
        "cpu-cell",
        {
            **desired(),
            "capacity_backend": "gcp-compute",
            "provider_options": {"machine_type": "n2-standard-8"},
        },
        context("create"),
    )

    assert compute_provider.ensured and not tpu_provider.ensured
    observation = router.observe("cpu-cell")
    assert observation.state is CapacityStatus.WAITING
    assert tpu_provider.deleted == []
    router.release("cpu-cell", context("release", generation=2, token="leader-2"))
    assert len(compute_provider.deleted) == 1
    assert tpu_provider.deleted == []


def test_router_routes_a_damaged_compute_record_by_its_backend_marker(tmp_path) -> None:
    """A compute record that lost its config metadata must not fall through to TPU."""

    from eyvan.capacity.drivers.gcp import GcpCapacityDriver

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    tpu_provider = FakeProvider()
    compute_provider = FakeProvider()
    router = GcpCapacityDriver(
        state,
        tpu_provider=tpu_provider,
        compute_provider=compute_provider,
        execution_backend="native-local",
    )
    router.apply_capacity(
        "gpu-cell",
        {
            **desired(),
            "capacity_backend": "gcp-compute",
            "provider_options": {"machine_type": "n2-standard-8"},
        },
        context("create"),
    )

    def damage(record: FleetCellRecord) -> None:
        record.metadata.pop("capacity_kind", None)
        record.metadata.pop("compute_config", None)

    state.mutate_record("gpu-cell", damage)

    router.release("gpu-cell", context("release", generation=2, token="leader-2"))

    assert len(compute_provider.deleted) == 1
    assert tpu_provider.deleted == []


class IncarnationListingProvider(FakeProvider):
    """A queued-resource API that can enumerate the ids its zone holds."""

    def __init__(self, incarnations: tuple[str, ...]) -> None:
        super().__init__()
        self._incarnations = incarnations

    def list_cell_incarnations(self, *, project: str, zone: str, cell_id: str) -> tuple[str, ...]:
        del project, zone, cell_id
        return self._incarnations


def test_disowned_incarnations_carry_the_cell_id_then_r_then_digits(tmp_path) -> None:
    """The disowned check anchors its suffix so sibling cells are never flagged.

    ``{cell}-r`` as a bare prefix also matches ids belonging to *other* cells
    whose operator-chosen id extends it: cells ``cell-a`` and ``cell-a-r1`` are
    both legal, and reporting the sibling's queued resource as disowned flapped
    the healthy cell into recreate churn. Only ids shaped ``{cell}-r<digits>``
    are this cell's incarnations; everything else belongs to someone else.
    """

    provider = IncarnationListingProvider(
        (
            "cell-a-r9",  # this cell's leaked generation -> disowned
            "cell-a-r1-r2",  # sibling cell "cell-a-r1", generation 2 -> not ours
            "cell-ab-r1",  # different cell id that merely shares the prefix
            "cell-a-r",  # suffix without a generation -> not an incarnation
        )
    )
    state, _, driver = make_driver(tmp_path, provider)
    driver.apply_capacity("cell-a", desired(), context("create"))
    record = state.get("cell-a")
    assert record is not None
    assert record.qr_id == "cell-a-r1"

    observation = driver.observe("cell-a")

    assert observation.state is CapacityStatus.NEEDS_RECREATE
    assert observation.metadata["disowned_incarnations"] == ("cell-a-r9",)


def test_a_cell_whose_listing_holds_only_sibling_ids_is_not_recreated(tmp_path) -> None:
    _state, _, driver = make_driver(tmp_path, IncarnationListingProvider(("cell-a-r1-r2", "cell-ab-r1")))
    driver.apply_capacity("cell-a", desired(), context("create"))

    observation = driver.observe("cell-a")

    assert observation.state is CapacityStatus.WAITING
    assert "disowned_incarnations" not in observation.metadata


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow


@pytest.mark.parametrize("operation", ["apply", "reconcile"])
def test_a_slow_provider_call_for_one_cell_does_not_stall_another(tmp_path, operation: str) -> None:
    """Keep slow provider ensure calls outside shared locks so other cells can apply or reconcile.

    The first call blocks inside the provider while the second must finish;
    a bounded join turns serialization into a failure instead of a hung suite.
    """

    import threading

    inside_provider = threading.Event()
    may_finish = threading.Event()

    class BlockingProvider(FakeProvider):
        def ensure(self, record: FleetCellRecord) -> ProviderObservation:
            if record.name == "first" and not may_finish.is_set():
                inside_provider.set()
                assert may_finish.wait(timeout=10), "the other cell's mutation never completed"
            return super().ensure(record)

    state, _provider, driver = make_driver(tmp_path, BlockingProvider())
    if operation == "reconcile":
        may_finish.set()
        driver.apply_capacity("first", desired(), context("create-first"))
        driver.apply_capacity("second", desired(), context("create-second"))
        may_finish.clear()

        def slow_call() -> None:
            driver.reconcile("first", context("reconcile-first"))

        def fast_call() -> None:
            driver.reconcile("second", context("reconcile-second"))
    else:

        def slow_call() -> None:
            driver.apply_capacity("first", desired(), context("create-first"))

        def fast_call() -> None:
            driver.apply_capacity("second", desired(), context("create-second"))

    slow = threading.Thread(target=slow_call)
    slow.start()
    assert inside_provider.wait(timeout=10), "the first mutation never reached the provider"

    finished = threading.Thread(target=fast_call)
    finished.start()
    finished.join(timeout=10)
    assert not finished.is_alive(), "the second cell waited on the first cell's provider call"
    assert state.get("second") is not None

    may_finish.set()
    slow.join(timeout=10)
    assert not slow.is_alive()
    assert state.get("first") is not None


def test_two_mutations_of_the_same_cell_never_reach_the_provider_together(tmp_path) -> None:
    """Parallelism across cells must not become two creates racing one resource id."""

    import threading

    active = 0
    overlap: list[int] = []
    counter_guard = threading.Lock()
    inside_provider = threading.Event()
    may_finish = threading.Event()

    class CountingProvider(FakeProvider):
        def ensure(self, record: FleetCellRecord) -> ProviderObservation:
            nonlocal active
            with counter_guard:
                active += 1
                overlap.append(active)
            inside_provider.set()
            may_finish.wait(timeout=10)
            try:
                return super().ensure(record)
            finally:
                with counter_guard:
                    active -= 1

    state, _provider, driver = make_driver(tmp_path, CountingProvider())
    first = threading.Thread(target=driver.apply_capacity, args=("cell", desired(), context("create")))
    first.start()
    assert inside_provider.wait(timeout=10)
    second = threading.Thread(target=driver.reconcile, args=("cell", context("reconcile")))
    second.start()
    second.join(timeout=0.5)
    assert second.is_alive(), "a reconcile of the same cell entered while its create was in the provider"

    may_finish.set()
    first.join(timeout=10)
    second.join(timeout=10)
    assert not first.is_alive() and not second.is_alive()
    assert max(overlap) == 1, overlap
    assert state.get("cell") is not None


def test_a_second_agent_process_cannot_touch_a_cell_whose_provider_call_is_in_flight(tmp_path) -> None:
    """The fleet lease no longer spans provider calls, so a per-cell claim must.

    Two agents sharing one fleet-state document -- or a redeploy's outgoing agent,
    still inside ``gcloud ... create`` while its successor takes the retry -- must
    not both drive the same cell's provider resource. The other process is
    represented by the claim it leaves in the shared record.
    """

    from datetime import UTC, datetime, timedelta

    from eyvan.capacity.drivers.base import DriverBusyError
    from eyvan.capacity.drivers.gcp import PROVIDER_OPERATION_KEY

    state, provider, driver = make_driver(tmp_path)
    driver.apply_capacity("cell", desired(), context("create"))
    calls_before = len(provider.ensured)
    foreign = {"owner": "other-host:4242:agent-b", "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat()}

    def plant(record: FleetCellRecord) -> None:
        record.metadata[PROVIDER_OPERATION_KEY] = foreign

    state.mutate_record("cell", plant)

    with pytest.raises(DriverBusyError, match="provider operation in flight"):
        driver.reconcile("cell", context("reconcile"))
    with pytest.raises(DriverBusyError, match="provider operation in flight"):
        driver.apply_capacity("cell", desired(), context("create-again"))

    assert len(provider.ensured) == calls_before, "the provider is never called while another process holds the cell"
    record = state.get("cell")
    assert record is not None and record.metadata[PROVIDER_OPERATION_KEY] == foreign, "its claim is left intact"


@pytest.mark.parametrize("operation", ["replay", "release", "release-replay", "drain", "drain-replay"])
def test_live_foreign_claim_blocks_replayed_apply_release_and_drain(tmp_path, operation: str) -> None:
    from datetime import UTC, datetime, timedelta

    from eyvan.capacity.drivers.base import DriverBusyError
    from eyvan.capacity.drivers.gcp import PROVIDER_OPERATION_KEY

    state, provider, driver = make_driver(tmp_path)
    driver.apply_capacity("cell", desired(), context("create"))
    foreign = {"owner": "other-host:4242:agent-b", "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat()}

    def plant(record: FleetCellRecord) -> None:
        record.metadata[PROVIDER_OPERATION_KEY] = foreign
        if operation == "release-replay":
            record.state = CapacityStatus.DELETING

    state.mutate_record("cell", plant)
    before = state.get("cell").to_dict()
    calls_before = len(provider.ensured)
    with pytest.raises(DriverBusyError, match="provider operation in flight"):
        if operation == "replay":
            driver.apply_capacity("cell", desired(), context("create"))
        elif operation.startswith("release"):
            driver.release("cell", context("create") if operation.endswith("replay") else context("release", 2, "new"))
        else:
            driver.drain("cell", context("create") if operation.endswith("replay") else context("drain", 2, "new"))
    assert len(provider.ensured) == calls_before
    assert provider.deleted == []
    assert state.get("cell").to_dict() == before
    assert state.cell_tombstone("cell") is None


@pytest.mark.parametrize("replay", [False, True])
def test_own_provider_claim_allows_release_and_tombstone_replay(tmp_path, replay: bool) -> None:
    from eyvan.capacity.drivers.gcp import PROVIDER_OPERATION_KEY

    state, provider, driver = make_driver(tmp_path)
    driver.apply_capacity("cell", desired(), context("create"))

    def plant(record: FleetCellRecord) -> None:
        driver._claim_provider_operation(record)
        if replay:
            record.state = CapacityStatus.DELETING

    state.mutate_record("cell", plant)
    assert state.get("cell").metadata[PROVIDER_OPERATION_KEY]["owner"] == state.owner
    release_context = context("create") if replay else context("release", 2, "new")
    assert driver.release("cell", release_context).state is CapacityStatus.RELEASED
    assert state.get("cell") is None
    assert state.cell_tombstone("cell")["generation"] == release_context.generation
    assert len(provider.deleted) == 1
    assert driver.release("cell", release_context).state is CapacityStatus.RELEASED
    assert len(provider.deleted) == 1


@pytest.mark.parametrize("failure_stage", ["delete", "auxiliary", "tombstone"])
def test_release_holds_claim_through_io_and_clears_it_on_failure(tmp_path, monkeypatch, failure_stage: str) -> None:
    from eyvan.capacity.drivers.gcp import PROVIDER_OPERATION_KEY

    state, _provider, driver = make_driver(tmp_path)
    driver.apply_capacity("cell", desired(), context("create"))
    stages = []

    def stage(name):
        def run(*args, **kwargs):
            assert state.get("cell").metadata[PROVIDER_OPERATION_KEY]["owner"] == state.owner
            stages.append(name)
            if name == failure_stage:
                raise RuntimeError("injected release failure")

        return run

    monkeypatch.setattr(driver, "_delete_owned_capacity", stage("delete"))
    monkeypatch.setattr(driver, "_release_auxiliary_resources", stage("auxiliary"))
    monkeypatch.setattr(driver, "_tombstone_after_release", stage("tombstone"))
    with pytest.raises(RuntimeError, match="injected release failure"):
        driver.release("cell", context("release", 2, "new"))
    assert stages[-1] == failure_stage
    assert state.get("cell") is not None
    assert PROVIDER_OPERATION_KEY not in state.get("cell").metadata
    assert state.cell_tombstone("cell") is None


@pytest.mark.parametrize("operation", ["replay", "release"])
def test_real_second_process_cannot_replay_or_release_inflight_create(tmp_path, operation: str) -> None:
    import json
    import threading
    from pathlib import Path

    from eyvan.capacity.drivers.gcp import PROVIDER_OPERATION_KEY

    inside = threading.Event()
    may_finish = threading.Event()
    errors = []

    class BlockingProvider(FakeProvider):
        def ensure(self, record):
            inside.set()
            assert may_finish.wait(timeout=30), "child did not finish in time"
            return super().ensure(record)

    state, provider, driver = make_driver(tmp_path, BlockingProvider())

    def create():
        try:
            driver.apply_capacity("cell", desired(), context("create"))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=create)
    thread.start()
    try:
        assert inside.wait(timeout=10)
        before = state.get("cell").to_dict()
        script = """
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('driver_helpers', sys.argv[1])
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
spec.loader.exec_module(helpers)
state, provider, driver = helpers.make_driver(Path(sys.argv[2]))
error = None
try:
    if sys.argv[3] == 'replay':
        driver.apply_capacity('cell', helpers.desired(), helpers.context('create'))
    else:
        driver.release('cell', helpers.context('release', 2, 'new'))
except Exception as exc:
    error = type(exc).__name__
print(json.dumps({'error': error, 'ensure': len(provider.ensured), 'delete': provider.deleted, 'owner': state.owner}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script, str(Path(__file__).resolve()), str(tmp_path), operation],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        result = json.loads(completed.stdout)
        assert result["owner"] != state.owner
        assert result["error"] == "DriverBusyError", result
        assert result["ensure"] == 0 and result["delete"] == []
        assert state.get("cell").to_dict() == before
        assert state.cell_tombstone("cell") is None
    finally:
        may_finish.set()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert errors == []
    assert len(provider.ensured) == 1 and provider.deleted == []
    assert PROVIDER_OPERATION_KEY not in state.get("cell").metadata


@pytest.mark.parametrize("operation", ["apply", "reconcile", "release"])
def test_a_claim_persisted_before_an_interrupt_is_still_cleared(tmp_path, monkeypatch, operation) -> None:
    """The claim is durable as soon as the fleet-locked mutation commits.

    An interrupt between that commit and the provider call must still clear it,
    or another agent is locked out of the cell until the TTL expires.
    """

    from contextlib import contextmanager

    from eyvan.capacity.drivers.gcp import PROVIDER_OPERATION_KEY

    state, _provider, driver = make_driver(tmp_path)
    driver.apply_capacity("cell", desired(), context("create"))
    original = state.exclusive
    interrupts = [KeyboardInterrupt()]

    @contextmanager
    def interrupting():
        with original():
            yield
        if interrupts:
            raise interrupts.pop()

    monkeypatch.setattr(state, "exclusive", interrupting)

    with pytest.raises(KeyboardInterrupt):
        if operation == "apply":
            driver.apply_capacity("cell", desired(), context("create-2"))
        elif operation == "reconcile":
            driver.reconcile("cell", context("reconcile"))
        else:
            driver.release("cell", context("release", generation=2, token="leader-2"))

    assert not interrupts, "the interrupt fired after the claim was persisted"
    record = state.get("cell")
    assert record is not None
    assert PROVIDER_OPERATION_KEY not in record.metadata


def test_this_process_releases_its_claim_after_the_provider_call(tmp_path) -> None:
    from eyvan.capacity.drivers.gcp import PROVIDER_OPERATION_KEY

    state, _provider, driver = make_driver(tmp_path)
    driver.apply_capacity("cell", desired(), context("create"))
    driver.reconcile("cell", context("reconcile"))

    record = state.get("cell")
    assert record is not None and PROVIDER_OPERATION_KEY not in record.metadata


@pytest.mark.parametrize("stale", ["expired", "dead-local-process"])
def test_a_claim_left_by_a_crashed_process_does_not_wedge_its_cell(tmp_path, stale: str) -> None:
    from datetime import UTC, datetime, timedelta

    from eyvan.capacity.drivers.gcp import PROVIDER_OPERATION_KEY

    state, _provider, driver = make_driver(tmp_path)
    driver.apply_capacity("cell", desired(), context("create"))
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    host = __import__("socket").gethostname()
    claim = (
        {"owner": "other-host:1:x", "expires_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat()}
        if stale == "expired"
        else {"owner": f"{host}:{process.pid}:x", "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat()}
    )

    def plant(record: FleetCellRecord) -> None:
        record.metadata[PROVIDER_OPERATION_KEY] = claim

    state.mutate_record("cell", plant)

    driver.reconcile("cell", context("reconcile"))

    record = state.get("cell")
    assert record is not None and PROVIDER_OPERATION_KEY not in record.metadata


def test_the_claim_never_reaches_the_control_plane_as_cell_metadata(tmp_path) -> None:
    from eyvan.capacity.drivers.gcp import PROVIDER_OPERATION_KEY

    captured: list[dict] = []

    class Peeking(FakeProvider):
        def ensure(self, record: FleetCellRecord) -> ProviderObservation:
            captured.append(dict(record.metadata))
            return super().ensure(record)

    _state, _provider, driver = make_driver(tmp_path, Peeking())
    observation = driver.apply_capacity("cell", desired(), context("create"))

    assert PROVIDER_OPERATION_KEY in captured[0], "held while the provider is being called"
    assert PROVIDER_OPERATION_KEY not in observation.metadata
