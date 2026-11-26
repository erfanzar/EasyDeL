# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from eyvan.capacity.drivers import gcp as driver_module
from eyvan.capacity.drivers.base import CapacityStatus, MutationContext
from eyvan.capacity.drivers.gcp import GcpCapacityDriver, GcpComputeCapacityDriver, GcpTpuCapacityDriver
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.capacity.providers.gcp import ProviderObservation


def context(key: str, generation: int = 1) -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=f"leader-{generation}")


def desired() -> dict:
    return {
        "project": "project-a",
        "zone": "us-central2-b",
        "accelerator_type": "v5p-16",
        "capacity_class": "spot",
        "executor": "native-local",
    }


class CountableProvider:
    """A local provider that tracks actual owned resources rather than only calls."""

    def __init__(self) -> None:
        self.resources: set[tuple[str, str]] = set()
        self.create_entered = threading.Event()
        self.allow_create = threading.Event()
        self.block_replacement = False
        self.ensured = []

    def ensure(self, record):
        self.ensured.append(record)
        if self.block_replacement and record.name == "cell":
            self.create_entered.set()
            assert self.allow_create.wait(10), "replacement create was not unblocked"
        self.resources.add((record.qr_id, record.metadata["management_uid"]))
        return ProviderObservation(CapacityStatus.HEALTHY, "local capacity ready")

    def delete(self, resource_id, *, project, zone, management_uid=None):
        self.resources.discard((resource_id, management_uid))


def test_release_eviction_cannot_disown_a_resurrected_cell_lock(monkeypatch, tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = CountableProvider()
    driver = GcpTpuCapacityDriver(state, provider=provider, execution_backend="native-local")
    driver.apply_capacity("cell", desired(), context("first"))
    eviction_entered = threading.Event()
    allow_eviction = threading.Event()
    release_started = threading.Event()
    original_forget = driver_module._forget_cell_mutation_lock

    def paused_forget(scope, cell_id):
        if threading.current_thread().name.startswith("old-release"):
            eviction_entered.set()
            assert allow_eviction.wait(10), "eviction was not unblocked"
        original_forget(scope, cell_id)

    monkeypatch.setattr(driver_module, "_forget_cell_mutation_lock", paused_forget)
    errors = []

    def old_release():
        try:
            driver.release("cell", context("first-release", 2))
        except BaseException as exc:
            errors.append(exc)

    old = threading.Thread(target=old_release, name="old-release")
    old.start()
    with ThreadPoolExecutor(max_workers=2) as pool:
        replacement = None
        final_release = None
        premature_release = False
        try:
            assert eviction_entered.wait(10)
            assert state.get("cell") is None
            provider.block_replacement = True
            replacement = pool.submit(driver.apply_capacity, "cell", desired(), context("replacement", 3))
            assert provider.create_entered.wait(10)
            allow_eviction.set()
            old.join(10)
            assert not old.is_alive()
            assert not errors

            def release_replacement():
                release_started.set()
                return driver.release("cell", context("replacement-release", 4))

            final_release = pool.submit(release_replacement)
            assert release_started.wait(10)
            try:
                final_release.result(timeout=0.2)
                premature_release = True
            except TimeoutError:
                pass
        finally:
            allow_eviction.set()
            provider.allow_create.set()
            old.join(10)
        replacement_error = None
        if replacement is not None:
            try:
                replacement.result(timeout=10)
            except Exception as exc:
                replacement_error = exc
        if final_release is not None:
            assert final_release.result(timeout=10).state is CapacityStatus.RELEASED
    assert not premature_release, f"release completed during create; resources left: {provider.resources!r}"
    assert replacement_error is None
    assert state.get("cell") is None
    assert state.cell_tombstone("cell")["generation"] == 4
    assert provider.resources == set()


class LocalSecrets:
    def ensure(self, *, project, management_uid, value, reader_service_account):
        return f"projects/{project}/secrets/eyvan-native-{management_uid}/versions/latest"

    def delete(self, *, project, management_uid):
        pass


class LocalIngress:
    network_tag = "eyvan-nw-profile"

    def ensure(self, *, project, network, port):
        return self.network_tag


def native_desired(kind, network):
    options = {
        "bootstrap_egress": "public-internet",
        "service_account": "worker@project-a.iam.gserviceaccount.com",
    }
    if network is not None:
        options["network"] = network
    if kind == "compute":
        options["machine_type"] = "c4-standard-8"
    return {
        **desired(),
        "capacity_backend": f"gcp-{kind}",
        "accelerator_type": "v5p-16" if kind == "tpu" else "c4-standard-8",
        "executor": "eyvan-native",
        "provider_options": options,
    }


def native_router(tmp_path):
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = CountableProvider()
    router = GcpCapacityDriver(
        state,
        tpu_provider=provider,
        compute_provider=provider,
        execution_backend="eyvan-native",
        native_worker_port=8100,
        native_worker_package="gs://artifacts/eyvan/eyvan.whl",
        native_worker_package_sha256="a" * 64,
        native_worker_shared_secret="local-test-root",
        native_worker_secrets=LocalSecrets(),
        native_worker_public_ingress=LocalIngress(),
    )
    return state, provider, router


def managed_rules(monkeypatch):
    import eyvan.capacity.providers.gcp.native_worker_ingress as ingress_module

    rule = ingress_module.ManagedIngressRule(
        name="owned-rule",
        network=ingress_module.FirewallNetworkReference.from_config(capacity_project="project-a", network=None),
        network_tag=LocalIngress.network_tag,
        port=8100,
        source_range="127.0.0.1/32",
    )
    removed = []
    monkeypatch.setattr(ingress_module, "list_managed_ingress_rules", lambda *, project: [rule])
    monkeypatch.setattr(
        ingress_module, "release_managed_ingress_rule", lambda *, project, rule_name: removed.append(rule_name)
    )
    return removed


@pytest.mark.parametrize(
    "first_kind,sibling_kind", [("tpu", "tpu"), ("compute", "compute"), ("tpu", "compute"), ("compute", "tpu")]
)
@pytest.mark.parametrize("network", [None, "default"])
def test_default_network_shared_ingress_survives_until_last_native_user(
    monkeypatch, tmp_path, first_kind, sibling_kind, network
) -> None:
    state, provider, router = native_router(tmp_path)
    removed = managed_rules(monkeypatch)
    router.apply_capacity("first", native_desired(first_kind, network), context("create-first"))
    router.apply_capacity("sibling", native_desired(sibling_kind, network), context("create-sibling"))
    sibling = state.get("sibling").to_dict()
    assert sibling["metadata"][f"{sibling_kind}_config"]["network"] == network
    assert sibling["state"] == CapacityStatus.HEALTHY.value
    assert sibling["metadata"]["native_worker_endpoint_mode"] == "profile-managed-public"

    router.release("first", context("release-first", 2))

    assert state.get("sibling").to_dict() == sibling
    assert removed == [], "a healthy sibling still needs this rule"
    router.release("sibling", context("release-sibling", 2))
    assert removed == ["owned-rule"]
    assert provider.resources == set()


@pytest.mark.parametrize("kind", ["tpu", "compute"])
@pytest.mark.parametrize("config", [None, [], {"network": []}, {"network": " invalid "}])
def test_unreadable_native_sibling_configuration_preserves_ingress(monkeypatch, tmp_path, kind, config) -> None:
    state, _provider, router = native_router(tmp_path)
    removed = managed_rules(monkeypatch)
    router.apply_capacity("first", native_desired("tpu", None), context("create-first"))
    router.apply_capacity("sibling", native_desired(kind, None), context("create-sibling"))
    state.mutate_record("sibling", lambda record: record.metadata.update({f"{kind}_config": config}))
    sibling = state.get("sibling").to_dict()

    router.release("first", context("release-first", 2))

    assert state.get("sibling").to_dict() == sibling
    assert removed == []


def alias_driver(state, kind, mode, provider):
    if mode == "leaf":
        cls = GcpTpuCapacityDriver if kind == "tpu" else GcpComputeCapacityDriver
        return cls(state, provider=provider, execution_backend="native-local")
    return GcpCapacityDriver(
        state,
        tpu_provider=provider,
        compute_provider=provider,
        execution_backend="native-local",
        mode="auto" if mode == "auto" else kind,
    )


@pytest.mark.parametrize("kind", ["tpu", "compute"])
@pytest.mark.parametrize("mode", ["auto", "pinned", "leaf"])
@pytest.mark.parametrize("backend_form", ["short", "canonical", "generic"])
def test_every_advertised_backend_reaches_its_matching_provider(tmp_path, kind, mode, backend_form) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = CountableProvider()
    driver = alias_driver(state, kind, mode, provider)
    backend = {"short": kind, "canonical": f"gcp-{kind}", "generic": "gcp"}[backend_form]
    assert backend in driver.capabilities().capacity_backends
    document = {**native_desired(kind, None), "capacity_backend": backend, "executor": "native-local"}

    observation = driver.apply_capacity("cell", document, context("create"))

    assert observation.state is CapacityStatus.HEALTHY
    assert len(provider.ensured) == 1
    record = state.get("cell")
    assert record.metadata["capacity_backend"] == f"gcp-{kind}"
    assert record.metadata["capacity_kind"] == kind
    assert f"{kind}_config" in provider.ensured[0].metadata


@pytest.mark.parametrize("kind", ["tpu", "compute"])
@pytest.mark.parametrize("mode", ["pinned", "leaf"])
@pytest.mark.parametrize("backend_form", ["short", "canonical"])
def test_wrong_kind_backend_is_still_refused(tmp_path, kind, mode, backend_form) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    provider = CountableProvider()
    driver = alias_driver(state, kind, mode, provider)
    other = "compute" if kind == "tpu" else "tpu"
    backend = other if backend_form == "short" else f"gcp-{other}"
    document = {**native_desired(kind, None), "capacity_backend": backend, "executor": "native-local"}

    with pytest.raises(ValueError, match="cannot satisfy"):
        driver.apply_capacity("cell", document, context("create"))

    assert state.get("cell") is None
    assert provider.resources == set()
    assert provider.ensured == []


def test_lock_eviction_keeps_existing_waiters_and_cross_cell_progress(monkeypatch, tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    driver = GcpTpuCapacityDriver(state, provider=CountableProvider(), execution_backend="native-local")
    scope = state.backend.mutation_scope
    waiter_attempted = threading.Event()
    late_attempted = threading.Event()
    late_entered = threading.Event()
    waiter_entered = threading.Event()
    allow_waiter = threading.Event()

    class TrackingLock:
        def __init__(self):
            self.lock = threading.RLock()

        def __enter__(self):
            name = threading.current_thread().name
            if name == "waiter":
                waiter_attempted.set()
            elif name == "late":
                late_attempted.set()
            self.lock.acquire()
            return self

        def __exit__(self, *_exc):
            self.lock.release()

    monkeypatch.setattr(driver_module, "RLock", TrackingLock)
    errors = []

    def waiter():
        try:
            with driver._cell_mutation("cell"):
                waiter_entered.set()
                assert allow_waiter.wait(10)
        except BaseException as exc:
            errors.append(exc)

    def late():
        with driver._cell_mutation("cell"):
            late_entered.set()

    waiting = threading.Thread(target=waiter, name="waiter")
    later = threading.Thread(target=late, name="late")
    try:
        with driver._cell_mutation("cell"):
            waiting.start()
            assert waiter_attempted.wait(10)
            # Reentrant acquisition must not forget the outer holder/waiter.
            with driver._cell_mutation("cell"):
                driver_module._forget_cell_mutation_lock(scope, "cell")
            with driver._cell_mutation("other-cell"):
                pass  # Unrelated cells still make progress while a waiter blocks.
        assert waiter_entered.wait(10)
        # The original waiter is now a holder. Another eviction must still
        # preserve the same registry lock for the later operation.
        driver_module._forget_cell_mutation_lock(scope, "cell")
        later.start()
        assert late_attempted.wait(10)
        assert not late_entered.wait(0.2)
    finally:
        allow_waiter.set()
        waiting.join(10)
        if later.ident is not None:
            later.join(10)
    assert not errors
    assert late_entered.is_set()
    assert not waiting.is_alive() and not later.is_alive()
    assert (scope, "cell") not in driver_module._CELL_MUTATION_LOCKS
    assert (scope, "cell") not in driver_module._CELL_MUTATION_LOCK_USERS
