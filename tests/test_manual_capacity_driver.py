# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fixed-host capacity: the whole CapacityDriver contract against a non-cloud backend."""

from __future__ import annotations

import pytest

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import MutationContext, StaleMutationError
from eyvan.capacity.drivers.manual import ManualCapacityDriver
from eyvan.capacity.drivers.registry import DEFAULT_CAPACITY_BACKENDS, CapacityBackendRegistry
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.config import EyvanSettings


def _context(key: str, generation: int = 1, token: str = "fence-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


def _desired(*hosts: str) -> dict:
    return {
        "provider": "manual",
        "capacity_backend": "manual",
        "executor": "eyvan-native",
        "accelerator_type": "cpu",
        "capacity_class": "on_demand",
        "region": "on-prem",
        "hosts": list(hosts) or ["http://host-a:8100"],
    }


def _driver(tmp_path) -> ManualCapacityDriver:
    return ManualCapacityDriver(FleetStateStore(LocalBackend(tmp_path / "fleet.json")))


def test_manual_driver_walks_the_whole_capacity_lifecycle(tmp_path) -> None:
    driver = _driver(tmp_path)

    applied = driver.apply_capacity("cell-a", _desired("http://host-a:8100"), _context("apply"))
    assert applied.state is CapacityStatus.HEALTHY
    assert applied.execution_endpoint == "http://host-a:8100"
    assert applied.metadata["capacity_backend"] == "manual"
    assert applied.metadata["resource_id"] == "manual:cell-a"

    assert driver.reconcile("cell-a", _context("reconcile")).state is CapacityStatus.HEALTHY
    assert driver.drain("cell-a", _context("drain")).state is CapacityStatus.DRAINING
    assert driver.release("cell-a", _context("release", generation=2)).state is CapacityStatus.RELEASED
    assert driver.observe("cell-a").state is CapacityStatus.ABSENT


def test_multi_host_cell_publishes_worker_endpoints_for_slice_execution(tmp_path) -> None:
    """eyvan-native resolves a slice executor from native_worker_endpoints."""

    driver = _driver(tmp_path)

    applied = driver.apply_capacity(
        "cell-a",
        _desired("http://host-a:8100", "http://host-b:8100"),
        _context("apply"),
    )

    assert applied.metadata["native_worker_endpoints"] == ["http://host-a:8100", "http://host-b:8100"]
    assert applied.runtime_process_count == 2


def test_management_uid_is_derived_so_worker_tokens_stay_stable(tmp_path) -> None:
    """A random UID would rotate the per-cell token and lock out live workers."""

    first = _driver(tmp_path / "a").apply_capacity("cell-a", _desired(), _context("apply"))
    second = _driver(tmp_path / "b").apply_capacity("cell-a", _desired(), _context("apply"))

    assert first.metadata["management_uid"] == second.metadata["management_uid"]
    assert first.metadata["provider_generation"] == 1


def test_observe_on_an_unknown_cell_is_absent_not_an_error(tmp_path) -> None:
    assert _driver(tmp_path).observe("never-applied").state is CapacityStatus.ABSENT


def test_manual_capacity_refuses_spot(tmp_path) -> None:
    """Operator-owned hosts are never provider-preemptible."""

    desired = {**_desired(), "capacity_class": "spot"}

    with pytest.raises(ValueError, match="never provider-preemptible"):
        _driver(tmp_path).apply_capacity("cell-a", desired, _context("apply"))


@pytest.mark.parametrize(
    ("hosts", "match"),
    (
        ([], "non-empty 'hosts' list"),
        (["host-a:8100"], "must be an http"),
        (["http://a:1", "http://a:1"], "must not contain duplicates"),
        (["   "], "non-empty string"),
    ),
)
def test_manual_capacity_validates_its_host_list(tmp_path, hosts, match) -> None:
    desired = {**_desired(), "hosts": hosts}

    with pytest.raises(ValueError, match=match):
        _driver(tmp_path).apply_capacity("cell-a", desired, _context("apply"))


def test_manual_capacity_enforces_the_executor_contract(tmp_path) -> None:
    desired = {**_desired(), "executor": "native-local"}

    with pytest.raises(ValueError, match="requires executor 'eyvan-native'"):
        _driver(tmp_path).apply_capacity("cell-a", desired, _context("apply"))


def test_release_rejects_a_stale_fencing_token(tmp_path) -> None:
    driver = _driver(tmp_path)
    driver.apply_capacity("cell-a", _desired(), _context("apply"))

    with pytest.raises(StaleMutationError):
        driver.release("cell-a", _context("release", generation=1, token="someone-else"))


def test_capabilities_declare_real_backend_enforcement(tmp_path) -> None:
    """Unlike the local dev driver, the native worker really does fence workloads."""

    capabilities = _driver(tmp_path).capabilities()

    assert capabilities.provider == "manual"
    assert capabilities.executor == "eyvan-native"
    assert capabilities.resource_enforcement == "backend"
    assert "spot" not in capabilities.capacity_classes


# --------------------------------------------------------------------------- the seam


def test_registry_exposes_every_backend_including_manual() -> None:
    assert DEFAULT_CAPACITY_BACKENDS.names() == (
        "aws",
        "aws-ec2",
        "azure",
        "azure-vm",
        "gcp",
        "gcp-attach",
        "gcp-compute",
        "gcp-tpu",
        "k8s",
        "kubernetes",
        "local",
        "manual",
        "onprem-ssh",
    )


def test_registry_builds_a_manual_driver_from_settings(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))

    driver = DEFAULT_CAPACITY_BACKENDS.create(
        "manual",
        state=state,
        settings=EyvanSettings(capacity_backend="manual"),
    )

    assert isinstance(driver, ManualCapacityDriver)


def test_registry_normalizes_names_and_rejects_unknown_backends(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    settings = EyvanSettings()

    assert DEFAULT_CAPACITY_BACKENDS.create("  MANUAL  ", state=state, settings=settings) is not None
    with pytest.raises(ValueError, match="unknown Eyvan capacity backend"):
        DEFAULT_CAPACITY_BACKENDS.create("not-a-provider", state=state, settings=settings)


def test_registry_rejects_duplicate_registration_without_replace() -> None:
    registry = CapacityBackendRegistry()
    registry.register("x", lambda name, *, state, settings: None)  # type: ignore[arg-type,return-value]

    with pytest.raises(ValueError, match="already registered"):
        registry.register("x", lambda name, *, state, settings: None)  # type: ignore[arg-type,return-value]
    registry.register("x", lambda name, *, state, settings: None, replace=True)  # type: ignore[arg-type,return-value]


def test_drain_refuses_a_superseded_fence(tmp_path) -> None:
    """A controller that lost the cell must not be able to drain it."""

    driver = _driver(tmp_path)
    driver.apply_capacity("cell-a", _desired("http://host-a:8100"), _context("apply"))

    with pytest.raises(StaleMutationError):
        driver.drain("cell-a", _context("drain", generation=0))

    assert driver.observe("cell-a").state is CapacityStatus.HEALTHY
