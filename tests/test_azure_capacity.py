# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Azure driver, and the fake ARM it is held against.

The third cloud, and the one that tests whether the boundary is real. If the
work that went into D1 was worth doing, a second provider after AWS should be
shorter and duller than the first -- same six methods, same ownership rules, same
conformance suite, no edit anywhere that decides.

What is asserted is not "Azure works". Nothing here has run against a
subscription. It is that the driver's ownership discipline holds, that the two
Azure-specific decisions are made the right way round -- the VM name as
idempotency key, and Spot eviction that deletes rather than deallocates -- and
that a reclamation is reported as a preemption rather than as a machine that
vanished.
"""

from __future__ import annotations

import base64
import typing as tp
from contextlib import contextmanager

import httpx
import pytest

from eyvan._internal.json_types import JsonObject
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.azure import AzureVmCapacityDriver
from eyvan.capacity.drivers.base import (
    CapacityProviderError,
    CapacityProviderFailureCode,
    MutationContext,
    StaleMutationError,
)
from eyvan.capacity.fleet_state import FleetCellRecord, FleetStateStore
from eyvan.capacity.providers.azure.api import API_VERSION, AzureVirtualMachine, HttpAzureApi, failure_for, parse_vm
from eyvan.capacity.providers.azure.machines import CELL_TAG, FENCE_TAG, GENERATION_TAG, vm_name


class FakeArm:
    """The parts of ARM that decide whether the driver is correct.

    Name-keyed, because that is the property the driver relies on instead of a
    client token: a fake that created a second machine for a repeated PUT would
    make the driver look wrong for the wrong reason.
    """

    def __init__(self) -> None:
        self.machines: dict[str, AzureVirtualMachine] = {}
        self.creates: list[tuple[str, JsonObject]] = []
        self.deleted: list[str] = []
        self.location = "eastus"
        self.nics: dict[str, JsonObject] = {}
        self._fail: Exception | None = None

    @contextmanager
    def failing(self, error: Exception) -> tp.Iterator[None]:
        self._fail = error
        try:
            yield
        finally:
            self._fail = None

    def _check(self) -> None:
        if self._fail is not None:
            raise self._fail

    def create_or_update_vm(self, name: str, body: JsonObject) -> AzureVirtualMachine:
        self._check()
        self.creates.append((name, body))
        properties = body.get("properties") or {}
        machine = AzureVirtualMachine(
            name=name,
            provisioning_state="Succeeded",
            tags={str(k): str(v) for k, v in (body.get("tags") or {}).items()},
            private_ip="10.1.0.4",
            vm_size=str((properties.get("hardwareProfile") or {}).get("vmSize") or ""),
            zone=(body.get("zones") or [None])[0],
        )
        self.machines[name] = machine
        return machine

    def get_vm(self, name: str) -> AzureVirtualMachine | None:
        self._check()
        return self.machines.get(name)

    def list_vms(self) -> list[AzureVirtualMachine]:
        self._check()
        return list(self.machines.values())

    def delete_vm(self, name: str) -> None:
        self._check()
        self.deleted.append(name)
        self.machines.pop(name, None)

    def get_network_interface(self, nic_id: str) -> JsonObject | None:
        return self.nics.get(nic_id)

    def live_for(self, cell_id: str) -> int:
        return sum(1 for item in self.machines.values() if item.tags.get(CELL_TAG) == cell_id)


def desired(**overrides: object) -> dict[str, object]:
    document = {
        "capacity_backend": "azure-vm",
        "executor": "eyvan-native",
        "capacity_class": "on_demand",
        "accelerator_type": "cpu",
        "region": "eastus",
        "vm_size": "Standard_D4s_v5",
        "network_interface_id": "/subscriptions/s/resourceGroups/g/providers/Microsoft.Network/networkInterfaces/nic",
    }
    document.update(overrides)
    return document


def context(key: str, *, generation: int = 1, token: str = "fence-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


@pytest.fixture
def driver(tmp_path) -> AzureVmCapacityDriver:
    return AzureVmCapacityDriver(
        FleetStateStore.from_uri(str(tmp_path / "state.json")),
        api=FakeArm(),
        location="eastus",
    )


# --------------------------------------------------------------------------- creating


def test_a_vm_is_tagged_with_who_asked_for_it(driver) -> None:
    """ARM has no notion of a cell, so the tags are the fence."""

    driver.apply_capacity("cell-a", desired(), context("apply-1"))

    machine = next(iter(driver.api.machines.values()))
    assert machine.tags[CELL_TAG] == "cell-a"
    assert machine.tags[GENERATION_TAG] == "1"
    assert machine.tags[FENCE_TAG] == "fence-1"


def test_the_name_is_the_idempotency_key(driver) -> None:
    """What EC2 needs a client token for, ARM answers by construction.

    A retried create either returns what exists or makes what does not, so there
    is no window in which a second machine appears.
    """

    driver.apply_capacity("cell-a", desired(), context("apply-1"))
    driver.apply_capacity("cell-a", desired(), context("apply-2"))

    assert driver.api.live_for("cell-a") == 1


def test_a_long_cell_id_is_hashed_rather_than_truncated() -> None:
    """Truncating makes two distinct cells collide on one machine.

    Which is the worst outcome available here: two incarnations quietly sharing
    hardware, each believing it owns it.
    """

    long_a = "a" * 80 + "-one"
    long_b = "a" * 80 + "-two"

    assert vm_name(long_a) != vm_name(long_b)
    assert len(vm_name(long_a)) <= 64
    assert vm_name(long_a) == vm_name(long_a), "the name is the key, so it must be deterministic"


def test_a_spot_vm_is_deleted_on_eviction_rather_than_deallocated(driver) -> None:
    """Deallocated is the worst of both: no compute, still billing for storage.

    And a machine no scheduler can place work on, holding the name a later cell
    of the same id would need.
    """

    driver.apply_capacity("cell-a", desired(capacity_class="spot"), context("apply-1"))

    _name, body = driver.api.creates[0]
    assert body["properties"]["priority"] == "Spot"
    assert body["properties"]["evictionPolicy"] == "Delete"


def test_user_data_reaches_arm_base64_encoded(driver) -> None:
    """ARM decodes userData itself: sent raw, a cloud-init script boots as
    garbage while provisioningState still reads Succeeded."""

    script = "#!/bin/sh\necho booted > /var/eyvan/booted\n"
    driver.apply_capacity("cell-a", desired(user_data=script), context("apply-1"))

    _name, body = driver.api.creates[0]
    wire = body["properties"]["userData"]
    assert wire != script, "raw text on the wire is what ARM decodes to garbage"
    assert base64.b64decode(wire, validate=True).decode("utf-8") == script


def test_the_network_interface_is_deleted_with_the_machine(driver) -> None:
    """A NIC left behind holds an address and a name the next cell needs."""

    driver.apply_capacity("cell-a", desired(), context("apply-1"))

    _name, body = driver.api.creates[0]
    interface = body["properties"]["networkProfile"]["networkInterfaces"][0]
    assert interface["properties"]["deleteOption"] == "Delete"


# --------------------------------------------------------------------------- ownership


def test_a_vm_from_a_previous_generation_is_not_adopted(driver) -> None:
    driver.api.machines[vm_name("cell-a")] = AzureVirtualMachine(
        name=vm_name("cell-a"),
        provisioning_state="Succeeded",
        tags={CELL_TAG: "cell-a", GENERATION_TAG: "1", FENCE_TAG: "fence-1"},
    )

    observation = driver.apply_capacity("cell-a", desired(), context("apply-1", generation=2, token="fence-2"))

    assert observation.metadata["provisioning_state"] == "Succeeded"
    assert driver.api.machines[vm_name("cell-a")].tags[GENERATION_TAG] == "2"


def test_a_cell_whose_vm_vanished_waits_rather_than_reporting_health(driver) -> None:
    driver.apply_capacity("cell-a", desired(), context("apply-1"))
    driver.api.machines.clear()

    observation = driver.observe("cell-a")

    assert observation.state is CapacityStatus.WAITING
    assert observation.execution_endpoint is None


def test_an_eviction_is_reported_as_a_preemption(driver) -> None:
    """A deleted VM and an evicted one are both gone; only one is a preemption."""

    driver.apply_capacity("cell-a", desired(capacity_class="spot"), context("apply-1"))
    name = vm_name("cell-a")
    current = driver.api.machines[name]
    driver.api.machines[name] = AzureVirtualMachine(
        name=current.name,
        provisioning_state="Succeeded",
        tags=current.tags,
        private_ip=current.private_ip,
        eviction_reason="VM evicted",
    )

    observation = driver.observe("cell-a")

    assert observation.metadata["preemption_reason"] == "VM evicted"


# --------------------------------------------------------------------------- release


def test_release_deletes_the_vm_before_forgetting_it(driver) -> None:
    driver.apply_capacity("cell-a", desired(), context("apply-1"))

    driver.release("cell-a", context("release-1"))

    assert driver.api.live_for("cell-a") == 0
    assert driver.state.get("cell-a") is None


def test_a_stale_generation_cannot_delete_a_live_cell(driver) -> None:
    """Freeing the substrate first is right against a crash and wrong against a
    stale replica, so the fence is checked before anything is deleted."""

    driver.apply_capacity("cell-a", desired(), context("apply-1", generation=3, token="fence-3"))

    with pytest.raises(StaleMutationError):
        driver.release("cell-a", context("release-1", generation=2, token="fence-2"))

    assert driver.api.live_for("cell-a") == 1


# --------------------------------------------------------------------------- reading ARM's answers


def test_a_server_error_is_never_conclusive() -> None:
    """ARM accepted it and something downstream failed; the VM may exist."""

    assert failure_for(503, {"error": {"code": "ServiceUnavailable", "message": "x"}}).code is (
        CapacityProviderFailureCode.PROVIDER_UNAVAILABLE
    )


def test_an_allocation_failure_is_a_stockout_not_a_bad_request() -> None:
    """One says the region is full and the other says the request is wrong."""

    assert failure_for(409, {"error": {"code": "AllocationFailed", "message": "x"}}).code is (
        CapacityProviderFailureCode.STOCKOUT
    )
    assert failure_for(403, {"error": {"code": "AuthorizationFailed", "message": "x"}}).code is (
        CapacityProviderFailureCode.CONFIGURATION
    )


def test_throttling_is_not_a_stockout() -> None:
    assert failure_for(429, {"error": {"code": "TooManyRequests", "message": "x"}}).code is (
        CapacityProviderFailureCode.PROVIDER_UNAVAILABLE
    )


def test_a_vm_parses_out_of_a_real_shaped_document() -> None:
    machine = parse_vm(
        {
            "name": "cell-a",
            "zones": ["1"],
            "tags": {"eyvan-cell": "cell-a"},
            "properties": {
                "provisioningState": "Succeeded",
                "priority": "Spot",
                "hardwareProfile": {"vmSize": "Standard_D4s_v5"},
                "networkProfile": {"networkInterfaces": [{"privateIPAddress": "10.1.0.4"}]},
                "instanceView": {"statuses": [{"code": "PowerState/deallocated", "displayStatus": "VM deallocated"}]},
            },
        }
    )

    assert machine.provisioning_state == "Succeeded"
    assert machine.private_ip == "10.1.0.4"
    assert machine.zone == "1"
    assert machine.eviction_reason == "VM deallocated"


def test_a_deallocated_on_demand_vm_is_not_read_as_an_eviction() -> None:
    """Somebody stopped it. That is not a reclamation and must not be charged
    to the preemption budget."""

    machine = parse_vm(
        {
            "name": "cell-a",
            "properties": {
                "provisioningState": "Succeeded",
                "instanceView": {"statuses": [{"code": "PowerState/deallocated"}]},
            },
        }
    )

    assert machine.eviction_reason is None


def test_a_transport_failure_is_never_conclusive(driver) -> None:
    with (
        driver.api.failing(
            CapacityProviderError("connection reset", code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE)
        ),
        pytest.raises(CapacityProviderError) as raised,
    ):
        driver.apply_capacity("cell-a", desired(), context("apply-1"))

    assert raised.value.code is CapacityProviderFailureCode.PROVIDER_UNAVAILABLE


@pytest.mark.parametrize(("field", "value"), [("vm_size", ""), ("network_interface_id", "")])
def test_a_document_that_cannot_create_is_refused_by_name(driver, field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field.split("_")[0]):
        driver.apply_capacity("cell-a", desired(**{field: value}), context("apply-1"))


def test_an_endpoint_is_resolved_through_the_nic_when_the_vm_document_has_none(tmp_path) -> None:
    """Real ARM documents carry interface references, never addresses."""

    state = FleetStateStore.from_uri(str(tmp_path / "state.json"))
    api = FakeArm()
    driver = AzureVmCapacityDriver(state, api=api, location="eastus")
    applied = driver.apply_capacity("cell-a", desired(), context("apply"))
    assert applied.state is CapacityStatus.HEALTHY

    machine = api.machines[applied.metadata["azure_vm_name"]]
    object.__setattr__(machine, "private_ip", None)

    def forget(record: FleetCellRecord) -> None:
        # The record remembered the address it saw at apply time; the point of
        # this scenario is resolving one that was never embedded.
        record.execution_endpoint = None

    state.mutate_record("cell-a", forget)
    nic_id = "/subscriptions/s/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/cell-a-nic"
    object.__setattr__(machine, "nic_ids", (nic_id,))
    api.nics[nic_id] = {
        "properties": {
            "ipConfigurations": [
                {"properties": {"privateIPAddress": "10.0.0.7"}},
            ]
        }
    }

    observation = driver.observe("cell-a")

    assert observation.execution_endpoint == "10.0.0.7"


def test_get_vm_asks_for_the_instance_view_where_evictions_are_reported() -> None:
    """ARM omits the instance view from a plain GET, so without $expand the
    eviction reader can never fire against a real subscription."""

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "name": "cell-a",
                "properties": {
                    "provisioningState": "Succeeded",
                    "priority": "Spot",
                    "instanceView": {"statuses": [{"code": "PowerState/deallocated", "displayStatus": "VM evicted"}]},
                },
            },
        )

    api = HttpAzureApi(
        subscription_id="s",
        resource_group="g",
        location="eastus",
        token=lambda: "token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    machine = api.get_vm("cell-a")

    params = dict(requests[0].url.params)
    assert params["$expand"] == "instanceView"
    assert params["api-version"] == API_VERSION
    assert machine is not None
    assert machine.eviction_reason == "VM evicted"


def test_parse_nic_private_ip_reads_the_arm_shape() -> None:
    from eyvan.capacity.providers.azure.api import parse_nic_private_ip

    payload = {"properties": {"ipConfigurations": [{"properties": {"privateIPAddress": "10.1.2.3"}}]}}
    assert parse_nic_private_ip(payload) == "10.1.2.3"
    assert parse_nic_private_ip({"properties": {}}) is None
    assert parse_nic_private_ip(None) is None
