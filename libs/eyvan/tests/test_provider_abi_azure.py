# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Azure driver, held to the same twelve properties as every other.

The third cloud through the same gate, and the point of running it is not that
Azure is likely to be different -- it is that "supported" has to mean the same
thing for the third provider as for the first, or the word stops carrying
information.

It does not establish that Azure behaves this way. Nothing here has run against
a subscription. The six hooks are facts about the fake: machines are counted
from what it holds, and the leak is a VM tagged with a generation this cell
never wrote.
"""

from __future__ import annotations

import typing as tp
from contextlib import contextmanager

import pytest
from test_azure_capacity import FakeArm, desired

from eyvan._internal.json_types import JsonObject
from eyvan.capacity.conformance import (
    CONFORMANCE_CHECK_NAMES,
    ProviderConformanceHarness,
    run_conformance,
)
from eyvan.capacity.drivers.azure import AzureVmCapacityDriver
from eyvan.capacity.drivers.base import (
    CapacityDriver,
    CapacityProviderError,
    CapacityProviderFailureCode,
)
from eyvan.capacity.fleet_state import FleetStateStore
from eyvan.capacity.providers.azure.api import AzureVirtualMachine
from eyvan.capacity.providers.azure.machines import CELL_TAG, FENCE_TAG, GENERATION_TAG, vm_name


class AzureConformanceHarness(ProviderConformanceHarness):
    """Six facts about a fake ARM, none of them a promise."""

    def __init__(self, tmp_path) -> None:
        self._tmp_path = tmp_path
        self._fakes: dict[int, FakeArm] = {}
        self._drivers = 0

    def new_driver(self) -> CapacityDriver:
        self._drivers += 1
        api = FakeArm()
        # A file per driver. Two drivers sharing a store would let one check
        # observe another's cells, which is the failure this isolation exists
        # to prevent and not a detail of the fake.
        store = FleetStateStore.from_uri(str(self._tmp_path / f"state-{self._drivers}.json"))
        driver = AzureVmCapacityDriver(store, api=api, location="eastus")
        self._fakes[id(driver)] = api
        return driver

    def desired_capacity(self, cell_id: str) -> JsonObject:
        del cell_id
        return desired()

    def provider_incarnations(self, driver: CapacityDriver, cell_id: str) -> int:
        """Count machines, not records.

        The leak counter. A VM nothing points at keeps billing until somebody
        finds it by hand, so what matters is what the substrate holds rather
        than what the durable record believes.
        """

        return self._fakes[id(driver)].live_for(cell_id)

    def leak_previous_incarnation(self, driver: CapacityDriver, cell_id: str) -> None:
        """Seed a VM tagged with a generation this cell never wrote.

        What a crash between two mutations leaves behind: hardware carrying the
        cell's name and the wrong incarnation's fence.
        """

        api = self._fakes[id(driver)]
        name = vm_name(cell_id)
        api.machines[name] = AzureVirtualMachine(
            name=name,
            provisioning_state="Succeeded",
            tags={CELL_TAG: cell_id, GENERATION_TAG: "99", FENCE_TAG: "a-fence-this-cell-never-held"},
            private_ip="10.255.255.255",
        )

    @contextmanager
    def provider_unreachable(self, driver: CapacityDriver) -> tp.Iterator[None]:
        """A call that may or may not have taken effect."""

        error = CapacityProviderError(
            "Azure PUT did not complete: connection refused",
            code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
        )
        with self._fakes[id(driver)].failing(error):
            yield

    @contextmanager
    def provider_rejects_permanently(self, driver: CapacityDriver) -> tp.Iterator[None]:
        """A call that will fail identically however often it is retried."""

        error = CapacityProviderError(
            "Azure refused this request (AuthorizationFailed): no permission to create",
            code=CapacityProviderFailureCode.CONFIGURATION,
        )
        with self._fakes[id(driver)].failing(error):
            yield

    @property
    def provider_name(self) -> str:
        return "azure-vm"


@pytest.fixture
def harness(tmp_path) -> AzureConformanceHarness:
    return AzureConformanceHarness(tmp_path)


@pytest.mark.parametrize("name", CONFORMANCE_CHECK_NAMES)
def test_the_azure_driver_satisfies_one_conformance_property(harness, name: str) -> None:
    """One case per property, so a failure names the property rather than a count."""

    report = run_conformance(harness, checks=[check for check in _checks() if check.name == name])

    assert report.passed, report.failures[0].detail if report.failures else "no result"


def _checks():
    from eyvan.capacity.conformance import CONFORMANCE_CHECKS

    return CONFORMANCE_CHECKS


def test_the_harness_counts_the_substrate_rather_than_the_record(harness) -> None:
    """A harness that counted records would pass its own leak check for free.

    This is the check on the check: seed an instance the durable store has never
    heard of, and the counter must see it.
    """

    driver = harness.new_driver()
    harness.leak_previous_incarnation(driver, "cell-a")

    assert harness.provider_incarnations(driver, "cell-a") == 1
    assert driver.state.get("cell-a") is None


def test_the_harness_leaks_something_the_cell_will_actually_disown(harness) -> None:
    """A leak seeded with this cell's own fence would be adopted, correctly.

    The check would then pass while proving nothing, which is the failure mode
    of a harness that fakes its own facts.
    """

    driver = harness.new_driver()
    harness.leak_previous_incarnation(driver, "cell-a")
    api = harness._fakes[id(driver)]

    leaked = api.machines[vm_name("cell-a")]
    assert leaked.tags[GENERATION_TAG] == "99"
    assert leaked.tags[FENCE_TAG] != "fence-1"
