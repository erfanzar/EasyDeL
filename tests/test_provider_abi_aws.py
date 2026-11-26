# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The EC2 driver, held to the same twelve properties as every other.

"Supported" is meant to stop being a claim somebody makes. A second cloud that
passes the same suite as the first is a second cloud; one that merely exists is
a file. So this is the acceptance bar for D2, and the only claim it establishes:
the driver's fencing, idempotency and failure semantics hold against a substrate
that behaves as EC2's documented API says EC2 does.

It does not establish that AWS behaves that way. Nothing here has run against an
account. The six hooks are facts about the fake -- incarnations are counted from
the instances it actually holds, and the leak is an instance tagged with a
generation this cell never wrote -- which makes them honest about the substrate
they describe and silent about the one they do not.
"""

from __future__ import annotations

import typing as tp
from contextlib import contextmanager

import pytest
from test_aws_capacity import FakeEc2, desired

from eyvan._internal.json_types import JsonObject
from eyvan.capacity.conformance import (
    CONFORMANCE_CHECK_NAMES,
    ProviderConformanceHarness,
    run_conformance,
)
from eyvan.capacity.drivers.aws import AwsEc2CapacityDriver
from eyvan.capacity.drivers.base import (
    CapacityDriver,
    CapacityProviderError,
    CapacityProviderFailureCode,
)
from eyvan.capacity.fleet_state import FleetStateStore
from eyvan.capacity.providers.aws.api import Ec2Instance
from eyvan.capacity.providers.aws.instances import CELL_TAG, FENCE_TAG, GENERATION_TAG


class AwsConformanceHarness(ProviderConformanceHarness):
    """Six facts about a fake EC2, none of them a promise."""

    def __init__(self, tmp_path) -> None:
        self._tmp_path = tmp_path
        self._fakes: dict[int, FakeEc2] = {}
        self._drivers = 0

    def new_driver(self) -> CapacityDriver:
        self._drivers += 1
        api = FakeEc2()
        # A file per driver. Two drivers sharing a store would let one check
        # observe another's cells, which is the failure this isolation exists
        # to prevent and not a detail of the fake.
        store = FleetStateStore.from_uri(str(self._tmp_path / f"state-{self._drivers}.json"))
        driver = AwsEc2CapacityDriver(store, api=api)
        self._fakes[id(driver)] = api
        return driver

    def desired_capacity(self, cell_id: str) -> JsonObject:
        del cell_id
        return desired()

    def provider_incarnations(self, driver: CapacityDriver, cell_id: str) -> int:
        """Count instances, not records.

        The leak counter. An EC2 instance nothing points at keeps billing until
        somebody finds it by hand, so what matters is what the substrate holds
        rather than what the durable record believes.
        """

        return self._fakes[id(driver)].live_for(cell_id)

    def leak_previous_incarnation(self, driver: CapacityDriver, cell_id: str) -> None:
        """Seed an instance tagged with a generation this cell never wrote.

        What a crash between two mutations leaves behind: hardware carrying the
        cell's name and the wrong incarnation's fence.
        """

        api = self._fakes[id(driver)]
        api.instances["i-leaked"] = Ec2Instance(
            instance_id="i-leaked",
            state="running",
            tags={CELL_TAG: cell_id, GENERATION_TAG: "99", FENCE_TAG: "a-fence-this-cell-never-held"},
            private_ip="10.255.255.255",
        )

    @contextmanager
    def provider_unreachable(self, driver: CapacityDriver) -> tp.Iterator[None]:
        """A call that may or may not have taken effect."""

        error = CapacityProviderError(
            "EC2 RunInstances did not complete: connection refused",
            code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
        )
        with self._fakes[id(driver)].failing(error):
            yield

    @contextmanager
    def provider_rejects_permanently(self, driver: CapacityDriver) -> tp.Iterator[None]:
        """A call that will fail identically however often it is retried."""

        error = CapacityProviderError(
            "EC2 refused this request (UnauthorizedOperation): no permission to launch",
            code=CapacityProviderFailureCode.CONFIGURATION,
        )
        with self._fakes[id(driver)].failing(error):
            yield

    @property
    def provider_name(self) -> str:
        return "aws-ec2"


@pytest.fixture
def harness(tmp_path) -> AwsConformanceHarness:
    return AwsConformanceHarness(tmp_path)


@pytest.mark.parametrize("name", CONFORMANCE_CHECK_NAMES)
def test_the_aws_driver_satisfies_one_conformance_property(harness, name: str) -> None:
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

    assert api.instances["i-leaked"].tags[GENERATION_TAG] == "99"
    assert api.instances["i-leaked"].tags[FENCE_TAG] != "fence-1"
