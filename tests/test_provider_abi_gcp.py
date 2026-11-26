# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The GCP TPU driver, held to the same twelve properties as every other.

GCP is the reference implementation -- the driver every other one was shaped
after -- which is exactly why it must be *measured* rather than trusted: the
suite exists because two real AWS bugs were caught by properties nobody had
re-run against GCP since it was written.

As with the AWS harness, nothing here has run against a project. The six hooks
are facts about a fake that models queued resources as countable incarnations:
one is created per successful ensure, one is removed per delete, and the leak is
a queued resource tagged with a generation this cell never wrote.
"""

from __future__ import annotations

import typing as tp
from contextlib import contextmanager

import pytest
from test_gcp_driver import FakeProvider, desired

from eyvan._internal.json_types import JsonObject
from eyvan.capacity.conformance import (
    CONFORMANCE_CHECK_NAMES,
    ProviderConformanceHarness,
    run_conformance,
)
from eyvan.capacity.drivers.base import (
    CapacityDriver,
    CapacityProviderError,
    CapacityProviderFailureCode,
)
from eyvan.capacity.drivers.gcp import GcpTpuCapacityDriver
from eyvan.capacity.fleet_state import FleetStateStore


class CountingProvider(FakeProvider):
    """A fake queued-resource API whose live resources are countable.

    ``ensure`` materializes one incarnation and ``delete`` removes exactly the
    one named, so ``live_for`` reports what the substrate holds rather than what
    any record claims.
    """

    def __init__(self) -> None:
        super().__init__()
        self.live: dict[str, list[str]] = {}
        self._incarnation_cell: dict[str, str] = {}
        self._failure: Exception | None = None

    @contextmanager
    def failing(self, error: Exception) -> tp.Iterator[None]:
        """Make ensure and delete raise ``error`` until the context exits."""

        self._failure = error
        try:
            yield
        finally:
            self._failure = None

    def _check_failure(self) -> None:
        if self._failure is not None:
            raise self._failure

    def list_cell_incarnations(self, *, project: str, zone: str, cell_id: str) -> tuple[str, ...]:
        """Return every queued-resource id the substrate holds for ``cell_id``."""

        self._check_failure()
        return tuple(self.live.get(cell_id, ()))

    def ensure(self, record):
        self._check_failure()
        result = super().ensure(record)
        cell = record.name
        incarnation = f"{record.qr_id}"
        self._incarnation_cell[incarnation] = cell
        # A real API materializes one resource per id: re-ensuring the same
        # queued resource does not create a second one.
        incarnations = self.live.setdefault(cell, [])
        if incarnation not in incarnations:
            incarnations.append(incarnation)
        return result

    def delete(self, qr_id: str, *, project: str, zone: str, management_uid: str | None = None) -> None:
        self._check_failure()
        super().delete(qr_id, project=project, zone=zone, management_uid=management_uid)
        cell = self._incarnation_cell.pop(qr_id, None)
        if cell is not None:
            incarnations = self.live.get(cell, [])
            if qr_id in incarnations:
                incarnations.remove(qr_id)

    def live_for(self, cell_id: str) -> int:
        return len(self.live.get(cell_id, []))


class GcpConformanceHarness(ProviderConformanceHarness):
    """Six facts about a fake queued-resource API, none of them a promise."""

    def __init__(self, tmp_path) -> None:
        self._tmp_path = tmp_path
        self._fakes: dict[int, CountingProvider] = {}
        self._drivers = 0

    def new_driver(self) -> CapacityDriver:
        self._drivers += 1
        provider = CountingProvider()
        # A file per driver. Two drivers sharing a store would let one check
        # observe another's cells, which is the failure this isolation exists
        # to prevent and not a detail of the fake.
        store = FleetStateStore.from_uri(str(self._tmp_path / f"state-{self._drivers}.json"))
        driver = GcpTpuCapacityDriver(store, provider=provider, execution_backend="native-local")
        self._fakes[id(driver)] = provider
        return driver

    def desired_capacity(self, cell_id: str) -> JsonObject:
        del cell_id
        return desired()

    def provider_incarnations(self, driver: CapacityDriver, cell_id: str) -> int:
        """Count queued resources, not records.

        The leak counter. A queued resource nothing points at keeps its
        reservation until somebody finds it by hand, so what matters is what
        the substrate holds rather than what the durable record believes.
        """

        return self._fakes[id(driver)].live_for(cell_id)

    def leak_previous_incarnation(self, driver: CapacityDriver, cell_id: str) -> None:
        """Seed a queued resource from a generation this cell never wrote.

        What a crash between two mutations leaves behind: hardware carrying the
        cell's resource-id scheme but the wrong incarnation's fence.
        """

        provider = self._fakes[id(driver)]
        leaked = f"{cell_id}-r99"
        provider._incarnation_cell[leaked] = cell_id
        provider.live.setdefault(cell_id, []).append(leaked)

    @contextmanager
    def provider_unreachable(self, driver: CapacityDriver) -> tp.Iterator[None]:
        """A call that may or may not have taken effect."""

        error = CapacityProviderError(
            "TPU API did not complete: connection refused",
            code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
        )
        with self._fakes[id(driver)].failing(error):
            yield

    @contextmanager
    def provider_rejects_permanently(self, driver: CapacityDriver) -> tp.Iterator[None]:
        """A call that will fail identically however often it is retried."""

        error = CapacityProviderError(
            "TPU API refused this request (PERMISSION_DENIED): missing tpu.nodes.create",
            code=CapacityProviderFailureCode.CONFIGURATION,
        )
        with self._fakes[id(driver)].failing(error):
            yield

    @property
    def provider_name(self) -> str:
        return "gcp-tpu"


@pytest.fixture
def harness(tmp_path) -> GcpConformanceHarness:
    return GcpConformanceHarness(tmp_path)


@pytest.mark.parametrize("name", CONFORMANCE_CHECK_NAMES)
def test_the_gcp_driver_satisfies_one_conformance_property(harness, name: str) -> None:
    """One case per property, so a failure names the property rather than a count."""

    report = run_conformance(harness, checks=[check for check in _checks() if check.name == name])

    assert report.passed, report.failures[0].detail if report.failures else "no result"


def _checks():
    from eyvan.capacity.conformance import CONFORMANCE_CHECKS

    return CONFORMANCE_CHECKS


def test_the_harness_counts_the_substrate_rather_than_the_record(harness) -> None:
    """A harness that counted records would pass its own leak check for free.

    This is the check on the check: seed a queued resource the durable store has
    never heard of, and the counter must see it.
    """

    driver = harness.new_driver()
    harness.leak_previous_incarnation(driver, "cell-a")

    assert harness.provider_incarnations(driver, "cell-a") == 1
    assert driver.state.get("cell-a") is None


def test_the_harness_leaks_something_the_cell_will_actually_disown(harness) -> None:
    """A leak seeded inside the cell's own generation would be adopted, correctly.

    The check would then pass while proving nothing, which is the failure mode
    of a harness that fakes its own facts.
    """

    driver = harness.new_driver()
    harness.leak_previous_incarnation(driver, "cell-a")
    provider = harness._fakes[id(driver)]

    # Generation 99 can never collide with a cell that has not applied yet.
    assert "cell-a-r99" in provider.live["cell-a"]
    assert driver.state.get("cell-a") is None


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow
