# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The contract a carbon number would have to satisfy, and the placement path it may not touch.

No provider is implemented. That is the deliverable: the honest version of this
feature is small and the dishonest version is easy -- publish a plausible
grams-per-kWh figure, let it steer placement, and nobody can tell it apart from
a measured one. A wrong carbon number that silently influences placement is
worse than no carbon number, because the resulting decisions look deliberate.

So what ships is the shape: nullable with a reason code, provenance-carrying,
freshness-gated, and structurally excluded from placement.
"""

from __future__ import annotations

import ast
import pathlib
from datetime import UTC, datetime, timedelta

import pytest

from eyvan.domain.carbon import (
    CarbonIntensity,
    CarbonUnavailable,
    CarbonUnavailableReason,
    NoCarbonSource,
    fresh_reading,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def _reading(**kwargs) -> CarbonIntensity:
    return CarbonIntensity(
        **{
            "region": "us-central1",
            "grams_co2e_per_kwh": 410.0,
            "source": "electricitymaps",
            "observed_at": NOW,
            **kwargs,
        }
    )


# --------------------------------------------------------------------------- nullable


def test_no_configured_provider_reports_a_reason_rather_than_a_number() -> None:
    """Zero is the most dangerous default this type could have.

    It says "this region is carbon-free" -- the strongest claim available --
    arrived at by having no data at all.
    """

    reading = NoCarbonSource().read("us-central1")

    assert isinstance(reading, CarbonUnavailable)
    assert reading.reason is CarbonUnavailableReason.NO_PROVIDER
    assert reading.detail


def test_absence_is_a_value_and_not_an_exception() -> None:
    """Absence is the *ordinary* case: the shipped configuration has no provider.

    Making the ordinary case raise would push every caller into a try block for
    a condition that is not an error.
    """

    assert isinstance(NoCarbonSource().read("anywhere"), CarbonUnavailable)


def test_an_unrecognized_reason_is_refused() -> None:
    """A free-form reason string would defeat branching on it."""

    with pytest.raises(ValueError, match="known reason code"):
        CarbonUnavailable("probably fine")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- provenance


def test_a_reading_cannot_be_built_without_a_source() -> None:
    """The point of the type.

    A reading constructible without a source would let an unattributed number
    look identical to a measured one, and no reader could tell them apart.
    """

    with pytest.raises(ValueError, match="unattributed number is not a reading"):
        _reading(source="")


def test_a_reading_requires_a_timezone_aware_observation_time() -> None:
    """A naive timestamp makes the freshness gate silently wrong by hours."""

    with pytest.raises(ValueError, match="timezone-aware"):
        _reading(observed_at=datetime(2026, 7, 29, 12, 0))


def test_a_negative_intensity_is_refused() -> None:
    """A grid cannot be better than carbon-free, and a negative would rank first."""

    with pytest.raises(ValueError, match="cannot be negative"):
        _reading(grams_co2e_per_kwh=-1.0)


# --------------------------------------------------------------------------- freshness


def test_a_fresh_reading_passes_through() -> None:
    """The gate must not reject the data it exists to admit."""

    reading = _reading(observed_at=NOW - timedelta(minutes=30))

    assert fresh_reading(reading, now=NOW) is reading


def test_a_stale_reading_becomes_unavailable_rather_than_being_used() -> None:
    """Grid intensity moves hour to hour.

    A reading from last week describes a grid that no longer exists, and using
    it is indistinguishable from using a current one.
    """

    result = fresh_reading(_reading(observed_at=NOW - timedelta(days=7)), now=NOW)

    assert isinstance(result, CarbonUnavailable)
    assert result.reason is CarbonUnavailableReason.STALE
    assert "7 days" in result.detail


def test_a_reading_from_the_future_is_refused() -> None:
    """Clock skew large enough to produce one is itself the signal.

    Whatever the timestamps mean, it is not what the gate assumes they mean.
    """

    result = fresh_reading(_reading(observed_at=NOW + timedelta(hours=2)), now=NOW)

    assert isinstance(result, CarbonUnavailable)
    assert result.reason is CarbonUnavailableReason.STALE
    assert "future" in result.detail


def test_the_gate_does_not_manufacture_a_reason_for_an_absent_one() -> None:
    """An unavailable reading keeps the reason it arrived with.

    Overwriting NO_PROVIDER with STALE would report a provider that had stopped
    answering, where in fact there was never one.
    """

    absent = CarbonUnavailable(CarbonUnavailableReason.NO_PROVIDER, "nothing configured")

    assert fresh_reading(absent, now=NOW) is absent


# --------------------------------------------------------------------------- exclusion


PLACEMENT_MODULES = (
    "eyvan/domain/scheduler.py",
    "eyvan/capacity/planning.py",
    "eyvan/server/service/scheduling.py",
    "eyvan/server/service/locality.py",
    "eyvan/server/service/groups.py",
)


def test_no_placement_code_imports_the_carbon_module() -> None:
    """The fourth property, enforced rather than documented.

    A wrong carbon number that silently influences placement is worse than none,
    because the decisions it produces look deliberate. Until someone brings a
    source they trust, reporting is the only sanctioned consumer -- and this
    fails the moment that stops being true, which is the only way the rule
    survives a future contributor who has not read the docstring.
    """

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    for relative in PLACEMENT_MODULES:
        path = root / relative
        assert path.exists(), f"{relative} moved; update this guard rather than deleting it"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and "carbon" in node.module:
                offenders.append(f"{relative}:{node.lineno}")
            elif isinstance(node, ast.Import):
                offenders.extend(f"{relative}:{node.lineno}" for alias in node.names if "carbon" in alias.name)

    assert not offenders, (
        "carbon must not reach the placement path; a number nobody has validated would silently "
        f"steer scheduling from: {', '.join(offenders)}"
    )


def test_the_placement_score_has_no_carbon_key() -> None:
    """Ranking is the mechanism by which a bad number would do damage.

    The import guard would already catch it, but this states the property in the
    terms the scheduler is written in: a key here is a placement decision.
    """

    from eyvan.domain.scheduler import PlacementScore

    assert not [field for field in PlacementScore.__dataclass_fields__ if "carbon" in field]


# --------------------------------------------------------------------------- reporting


async def _report():
    """Ask a real control plane for its carbon report."""

    from eyvan.records import (
        CapacityPoolRecord,
        Principal,
        ResourceVector,
        Role,
        RoleGrant,
    )
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    admin = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    async with repository.transaction() as tx:
        for index, region in enumerate(("us-east5", "us-central1", "us-east5")):
            await tx.put_capacity_pool(
                CapacityPoolRecord(
                    id=f"pool-{index}",
                    fleet_name="research",
                    node_type=f"n{index}",
                    provider="local",
                    region=region,
                    capacity=ResourceVector(cpu=8, memory_gib=64),
                    allocatable=ResourceVector(cpu=8, memory_gib=64),
                    agent_endpoint="http://agent:8090",
                )
            )
    return await service.carbon_report(admin)


async def test_the_report_names_the_absence_rather_than_returning_nothing() -> None:
    """The deliverable.

    A carbon feature whose unconfigured state is an empty response cannot be
    told apart from a broken one; one whose unconfigured state is zero claims
    every grid is clean. This says ``no_provider``, per region, in both.
    """

    report = await _report()

    assert not report.configured
    assert [region.reason for region in report.regions] == ["no_provider", "no_provider"]
    assert all(region.intensity is None for region in report.regions)
    assert all(region.detail for region in report.regions)


async def test_the_report_covers_the_regions_the_fleet_occupies_once_each() -> None:
    """Only regions with a pool, and one entry apiece.

    A global table of grid intensities is a dataset rather than a report, and
    one whose accuracy nobody here could vouch for.
    """

    report = await _report()

    assert [region.region for region in report.regions] == ["us-central1", "us-east5"]


async def test_a_configured_source_is_reported_with_its_provenance() -> None:
    """The path a real provider would take, exercised without inventing one."""

    from eyvan.records import Principal, Role, RoleGrant
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    class FixedSource:
        def read(self, region: str):
            return CarbonIntensity(
                region=region,
                grams_co2e_per_kwh=91.0,
                source="operator-supplied",
                observed_at=datetime.now(tz=UTC),
            )

    service = ControlPlaneService(InMemoryRepository(), scheduler=CoreSchedulingService(), carbon=FixedSource())
    async with service.repository.transaction() as tx:
        from eyvan.records import CapacityPoolRecord, ResourceVector

        await tx.put_capacity_pool(
            CapacityPoolRecord(
                id="pool-0",
                fleet_name="research",
                node_type="n0",
                provider="local",
                region="us-east5",
                capacity=ResourceVector(cpu=8, memory_gib=64),
                allocatable=ResourceVector(cpu=8, memory_gib=64),
                agent_endpoint="http://agent:8090",
            )
        )

    report = await service.carbon_report(Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),)))

    assert report.configured
    assert report.regions[0].intensity == 91.0
    assert report.regions[0].source == "operator-supplied"
    assert report.regions[0].reason is None


async def test_a_stale_source_reading_is_reported_as_unavailable_not_as_a_number() -> None:
    """The freshness gate applied where the report is built, not per provider.

    A source that forgets it would otherwise publish week-old numbers as current.
    """

    from eyvan.records import CapacityPoolRecord, Principal, ResourceVector, Role, RoleGrant
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    class StaleSource:
        def read(self, region: str):
            return CarbonIntensity(
                region=region,
                grams_co2e_per_kwh=91.0,
                source="operator-supplied",
                observed_at=datetime.now(tz=UTC) - timedelta(days=3),
            )

    service = ControlPlaneService(InMemoryRepository(), scheduler=CoreSchedulingService(), carbon=StaleSource())
    async with service.repository.transaction() as tx:
        await tx.put_capacity_pool(
            CapacityPoolRecord(
                id="pool-0",
                fleet_name="research",
                node_type="n0",
                provider="local",
                region="us-east5",
                capacity=ResourceVector(cpu=8, memory_gib=64),
                allocatable=ResourceVector(cpu=8, memory_gib=64),
                agent_endpoint="http://agent:8090",
            )
        )

    report = await service.carbon_report(Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),)))

    assert not report.configured
    assert report.regions[0].reason == "stale"
    assert report.regions[0].intensity is None


def test_ordinary_clock_skew_does_not_reject_a_current_reading() -> None:
    """A source that reads "now" is routinely milliseconds ahead of the report.

    Rejecting that would make a correctly configured provider fail about half
    the time, unpredictably -- which is how this was written before a test
    caught it.
    """

    reading = _reading(observed_at=NOW + timedelta(milliseconds=500))

    assert fresh_reading(reading, now=NOW) is reading


def test_an_hour_of_skew_is_still_refused() -> None:
    """The tolerance must stay far below what actually signals bad timestamps."""

    result = fresh_reading(_reading(observed_at=NOW + timedelta(hours=1)), now=NOW)

    assert isinstance(result, CarbonUnavailable)
    assert result.reason is CarbonUnavailableReason.STALE
