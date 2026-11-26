# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Warm capacity counts only when it is warm *for the work that will arrive*.

A warm floor exists to remove provisioning latency at 09:00. A cell that is up
but staged with a different image has not removed it -- the first workload still
waits for the pull, and the floor reported "held" the whole time.

**This property already holds, and nothing asserted it.** ``cell_provides_pool_supply``
calls ``cell_matches_pool``, which compares ``cell.runtime == desired.runtime``,
and ``NativeRuntimeConfig`` carries ``vm_image``, ``container``, ``artifacts``,
``init_commands`` and ``env``. So image-aware warmth is emergent from a general
template check rather than built as its own mechanism -- which is a good design
and a fragile guarantee: nothing in the suite says warmth depends on it, so a
future narrowing of ``cell_matches_template`` for some unrelated reason would
silently make a stale-image fleet report a floor it is not holding.

These tests pin it. Adding a second image-awareness mechanism would have been
duplicate machinery for a property the code already has, which is why the
finding here is a test rather than a feature.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from eyvan.capacity.planning import cell_provides_pool_supply
from eyvan.records import (
    CapacityPoolRecord,
    CellState,
    NativeRuntimeArtifact,
    NativeRuntimeConfig,
    ResourceVector,
)

NOW = datetime(2026, 7, 29, 10, 0, tzinfo=UTC)


def _pool(runtime: NativeRuntimeConfig | None = None) -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id="research/v5p",
        fleet_name="research",
        node_type="v5p",
        provider="local",
        region="local",
        capacity=ResourceVector(cpu=8, memory_gib=64),
        allocatable=ResourceVector(cpu=8, memory_gib=64),
        agent_endpoint="http://agent:8090",
        min_cells=1,
        max_cells=4,
        runtime=runtime or NativeRuntimeConfig(vm_image="research:2026-07"),
    )


def _cell(pool: CapacityPoolRecord, *, runtime: NativeRuntimeConfig | None = None, state=CellState.READY):
    cell = pool.make_cell("cell-a", worker_count=1).model_copy(
        update={"state": state, "created_at": NOW - timedelta(hours=1)}
    )
    return cell if runtime is None else cell.model_copy(update={"runtime": runtime})


# --------------------------------------------------------------------------- the property


def test_a_cell_on_the_pools_image_supplies_its_warm_floor() -> None:
    """The case the floor exists for: capacity that is genuinely ready."""

    pool = _pool()

    assert cell_provides_pool_supply(_cell(pool), pool, NOW) is True


def test_a_cell_on_a_different_image_does_not() -> None:
    """The defect a floor without this would hide.

    The cell is up, healthy and idle. It reports as capacity, the floor reports
    as held, and the first workload to arrive still pays the image pull the warm
    floor was configured to remove.
    """

    pool = _pool()
    stale = _cell(pool, runtime=NativeRuntimeConfig(vm_image="research:2026-05"))

    assert cell_provides_pool_supply(stale, pool, NOW) is False


def test_a_cell_missing_a_staged_artifact_does_not() -> None:
    """Warmth is about more than the image.

    A pool that stages a model checkpoint at boot is warm only when the
    checkpoint is there; a cell without it pays the download the floor exists to
    have already paid.
    """

    pool = _pool(
        NativeRuntimeConfig(
            vm_image="research:2026-07",
            artifacts=(NativeRuntimeArtifact(source="gs://weights/base", destination="/opt/weights"),),
        )
    )
    without = _cell(pool, runtime=NativeRuntimeConfig(vm_image="research:2026-07"))

    assert cell_provides_pool_supply(without, pool, NOW) is False


def test_a_cell_with_stale_init_commands_does_not() -> None:
    """The bootstrap is part of what "warm" means.

    A cell that booted before an init command was added has not run it, and
    nothing will run it retroactively.
    """

    pool = _pool(NativeRuntimeConfig(vm_image="research:2026-07", init_commands=("/opt/setup.sh",)))
    without = _cell(pool, runtime=NativeRuntimeConfig(vm_image="research:2026-07"))

    assert cell_provides_pool_supply(without, pool, NOW) is False


def test_the_check_errs_toward_under_counting() -> None:
    """A cell differing only in an environment variable is refused too.

    Stricter than strictly necessary, and the correct direction: under-counting
    launches a replacement cell, while over-counting reports a floor that is not
    held and produces the 09:00 latency the floor was bought to prevent.
    """

    pool = _pool(NativeRuntimeConfig(vm_image="research:2026-07", env={"HF_HOME": "/var/lib/hf"}))
    different = _cell(pool, runtime=NativeRuntimeConfig(vm_image="research:2026-07", env={"HF_HOME": "/tmp/hf"}))

    assert cell_provides_pool_supply(different, pool, NOW) is False


# --------------------------------------------------------------------------- the report


async def test_the_warmth_report_agrees_with_the_planner() -> None:
    """Reporting on a different definition than the one that acts is the worst answer.

    It would say the floor is met while the planner keeps launching -- and an
    operator reading it would delete a warm floor that was working.
    """

    from eyvan.records import Principal, Role, RoleGrant
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    pool = _pool()
    async with repository.transaction() as tx:
        await tx.put_capacity_pool(pool)
        await tx.put_cell(_cell(pool, runtime=NativeRuntimeConfig(vm_image="research:2026-05")))

    report = await service.warmth_report(Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),)), now=NOW)

    assert len(report.pools) == 1
    assert report.pools[0].floor == 1
    # The stale-image cell exists and is READY, and counts for nothing.
    assert report.pools[0].warm_cells == 0
    assert report.unsatisfied
