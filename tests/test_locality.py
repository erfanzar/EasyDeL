# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Named-asset locality: prefer the cell that probably already has the data.

A job pulling a 400 GB dataset spends real minutes doing it, and spends them
again on every cell that has not seen it. ``preferred_cells`` is already the
fourth key of the scheduler's lexicographic score, so the placement mechanism
existed -- what was missing is the index of which cells hold what.

Two design choices are asserted here because both are what make the feature
safe rather than merely present:

**Derived from lease history, not agent reports.** The control plane already
knows which workload held which cell and when, so the inference costs no new
protocol field and no new way for its belief to diverge from a disk it cannot
see.

**A preference, never a requirement.** A stale entry costs one cache miss. It
cannot make a workload unplaceable and it cannot outlive the TTL -- which is
exactly why an inference is acceptable here and would not be if the output
constrained placement.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.records import (
    CommandExecution,
    LeaseRecord,
    LeaseState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.locality import DEFAULT_LOCALITY_TTL_S, build_index, preferred_cells_for

_DATASET = "gs://acme/datasets/pile-v3"
_MODEL = "gs://acme/checkpoints/llama-70b"


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


async def _service() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(repository=repository)


def _spec(*, assets: tuple[str, ...] = (), preferred: tuple[str, ...] = ()) -> WorkloadDefinition:
    return WorkloadDefinition(
        name="train",
        execution=CommandExecution(kind="command", command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
        named_assets=assets,
        preferred_cells=preferred,
    )


def _workload(workload_id: str, *assets: str) -> WorkloadRecord:
    return WorkloadRecord(
        id=workload_id,
        org_id="acme",
        project_id="research",
        spec=_spec(assets=assets),
        created_by="alice",
    )


def _lease(lease_id: str, workload_id: str, cell_id: str, *, age_s: float = 60.0) -> LeaseRecord:
    stamp = utcnow() - timedelta(seconds=age_s)
    return LeaseRecord(
        id=lease_id,
        workload_id=workload_id,
        cell_id=cell_id,
        org_id="acme",
        project_id="research",
        state=LeaseState.RELEASED,
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
        generation=1,
        fencing_token=f"token-{lease_id}",
        expires_at=stamp + timedelta(hours=24),
        created_at=stamp,
        updated_at=stamp,
    )


def test_named_assets_must_not_repeat() -> None:
    with pytest.raises(ValueError, match="named assets"):
        _spec(assets=(_DATASET, _DATASET))


def test_an_asset_maps_to_the_cell_that_ran_it() -> None:
    index = build_index([_workload("wl-1", _DATASET)], [_lease("l-1", "wl-1", "cell-a")], now=utcnow())

    assert index == {_DATASET: ("cell-a",)}


def test_cells_are_ranked_most_recent_first() -> None:
    """The head of the list is the strongest guess, so ordering is the output."""

    index = build_index(
        [_workload("wl-1", _DATASET), _workload("wl-2", _DATASET)],
        [
            _lease("l-1", "wl-1", "cell-old", age_s=3_000),
            _lease("l-2", "wl-2", "cell-new", age_s=60),
        ],
        now=utcnow(),
    )

    assert index[_DATASET] == ("cell-new", "cell-old")


def test_an_entry_older_than_the_ttl_is_dropped() -> None:
    """A guess that cannot expire is a guess that is eventually always wrong."""

    index = build_index(
        [_workload("wl-1", _DATASET)],
        [_lease("l-1", "wl-1", "cell-a", age_s=DEFAULT_LOCALITY_TTL_S + 60)],
        now=utcnow(),
    )

    assert index == {}


def test_a_workload_naming_no_assets_contributes_nothing() -> None:
    index = build_index([_workload("wl-1")], [_lease("l-1", "wl-1", "cell-a")], now=utcnow())

    assert index == {}


def test_a_lease_for_an_unknown_workload_is_ignored() -> None:
    """A pruned workload must not make its old cell look like it holds something."""

    index = build_index([], [_lease("l-1", "wl-gone", "cell-a")], now=utcnow())

    assert index == {}


def test_repeated_runs_on_one_cell_produce_one_entry() -> None:
    index = build_index(
        [_workload("wl-1", _DATASET), _workload("wl-2", _DATASET)],
        [_lease("l-1", "wl-1", "cell-a", age_s=3_000), _lease("l-2", "wl-2", "cell-a", age_s=60)],
        now=utcnow(),
    )

    assert index[_DATASET] == ("cell-a",)


def test_a_cell_holding_more_of_the_wanted_assets_wins() -> None:
    """Otherwise a job wanting a dataset and a checkpoint chases whichever indexed first.

    The names are chosen so hit count is the *only* thing that can decide: the
    one-asset cell is both alphabetically first and top-ranked for the dataset,
    so any ordering that ignores how many assets a cell holds puts it first.
    """

    index = {_DATASET: ("cell-a", "cell-z"), _MODEL: ("cell-z",)}

    assert preferred_cells_for((_DATASET, _MODEL), index) == ("cell-z", "cell-a")


def test_preferences_are_capped() -> None:
    """Past the first few the entries are old enough that the guess is weak."""

    index = {_DATASET: tuple(f"cell-{n}" for n in range(50))}

    assert len(preferred_cells_for((_DATASET,), index, limit=3)) == 3


def test_a_workload_with_no_assets_gets_no_preferences() -> None:
    assert preferred_cells_for((), {_DATASET: ("cell-a",)}) == ()


def test_an_unindexed_asset_yields_no_preferences() -> None:
    """Empty, not an error: nothing has run this dataset yet, which is ordinary."""

    assert preferred_cells_for((_MODEL,), {_DATASET: ("cell-a",)}) == ()


@pytest.mark.asyncio
async def test_submission_fills_preferred_cells_from_history() -> None:
    service = await _service()
    async with service.repository.transaction() as tx:
        await tx.put_workload(_workload("wl-earlier", _DATASET))
        await tx.put_lease(_lease("l-1", "wl-earlier", "cell-warm"))

    workload, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_spec(assets=(_DATASET,)),
        idempotency_key="submit-1",
    )

    assert workload.spec.preferred_cells == ("cell-warm",)


@pytest.mark.asyncio
async def test_an_explicit_pin_is_never_overwritten_by_a_guess() -> None:
    """A pin is a statement the submitter made; a cache guess cannot argue with it."""

    service = await _service()
    async with service.repository.transaction() as tx:
        await tx.put_workload(_workload("wl-earlier", _DATASET))
        await tx.put_lease(_lease("l-1", "wl-earlier", "cell-warm"))

    workload, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_spec(assets=(_DATASET,), preferred=("cell-pinned",)),
        idempotency_key="submit-1",
    )

    assert workload.spec.preferred_cells == ("cell-pinned",)


@pytest.mark.asyncio
async def test_a_replay_is_not_a_conflict_when_history_moved() -> None:
    """The fingerprint covers the spec as submitted, not the resolved preference.

    Folding a history-derived field into the fingerprint would make the identical
    request hash differently minute to minute, so every retry of a slow client
    would 409 rather than replay.
    """

    service = await _service()
    first, replayed_first = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_spec(assets=(_DATASET,)),
        idempotency_key="submit-1",
    )
    assert replayed_first is False
    async with service.repository.transaction() as tx:
        await tx.put_workload(_workload("wl-other", _DATASET))
        await tx.put_lease(_lease("l-1", "wl-other", "cell-warm"))

    second, replayed = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_spec(assets=(_DATASET,)),
        idempotency_key="submit-1",
    )

    assert replayed is True
    assert second.id == first.id


@pytest.mark.asyncio
async def test_a_workload_naming_no_assets_is_left_alone() -> None:
    service = await _service()
    async with service.repository.transaction() as tx:
        await tx.put_workload(_workload("wl-earlier", _DATASET))
        await tx.put_lease(_lease("l-1", "wl-earlier", "cell-warm"))

    workload, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_spec(),
        idempotency_key="submit-1",
    )

    assert workload.spec.preferred_cells == ()
