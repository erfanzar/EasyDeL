# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Housekeeping the control plane claimed to do and never did.

Three service methods were fully implemented, well tested, and invoked by
nothing outside the test suite:

    collect_artifacts_once          managed artifact storage grew without bound
    mark_superseded_results_once    a replaced result never became collectable
    expire_budget_grants_once       federated grants never returned headroom

Every supporting part worked -- the artifact index, the reverse-reference index,
quota accounting, legal holds, the grace-period race analysis, the mandatory
expiry on a grant. Only the call was missing, and nothing reported the
difference: the tests passed, the feature did not exist in production.

So these tests assert the *wiring*, at the layer that was missing. Exercising
the service directly is what the existing suites already do, and is precisely
what failed to notice.
"""

from __future__ import annotations

import hashlib

import pytest
from test_artifact_collection import _DIGEST, _PAYLOAD, _service, _stored
from test_controller import principal

from eyvan.artifacts.managed import callable_uri
from eyvan.config import EyvanSettings
from eyvan.records import ProjectRecord, Role
from eyvan.server.controller import EyvanController
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService

HOUSEKEEPING = ("mark-superseded", "collect-artifacts", "expire-grants")


def _controller(**settings: object) -> EyvanController:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    return EyvanController(repository, service, settings=EyvanSettings(**settings))  # type: ignore[arg-type]


@pytest.mark.parametrize("phase", HOUSEKEEPING)
def test_each_housekeeping_phase_is_registered_with_the_leader(phase: str) -> None:
    """The whole defect was absence from this tuple."""

    assert phase in {registered.name for registered in _controller()._leader_phases()}


@pytest.mark.parametrize("phase", HOUSEKEEPING)
def test_each_housekeeping_phase_isolates_its_own_failures(phase: str) -> None:
    """Housekeeping must never stop placement.

    An accounting or storage problem is not a reason to stop scheduling work,
    and these run in the same leader cycle as the scheduler.
    """

    registered = next(entry for entry in _controller()._leader_phases() if entry.name == phase)

    # Every phase's failure is contained now, so what this asserts is the other
    # half: housekeeping names no observation it commits against, which is what
    # would let an unrelated failure skip it.
    assert registered.requires == ()
    assert registered.zero_interval_disables is True


def test_superseded_marking_runs_before_collection() -> None:
    """Order decides whether a replaced result waits a whole interval.

    Collection only sees a result once it is marked superseded, so marking after
    collecting would always leave it for the next pass.
    """

    names = [entry.name for entry in _controller()._leader_phases()]

    assert names.index("mark-superseded") < names.index("collect-artifacts")


def test_collection_runs_after_retention() -> None:
    """Both are housekeeping; neither may precede placement in the cycle."""

    names = [entry.name for entry in _controller()._leader_phases()]

    assert names.index("schedule") < names.index("collect-artifacts")


@pytest.mark.asyncio
async def test_a_zero_retention_window_collects_nothing() -> None:
    """Disabling age-based collection must not fall through to deleting.

    ``0`` means "keep everything" everywhere else in the retention settings, and
    a collector that read it as "collect immediately" would be catastrophic.
    """

    controller = _controller(artifact_retention_s=0)

    assert await controller.collect_artifacts_once() == 0


@pytest.mark.asyncio
async def test_collection_remains_dry_run_without_explicit_destructive_opt_in() -> None:
    """A scheduled pass observes candidates but cannot silently enable deletion."""

    controller = _controller(artifact_retention_s=3600)
    async with controller.repository.transaction() as tx:
        await tx.put_project(
            ProjectRecord(
                org_id="acme",
                id="research",
                owner="alice",
                created_by="alice",
            )
        )
    seen: dict[str, object] = {}

    async def _collect(**kwargs: object):
        seen.update(kwargs)

        class _Empty:
            collected_uris = ()
            reclaimed_bytes = 0
            dry_run = False

        return _Empty()

    controller.service.collect_artifacts_once = _collect  # type: ignore[method-assign]

    await controller.collect_artifacts_once()

    assert seen["dry_run"] is True
    assert (seen["org_id"], seen["project_id"]) == ("acme", "research")
    assert seen["grace_period_s"] == 3600.0
    assert seen["principal"] is not None, "an unaudited deletion is an unexplainable loss"


@pytest.mark.asyncio
async def test_the_grant_sweep_reaches_the_service() -> None:
    """A grant's expiry is mandatory, so an unrun sweep means it never applies."""

    controller = _controller()
    called = False

    async def _expire(**_kwargs: object) -> int:
        nonlocal called
        called = True
        return 2

    controller.service.expire_budget_grants_once = _expire  # type: ignore[method-assign]

    assert await controller.expire_budget_grants_once() == 2
    assert called


@pytest.mark.asyncio
async def test_the_superseded_sweep_reaches_the_service() -> None:
    """Without it, nothing a later attempt replaced is ever collectable."""

    controller = _controller()

    async def _mark(**_kwargs: object) -> int:
        return 3

    controller.service.mark_superseded_results_once = _mark  # type: ignore[method-assign]

    assert await controller.mark_superseded_results_once() == 3


@pytest.mark.asyncio
async def test_a_deleted_artifact_is_audited(tmp_path) -> None:
    """Bytes removed with no ledger entry is a loss nobody can explain later.

    Written in the same transaction as the index update, so a deletion and its
    audit record cannot diverge. Before this, collection emitted nothing at all
    -- only a ``logger.warning`` on the hold race.
    """

    service, store = await _service(tmp_path)
    await _stored(service, store, callable_uri(_DIGEST))
    admin = principal(Role.FLEET_ADMIN)

    collection = await service.collect_artifacts_once(
        org_id="acme", project_id="research", principal=admin, dry_run=False, grace_period_s=0.0
    )

    assert collection.collected_uris, "nothing was collected, so the audit assertion would be vacuous"
    async with service.repository.transaction() as tx:
        events = await tx.list_events(org_id="acme", project_id="research")
    collected = [event for event in events if event.action == "artifact.collected"]
    assert len(collected) == len(collection.collected_uris)
    assert collected[0].entity_type == "artifact"
    assert collected[0].previous_state == "deleting"
    assert collected[0].new_state == "deleted"
    assert collected[0].payload["provider_status"] in {"deleted", "absent"}
    assert collected[0].payload["record_generation"] >= 1
    assert collected[0].payload["object_version"]


@pytest.mark.asyncio
async def test_a_dry_run_collection_audits_nothing(tmp_path) -> None:
    """A candidate list is not a deletion, and must not read as one in the ledger."""

    service, store = await _service(tmp_path)
    await _stored(service, store, callable_uri(_DIGEST))

    await service.collect_artifacts_once(
        org_id="acme", project_id="research", principal=principal(Role.FLEET_ADMIN), dry_run=True, grace_period_s=0.0
    )

    async with service.repository.transaction() as tx:
        events = await tx.list_events(org_id="acme", project_id="research")
    assert not [event for event in events if event.action == "artifact.collected"]


@pytest.mark.asyncio
async def test_recovery_discovers_artifact_scopes_without_project_registry_rows(tmp_path) -> None:
    """A stranded upload cannot be invisible merely because its project row is absent."""

    service, store = await _service(tmp_path)
    uri = callable_uri(_DIGEST)
    payload = _PAYLOAD
    reserved = await service.reserve_managed_artifact_upload(
        org_id="orphan-org",
        project_id="orphan-project",
        uri=uri,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    store.put_versioned(uri, payload, org_id="orphan-org", project_id="orphan-project")
    controller = EyvanController(
        service.repository,
        service,
        settings=EyvanSettings(artifact_retention_s=0, artifact_recovery_grace_s=0),
    )

    assert await controller.collect_artifacts_once() == 0
    async with service.repository.transaction() as transaction:
        recovered = await transaction.get_managed_artifact("orphan-org", "orphan-project", uri)
        projects = await transaction.list_projects()
    assert projects == []
    assert recovered is not None
    assert recovered.record_generation == reserved.record_generation
    assert recovered.state.value == "active"


@pytest.mark.asyncio
async def test_explicit_destructive_collection_setting_reaches_the_service() -> None:
    controller = _controller(
        artifact_retention_s=3600,
        artifact_destructive_collection_enabled=True,
    )
    async with controller.repository.transaction() as transaction:
        await transaction.put_project(ProjectRecord(org_id="acme", id="research", owner="alice", created_by="alice"))
    seen: dict[str, object] = {}

    async def collect(**kwargs: object):
        seen.update(kwargs)

        class Empty:
            collected_uris = ()
            reclaimed_bytes = 0
            dry_run = False

        return Empty()

    controller.service.collect_artifacts_once = collect  # type: ignore[method-assign]
    await controller.collect_artifacts_once()
    assert seen["dry_run"] is False
