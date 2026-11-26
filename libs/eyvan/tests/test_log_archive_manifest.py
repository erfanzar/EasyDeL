# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""DR-0002's terminal archive manifest: a separate object, committed last.

The archive index used to be one array line inside rank zero's own blob. Rank
zero is written first, so that index could only ever name what the writer
*hoped* to upload -- and if rank zero itself failed, every rank that did land
became undiscoverable. It also carried observed ranks only, with no topology
digest, no per-rank status and no per-object digest, so "this rank printed
nothing", "this rank's host was unreachable" and "this rank's object was lost"
were the same absence.

These tests drive the real settlement path and read the bytes that reach the
managed artifact store.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest

from eyvan.artifacts.managed import (
    InvalidManagedArtifactURI,
    LocalManagedArtifactStore,
    log_manifest_uri,
    log_uri,
    parse_managed_uri,
    storage_scope,
)
from eyvan.records import ManagedArtifactState, WorkloadState
from eyvan.server.controller.settlement import (
    _LOG_ARCHIVE_LINE_LIMIT,
    _LOG_ARCHIVE_MAX_RANK_OBJECTS,
    LOG_ARCHIVE_MANIFEST_SCHEMA_V2,
)
from eyvan.worker.runtime.log_records import LOG_ARCHIVE_MANIFEST_KEY, LOG_ARCHIVE_MANIFEST_VERSION

ORG = "acme"
PROJECT = "research"


# --------------------------------------------------------------------------- the URI


def test_the_manifest_is_a_first_class_managed_artifact() -> None:
    """A separate URI is what lets the manifest be committed after every blob.

    An index embedded in rank zero cannot be: rank zero has to exist before the
    ranks it describes have been attempted.
    """

    uri = log_manifest_uri("wl-1", 3)

    assert uri == "eyvan://logs/wl-1-a3-manifest"
    assert parse_managed_uri(uri).kind.value == "logs"
    assert uri not in {log_uri("wl-1", 3, rank) for rank in range(64)}


def test_each_attempt_gets_its_own_manifest() -> None:
    """A retry's manifest must not claim the predecessor's objects."""

    assert log_manifest_uri("wl-1", 1) != log_manifest_uri("wl-1", 2)


@pytest.mark.parametrize("attempt", [0, -1])
def test_a_manifest_uri_requires_a_real_attempt(attempt: int) -> None:
    """Attempt zero would alias whatever the first attempt wrote."""

    with pytest.raises(InvalidManagedArtifactURI):
        log_manifest_uri("wl-1", attempt)


# --------------------------------------------------------------------------- harness


class _RecordingStore:
    """A local store that records every write and can fail chosen URIs."""

    def __init__(self, backing: LocalManagedArtifactStore) -> None:
        self.backing = backing
        self.put_uris: list[str] = []
        self.failures: set[str] = set()
        self.budget: int | None = None
        # Rank objects are archived concurrently, so the budget check and the
        # record of the attempt must be one step or two writers both pass it.
        self._lock = threading.Lock()

    def put_versioned(self, uri, payload, **scope):
        with self._lock:
            refused = uri in self.failures or (self.budget is not None and len(self.put_uris) >= self.budget)
            self.put_uris.append(uri)
        if refused:
            raise OSError("managed artifact storage is unreachable")
        return self.backing.put_versioned(uri, payload, **scope)

    def get_versioned(self, uri, **scope):
        return self.backing.get_versioned(uri, **scope)

    def put(self, uri, payload, **scope):
        return self.put_versioned(uri, payload, **scope).metadata.uri

    def get(self, uri, **scope):
        return self.get_versioned(uri, **scope).payload

    def exists(self, uri, **scope) -> bool:
        return self.backing.exists(uri, **scope)


def _artifact_access(root: Path, store: object | None = None):
    from eyvan.artifacts.managed import ArtifactTokenSigner
    from eyvan.artifacts.transfers import ManagedArtifactAccess

    access = ManagedArtifactAccess(
        store=LocalManagedArtifactStore(root),
        signer=ArtifactTokenSigner(b"k" * 32, default_ttl_seconds=3600),
        public_url="https://eyvan.test",
        token_ttl_seconds=3600,
    )
    if store is not None:
        object.__setattr__(access, "store", store)
    return access


async def _controller(tmp_path: Path, *, width: int = 1, attempt: int = 1):
    """A single-cell workload whose declared rank width is ``width``."""

    from eyvan.config import EyvanSettings
    from eyvan.records import (
        CellRecord,
        CellState,
        CommandExecution,
        LeaseRecord,
        LeaseState,
        ResourceGang,
        ResourceVector,
        WorkloadDefinition,
        WorkloadRecord,
        utcnow,
    )
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, settings=EyvanSettings())
    async with repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=8),
                allocatable=ResourceVector(cpu=8),
                endpoint="http://agent:8090",
                runtime_process_count=width,
                generation=7,
            )
        )
        await transaction.put_lease(
            LeaseRecord(
                id="lease-1",
                workload_id="wl-1",
                cell_id="cell-a",
                org_id=ORG,
                project_id=PROJECT,
                state=LeaseState.ACTIVE,
                resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
                generation=1,
                fencing_token="token-1",
                expires_at=utcnow() + timedelta(hours=24),
            )
        )
        await transaction.put_workload(
            WorkloadRecord(
                id="wl-1",
                org_id=ORG,
                project_id=PROJECT,
                spec=WorkloadDefinition(name="t", execution=CommandExecution(command="echo hi")),
                state=WorkloadState.RUNNING,
                attempt=attempt,
                lease_id="lease-1",
                remote_id="remote-1",
                created_by="alice",
            )
        )
    return controller


def _attach(controller, access) -> None:
    controller.artifact_access = access
    controller.service.artifact_store = access.store


def _scripted_page(**page):
    async def read(_workload_id, *, after="", limit=200, live_only=False):
        del after, limit, live_only
        return {"records": [], "truncated_ranks": [], "unavailable": False, **page}

    return read


async def _workload(controller):
    async with controller.repository.transaction() as transaction:
        return await transaction.get_workload("wl-1")


async def _manifest(controller, *, attempt: int = 1) -> dict:
    payload = await controller.service.read_managed_artifact(
        org_id=ORG,
        project_id=PROJECT,
        uri=log_manifest_uri("wl-1", attempt),
    )
    return json.loads(payload)


async def _artifact_row(controller, uri: str):
    async with controller.repository.transaction() as transaction:
        return await transaction.get_managed_artifact(ORG, PROJECT, uri)


async def _committed(controller, uri: str) -> bool:
    """Return whether ``uri`` is a committed object rather than a live reservation.

    A physical write that failed leaves a ``pending_upload`` row behind, which is
    a reservation and not an archive: nothing can read it.
    """

    row = await _artifact_row(controller, uri)
    return row is not None and row.state is ManagedArtifactState.ACTIVE


def _object_path(tmp_path: Path, uri: str) -> Path:
    """Return the on-disk file the local store keeps one managed object in."""

    return (
        tmp_path
        / "artifacts"
        / "scopes"
        / storage_scope(ORG, PROJECT)
        / "logs"
        / f"{parse_managed_uri(uri).object_id}.jsonl"
    )


def _by_rank(manifest: dict) -> dict[int, dict]:
    return {entry["rank"]: entry for entry in manifest["ranks"]}


@pytest.fixture
def manifest_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn on the DR-0002 stage-2 writer, which is off by default."""

    monkeypatch.setenv("EYVAN_LOG_ARCHIVE_MANIFEST_V2_ENABLED", "1")


# --------------------------------------------------------------------------- the default


async def test_the_default_replica_writes_no_manifest_object(tmp_path: Path, monkeypatch) -> None:
    """DR-0002 rolls the writer out before the emission switch.

    A replica that started emitting manifests before every serving and rollback
    image could read them would be producing a record only it understands.
    """

    monkeypatch.delenv("EYVAN_LOG_ARCHIVE_MANIFEST_V2_ENABLED", raising=False)
    controller = await _controller(tmp_path)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    controller.workload_log_page = _scripted_page(records=[{"message": "hi", "rank": 0}])

    await controller._archive_terminal_logs(await _workload(controller))

    assert await _artifact_row(controller, log_manifest_uri("wl-1", 1)) is None
    assert await _committed(controller, log_uri("wl-1", 1, 0))


async def test_the_embedded_v1_index_is_byte_identical_with_and_without_the_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One version tag never covers two payload schemas.

    A rolling reader that knows only the embedded index must see exactly the
    bytes it saw before the manifest shipped; the new document lives at a URI it
    never looks at.
    """

    page = {
        "records": [{"message": "hi", "rank": 0}, {"message": "there", "rank": 2}],
        "truncated_ranks": [1],
        "unavailable": False,
    }
    stored: list[bytes] = []
    for enabled, root in (("0", "off"), ("1", "on")):
        monkeypatch.setenv("EYVAN_LOG_ARCHIVE_MANIFEST_V2_ENABLED", enabled)
        controller = await _controller(tmp_path, width=3)
        access = _artifact_access(tmp_path / root)
        _attach(controller, access)
        controller.workload_log_page = _scripted_page(**page)
        await controller._archive_terminal_logs(await _workload(controller))
        stored.append(access.store.get(log_uri("wl-1", 1, 0), org_id=ORG, project_id=PROJECT))

    assert stored[0] == stored[1]
    index = json.loads(stored[1].splitlines()[0])
    assert index[0] == LOG_ARCHIVE_MANIFEST_KEY
    assert index[1]["version"] == LOG_ARCHIVE_MANIFEST_VERSION
    assert index[1]["ranks"] == [0, 1, 2]


# --------------------------------------------------------------------------- statuses


async def test_quiet_unreachable_truncated_and_failed_ranks_are_each_named(
    tmp_path: Path,
    manifest_v2: None,
) -> None:
    """The four fates the embedded index collapses into one absence.

    Rank 3 printed nothing, rank 2's host was unreachable, rank 1 kept only its
    tail, rank 4's object never landed. Without per-rank status all four look
    identical to a reader, and only one of them means the run was fine.
    """

    controller = await _controller(tmp_path, width=5)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    # Rank four's URI is already claimed by a row nothing can activate -- the
    # state a quarantined or half-indexed object leaves behind. The URI is
    # immutable, so no retry recovers it and the archive is final without it.
    await controller.service.index_managed_artifact(
        org_id=ORG,
        project_id=PROJECT,
        uri=log_uri("wl-1", 1, 4),
        size_bytes=0,
    )
    controller.workload_log_page = _scripted_page(
        records=[
            {"message": "rank zero", "rank": 0},
            *({"message": f"rank one line {line}", "rank": 1} for line in range(_LOG_ARCHIVE_LINE_LIMIT + 1)),
            {"message": "rank four", "rank": 4},
        ],
        truncated_ranks=[2],
    )

    await controller._archive_terminal_logs(await _workload(controller))
    manifest = await _manifest(controller)
    ranks = _by_rank(manifest)

    assert manifest["schema"] == LOG_ARCHIVE_MANIFEST_SCHEMA_V2
    assert sorted(ranks) == [0, 1, 2, 3, 4]
    assert ranks[0]["status"] == "archived"
    assert ranks[1]["status"] == "truncated"
    assert ranks[2]["status"] == "unreachable"
    assert ranks[3]["status"] == "quiet"
    assert ranks[4]["status"] == "failed"
    # A failed object has no bytes to describe, and saying otherwise would let a
    # reader verify a digest that names nothing.
    assert ranks[4]["sha256"] is None and ranks[4]["size_bytes"] is None
    assert manifest["complete"] is False
    assert manifest["unavailable"] is False


async def test_every_expected_rank_comes_from_the_declared_width(tmp_path: Path, manifest_v2: None) -> None:
    """A set derived from observation can only describe the ranks that were fine.

    Only rank 0 answered here. The other three exist, and the archive says so.
    """

    controller = await _controller(tmp_path, width=4)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    controller.workload_log_page = _scripted_page(records=[{"message": "only me", "rank": 0}])

    await controller._archive_terminal_logs(await _workload(controller))
    manifest = await _manifest(controller)

    assert sorted(_by_rank(manifest)) == [0, 1, 2, 3]
    assert [entry["status"] for entry in manifest["ranks"]] == ["archived", "quiet", "quiet", "quiet"]
    assert manifest["topology"]["width"] == 4
    assert manifest["topology"]["source"] == "leases"
    assert manifest["topology"]["digest"].startswith("sha256:")
    assert manifest["topology"]["cells"] == [{"cell": "cell-a", "generation": 7, "width": 4}]
    # Every named rank has a real object, so a reader can prove the silence.
    for entry in manifest["ranks"]:
        assert entry["uri"] is not None
        assert await _artifact_row(controller, entry["uri"]) is not None
    assert manifest["complete"] is True


async def test_a_read_that_reached_nothing_marks_every_expected_rank_unreachable(
    tmp_path: Path,
    manifest_v2: None,
) -> None:
    """ "The whole slice was unreachable" must not archive as "nobody printed"."""

    controller = await _controller(tmp_path, width=3)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    controller.workload_log_page = _scripted_page(unavailable=True, records=[])

    await controller._archive_terminal_logs(await _workload(controller))
    manifest = await _manifest(controller)

    assert [entry["status"] for entry in manifest["ranks"]] == ["unreachable"] * 3
    assert manifest["unavailable"] is True
    assert manifest["complete"] is False


async def test_an_unknown_topology_is_never_reported_complete(tmp_path: Path, manifest_v2: None) -> None:
    """Without the declared width there is no set of expected ranks to be complete over."""

    controller = await _controller(tmp_path)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    controller.workload_log_page = _scripted_page(records=[{"message": "hi", "rank": 0}])
    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-1")
        await transaction.put_workload(workload.model_copy(update={"lease_id": None}))

    await controller._archive_terminal_logs(await _workload(controller))
    manifest = await _manifest(controller)

    assert manifest["topology"] == {"source": "unknown", "width": None, "digest": None, "cells": []}
    assert manifest["complete"] is False
    assert _by_rank(manifest)[0]["status"] == "archived"


# --------------------------------------------------------------------------- object identity


async def test_the_manifest_digest_and_size_match_the_objects_it_names(
    tmp_path: Path,
    manifest_v2: None,
) -> None:
    """A digest nobody can check against the bytes is decoration."""

    controller = await _controller(tmp_path, width=3)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    controller.workload_log_page = _scripted_page(
        records=[{"message": "zero", "rank": 0}, {"message": "two", "rank": 2}],
    )

    await controller._archive_terminal_logs(await _workload(controller))
    manifest = await _manifest(controller)

    checked = 0
    for entry in manifest["ranks"]:
        payload = await controller.service.read_managed_artifact(org_id=ORG, project_id=PROJECT, uri=entry["uri"])
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert len(payload) == entry["size_bytes"]
        checked += 1
    assert checked == 3


async def test_the_manifest_is_committed_after_every_rank_object(tmp_path: Path, manifest_v2: None) -> None:
    """The defect this object exists for.

    Rank zero is written first and cannot know whether rank two will land. The
    manifest is written last, so what it claims is what exists.
    """

    controller = await _controller(tmp_path, width=3)
    store = _RecordingStore(LocalManagedArtifactStore(tmp_path / "artifacts"))
    _attach(controller, _artifact_access(tmp_path / "artifacts", store))
    controller.workload_log_page = _scripted_page(records=[{"message": "zero", "rank": 0}])

    await controller._archive_terminal_logs(await _workload(controller))

    assert store.put_uris == [
        log_uri("wl-1", 1, 0),
        log_uri("wl-1", 1, 1),
        log_uri("wl-1", 1, 2),
        log_manifest_uri("wl-1", 1),
    ]


async def test_a_missing_middle_rank_still_lets_later_ranks_be_read(tmp_path: Path, manifest_v2: None) -> None:
    """A gap is a named gap, not the end of the archive.

    A contiguous scan that stops at the first miss turns an unreachable host
    into a complete-looking smaller archive.
    """

    controller = await _controller(tmp_path, width=3)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    controller.workload_log_page = _scripted_page(
        records=[
            {"message": "zero", "rank": 0},
            {"message": "one", "rank": 1},
            {"message": "two", "rank": 2},
        ],
    )
    await controller._archive_terminal_logs(await _workload(controller))
    manifest = await _manifest(controller)

    # Rank one's object is destroyed underneath the control plane.
    os.unlink(_object_path(tmp_path, log_uri("wl-1", 1, 1)))

    # The manifest still names rank two, with the digest that proves it.
    entry = _by_rank(manifest)[2]
    payload = await controller.service.read_managed_artifact(org_id=ORG, project_id=PROJECT, uri=entry["uri"])
    assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
    assert json.loads(payload.splitlines()[0])["message"] == "two"

    # And the shipping reader keeps rank two's lines while naming rank one.
    replayed = await controller._archived_log_page(await _workload(controller))
    assert [(record.rank, record.message) for record in replayed.records] == [(0, "zero"), (2, "two")]
    assert replayed.truncated_ranks == (1,)
    assert replayed.unavailable is False


async def test_a_tampered_rank_object_is_detected(tmp_path: Path, manifest_v2: None) -> None:
    """Silently serving rewritten bytes as an attempt's output is the worst case."""

    controller = await _controller(tmp_path, width=2)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    controller.workload_log_page = _scripted_page(
        records=[{"message": "zero", "rank": 0}, {"message": "one", "rank": 1}],
    )
    await controller._archive_terminal_logs(await _workload(controller))
    manifest = await _manifest(controller)
    entry = _by_rank(manifest)[1]

    target = _object_path(tmp_path, log_uri("wl-1", 1, 1))
    forged = json.dumps({"message": "nothing went wrong", "rank": 1}, sort_keys=True).encode()
    os.unlink(target)
    target.write_bytes(forged)

    # The manifest's own record contradicts the bytes now on disk, so a reader
    # holding only the manifest can already tell.
    assert hashlib.sha256(forged).hexdigest() != entry["sha256"]
    assert len(forged) != entry["size_bytes"]

    # And the control plane refuses to serve them rather than replaying a forgery
    # as the attempt's output.
    from eyvan.server.service.errors import ConflictError

    with pytest.raises(ConflictError):
        await controller.service.read_managed_artifact(org_id=ORG, project_id=PROJECT, uri=entry["uri"])
    row = await _artifact_row(controller, entry["uri"])
    assert row.state is ManagedArtifactState.QUARANTINED
    # The untouched rank stays readable: one forged object is not a lost archive.
    intact = await controller.service.read_managed_artifact(
        org_id=ORG, project_id=PROJECT, uri=_by_rank(manifest)[0]["uri"]
    )
    assert hashlib.sha256(intact).hexdigest() == _by_rank(manifest)[0]["sha256"]


# --------------------------------------------------------------------------- crash and retry


@pytest.mark.parametrize("committed", [0, 1, 2, 3])
async def test_a_crash_at_any_blob_boundary_leaves_a_resumable_archive(
    tmp_path: Path,
    manifest_v2: None,
    committed: int,
) -> None:
    """An interrupted archive never publishes a manifest it cannot honour.

    Whatever landed stays landed and immutable; the next settlement of the same
    attempt reuses it and finishes the job.
    """

    backing = LocalManagedArtifactStore(tmp_path / "artifacts")
    crashing = _RecordingStore(backing)
    crashing.budget = committed
    controller = await _controller(tmp_path, width=3)
    _attach(controller, _artifact_access(tmp_path / "artifacts", crashing))
    page = _scripted_page(records=[{"message": "zero", "rank": 0}, {"message": "two", "rank": 2}])
    controller.workload_log_page = page

    await controller._archive_terminal_logs(await _workload(controller))

    # No committed manifest, because not every blob's fate is known.
    assert not await _committed(controller, log_manifest_uri("wl-1", 1))
    landed = {
        rank: row
        for rank in range(3)
        if (row := await _artifact_row(controller, log_uri("wl-1", 1, rank))) is not None
        and row.state is ManagedArtifactState.ACTIVE
    }
    assert len(landed) == committed

    # Resume on a healthy store.
    resumed = _RecordingStore(backing)
    _attach(controller, _artifact_access(tmp_path / "artifacts", resumed))
    await controller._archive_terminal_logs(await _workload(controller))

    manifest = await _manifest(controller)
    assert [entry["status"] for entry in manifest["ranks"]] == ["archived", "quiet", "archived"]
    assert manifest["complete"] is True
    # Blobs that already existed were reused, not rewritten; each missing one was
    # written exactly once, in any order (they are archived concurrently); and
    # the manifest was written last, after every blob's fate was known.
    rank_puts = resumed.put_uris[:-1]
    assert sorted(rank_puts) == sorted(log_uri("wl-1", 1, rank) for rank in range(3) if rank not in landed)
    assert len(rank_puts) == len(set(rank_puts))
    assert resumed.put_uris[-1] == log_manifest_uri("wl-1", 1)
    for rank, row in landed.items():
        current = await _artifact_row(controller, log_uri("wl-1", 1, rank))
        assert current.record_generation == row.record_generation
        assert current.sha256 == row.sha256
        assert _by_rank(manifest)[rank]["sha256"] == row.sha256


async def test_a_crash_at_the_manifest_boundary_leaves_a_resumable_archive(
    tmp_path: Path,
    manifest_v2: None,
) -> None:
    """Every blob is committed and the index is not. Retrying must finish, not conflict."""

    backing = LocalManagedArtifactStore(tmp_path / "artifacts")
    crashing = _RecordingStore(backing)
    crashing.failures.add(log_manifest_uri("wl-1", 1))
    controller = await _controller(tmp_path, width=2)
    _attach(controller, _artifact_access(tmp_path / "artifacts", crashing))
    controller.workload_log_page = _scripted_page(records=[{"message": "zero", "rank": 0}])

    await controller._archive_terminal_logs(await _workload(controller))
    assert not await _committed(controller, log_manifest_uri("wl-1", 1))
    blobs = {rank: await _artifact_row(controller, log_uri("wl-1", 1, rank)) for rank in (0, 1)}
    assert all(row.state is ManagedArtifactState.ACTIVE for row in blobs.values())

    resumed = _RecordingStore(backing)
    _attach(controller, _artifact_access(tmp_path / "artifacts", resumed))
    await controller._archive_terminal_logs(await _workload(controller))

    manifest = await _manifest(controller)
    assert resumed.put_uris == [log_manifest_uri("wl-1", 1)]
    assert {entry["rank"]: entry["sha256"] for entry in manifest["ranks"]} == {
        rank: row.sha256 for rank, row in blobs.items()
    }
    assert manifest["complete"] is True


async def test_an_immutable_retry_reuses_active_blobs_rather_than_conflicting(
    tmp_path: Path,
    manifest_v2: None,
) -> None:
    """Managed URIs are immutable, so evolved bytes cannot overwrite the record.

    Re-uploading them conflicts and quarantines the row -- losing the archive
    that already existed in order to store one that is no more true.
    """

    backing = LocalManagedArtifactStore(tmp_path / "artifacts")
    first = _RecordingStore(backing)
    controller = await _controller(tmp_path, width=2)
    _attach(controller, _artifact_access(tmp_path / "artifacts", first))
    controller.workload_log_page = _scripted_page(records=[{"message": "the original tail", "rank": 0}])
    await controller._archive_terminal_logs(await _workload(controller))
    original = {rank: await _artifact_row(controller, log_uri("wl-1", 1, rank)) for rank in (0, 1)}

    # A second settlement of the same attempt reads different lines.
    second = _RecordingStore(backing)
    _attach(controller, _artifact_access(tmp_path / "artifacts", second))
    controller.workload_log_page = _scripted_page(records=[{"message": "a different tail", "rank": 1}])
    await controller._archive_terminal_logs(await _workload(controller))

    assert second.put_uris == []
    for rank, row in original.items():
        current = await _artifact_row(controller, log_uri("wl-1", 1, rank))
        assert current.state is ManagedArtifactState.ACTIVE
        assert current.record_generation == row.record_generation
        assert current.sha256 == row.sha256
    stored = backing.get(log_uri("wl-1", 1, 0), org_id=ORG, project_id=PROJECT)
    assert b"the original tail" in stored
    assert b"a different tail" not in stored


async def test_a_lost_manifest_write_is_cleared_by_the_lifecycle_sweep_and_then_committed(
    tmp_path: Path,
    manifest_v2: None,
) -> None:
    """The one case an immutable URI cannot resolve inside a single settlement.

    A physical write that never landed still leaves a reservation holding the
    bytes it meant to store. A later settlement whose live read produced
    different bytes cannot take that reservation over -- that is what immutable
    means -- so it reports the archive as it stands and leaves the reservation to
    the scoped-lifecycle recovery DR-0001 already ships. Once the sweep releases
    it, the manifest commits and describes the bytes that actually exist.
    """

    backing = LocalManagedArtifactStore(tmp_path / "artifacts")
    crashing = _RecordingStore(backing)
    crashing.failures.add(log_manifest_uri("wl-1", 1))
    controller = await _controller(tmp_path, width=2)
    _attach(controller, _artifact_access(tmp_path / "artifacts", crashing))
    controller.workload_log_page = _scripted_page(records=[{"message": "the original tail", "rank": 0}])
    await controller._archive_terminal_logs(await _workload(controller))

    # A second settlement reads different lines. The blobs are reused, and the
    # stale manifest reservation still refuses evolved bytes.
    healthy = _RecordingStore(backing)
    _attach(controller, _artifact_access(tmp_path / "artifacts", healthy))
    controller.workload_log_page = _scripted_page(records=[{"message": "a different tail", "rank": 1}])
    await controller._archive_terminal_logs(await _workload(controller))
    # Nothing moved: the blobs were reused, and the stale reservation refused the
    # evolved manifest before a single byte was written.
    assert healthy.put_uris == []
    assert not await _committed(controller, log_manifest_uri("wl-1", 1))

    recovery = await controller.service.reconcile_artifact_lifecycle_once(
        org_id=ORG,
        project_id=PROJECT,
        recovery_grace_s=0.0,
    )
    assert recovery["released"] == 1

    await controller._archive_terminal_logs(await _workload(controller))
    manifest = await _manifest(controller)

    for entry in manifest["ranks"]:
        payload = await controller.service.read_managed_artifact(org_id=ORG, project_id=PROJECT, uri=entry["uri"])
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert len(payload) == entry["size_bytes"]
    # Both objects still hold the first settlement's bytes, and the manifest
    # describes those rather than the second read's rank-one line.
    assert b"the original tail" in backing.get(log_uri("wl-1", 1, 0), org_id=ORG, project_id=PROJECT)
    assert _by_rank(manifest)[0]["status"] == "archived"
    assert _by_rank(manifest)[1]["status"] == "quiet"
    assert "committed different bytes" in _by_rank(manifest)[1]["detail"]


# --------------------------------------------------------------------------- bounds


async def test_a_wide_attempt_bounds_its_objects_and_says_how_many_it_omitted(
    tmp_path: Path,
    manifest_v2: None,
) -> None:
    """One object per expected rank is unbounded I/O inside the release path.

    A release that is slow enough to matter strands an accelerator, so the
    ceiling is real -- and the manifest names every expected rank anyway, so an
    omitted object is a recorded fact rather than a silent gap.
    """

    width = _LOG_ARCHIVE_MAX_RANK_OBJECTS + 16
    controller = await _controller(tmp_path, width=width)
    store = _RecordingStore(LocalManagedArtifactStore(tmp_path / "artifacts"))
    _attach(controller, _artifact_access(tmp_path / "artifacts", store))
    controller.workload_log_page = _scripted_page(records=[{"message": "zero", "rank": 0}])

    await controller._archive_terminal_logs(await _workload(controller))
    manifest = await _manifest(controller)

    assert len(store.put_uris) == _LOG_ARCHIVE_MAX_RANK_OBJECTS + 1  # + the manifest
    assert store.put_uris[-1] == log_manifest_uri("wl-1", 1)
    assert len(manifest["ranks"]) == width
    assert manifest["objects_omitted"] == 16
    assert manifest["complete"] is False
    omitted = [entry for entry in manifest["ranks"] if entry["uri"] is None]
    assert len(omitted) == 16
    assert all(entry["status"] == "quiet" for entry in omitted)


# --------------------------------------------------------------------------- capacity


async def test_capacity_is_settled_when_every_archive_write_fails(tmp_path: Path, manifest_v2: None) -> None:
    """Settlement releases the lease. An unwritable archive must not hold one open."""

    controller = await _controller(tmp_path, width=2)
    store = _RecordingStore(LocalManagedArtifactStore(tmp_path / "artifacts"))
    store.budget = 0
    _attach(controller, _artifact_access(tmp_path / "artifacts", store))
    controller.workload_log_page = _scripted_page(records=[{"message": "zero", "rank": 0}])

    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-1")
        lease = await transaction.get_lease("lease-1")
    await controller._release_terminal(workload, lease, WorkloadState.SUCCEEDED, "workload.succeeded")

    async with controller.repository.transaction() as transaction:
        settled = await transaction.get_workload("wl-1")
        released = await transaction.get_lease("lease-1")
    assert settled.state is WorkloadState.SUCCEEDED
    assert released.settled_at is not None
    assert not await _committed(controller, log_manifest_uri("wl-1", 1))


async def test_capacity_is_settled_when_the_archive_raises_outright(tmp_path: Path, manifest_v2: None) -> None:
    """The guard is outermost, not merely around each upload.

    A malformed internal page makes the archive raise before any object is
    attempted; a finished job must still be recorded as finished.
    """

    controller = await _controller(tmp_path, width=2)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))

    async def broken(_workload_id, *, after="", limit=200, live_only=False):
        del after, limit, live_only
        return ["this is not a page document"]

    controller.workload_log_page = broken

    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-1")
        lease = await transaction.get_lease("lease-1")
    await controller._release_terminal(workload, lease, WorkloadState.SUCCEEDED, "workload.succeeded")

    async with controller.repository.transaction() as transaction:
        settled = await transaction.get_workload("wl-1")
        released = await transaction.get_lease("lease-1")
    assert settled.state is WorkloadState.SUCCEEDED
    assert released.settled_at is not None


async def test_the_release_path_commits_the_manifest(tmp_path: Path, manifest_v2: None) -> None:
    """Archival has to be wired into release, not merely callable."""

    controller = await _controller(tmp_path, width=2)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    controller.workload_log_page = _scripted_page(records=[{"message": "final line", "rank": 1}])

    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-1")
        lease = await transaction.get_lease("lease-1")
    await controller._release_terminal(workload, lease, WorkloadState.SUCCEEDED, "workload.succeeded")

    manifest = await _manifest(controller)
    assert [entry["status"] for entry in manifest["ranks"]] == ["quiet", "archived"]
    assert manifest["workload"] == "wl-1"
    assert manifest["attempt"] == 1
    async with controller.repository.transaction() as transaction:
        assert (await transaction.get_workload("wl-1")).state is WorkloadState.SUCCEEDED


async def test_capacity_is_settled_when_only_the_manifest_cannot_be_committed(
    tmp_path: Path,
    manifest_v2: None,
) -> None:
    """An archive that is missing its index is still an archive, and the lease still goes."""

    controller = await _controller(tmp_path, width=2)
    store = _RecordingStore(LocalManagedArtifactStore(tmp_path / "artifacts"))
    store.failures.add(log_manifest_uri("wl-1", 1))
    _attach(controller, _artifact_access(tmp_path / "artifacts", store))
    controller.workload_log_page = _scripted_page(records=[{"message": "zero", "rank": 0}])

    async with controller.repository.transaction() as transaction:
        workload = await transaction.get_workload("wl-1")
        lease = await transaction.get_lease("lease-1")
    await controller._release_terminal(workload, lease, WorkloadState.SUCCEEDED, "workload.succeeded")

    async with controller.repository.transaction() as transaction:
        settled = await transaction.get_workload("wl-1")
        released = await transaction.get_lease("lease-1")
    assert settled.state is WorkloadState.SUCCEEDED
    assert released.settled_at is not None
    assert not await _committed(controller, log_manifest_uri("wl-1", 1))
    assert await _committed(controller, log_uri("wl-1", 1, 0))


# --------------------------------------------------------------------------- rolling upgrade


async def test_the_shipping_reader_replays_an_archive_written_with_the_manifest(
    tmp_path: Path,
    manifest_v2: None,
) -> None:
    """The writer ships before the reader, so the reader must be unaffected by it.

    A replica running the new writer and the old archive reader has to serve the
    same page it served before, including the silent ranks the manifest adds.
    """

    controller = await _controller(tmp_path, width=4)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    controller.workload_log_page = _scripted_page(
        records=[{"message": "zero", "stream": "stdout", "rank": 0}, {"message": "two", "stream": "stderr", "rank": 2}],
        truncated_ranks=[3],
    )

    await controller._archive_terminal_logs(await _workload(controller))
    replayed = await controller._archived_log_page(await _workload(controller))

    assert [(record.rank, record.message, record.stream) for record in replayed.records] == [
        (0, "zero", "stdout"),
        (2, "two", "stderr"),
    ]
    assert replayed.truncated_ranks == (3,)
    assert replayed.unavailable is False
    # And the manifest names rank one, which the reader had no way to know was
    # expected at all.
    manifest = await _manifest(controller)
    assert _by_rank(manifest)[1]["status"] == "quiet"
    assert _by_rank(manifest)[3]["status"] == "unreachable"


async def test_a_gang_topology_spans_every_cell_in_the_declared_rank_order(tmp_path: Path, manifest_v2: None) -> None:
    """Global ranks are cell offsets plus local ranks, and the manifest records both.

    A width taken from one cell, or from whichever cell answered, would renumber
    the second cell's ranks the moment the first went quiet.
    """

    from datetime import timedelta

    from eyvan.records import (
        CellRecord,
        CellState,
        LeaseRecord,
        LeaseState,
        ResourceGang,
        ResourceVector,
        utcnow,
    )

    controller = await _controller(tmp_path, width=2)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    async with controller.repository.transaction() as transaction:
        await transaction.put_cell(
            CellRecord(
                id="cell-b",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(cpu=8),
                allocatable=ResourceVector(cpu=8),
                endpoint="http://agent:8090",
                runtime_process_count=3,
                generation=2,
            )
        )
        await transaction.put_lease(
            LeaseRecord(
                id="lease-2",
                workload_id="wl-1",
                cell_id="cell-b",
                org_id=ORG,
                project_id=PROJECT,
                state=LeaseState.ACTIVE,
                resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
                generation=1,
                fencing_token="token-2",
                expires_at=utcnow() + timedelta(hours=24),
            )
        )
        workload = await transaction.get_workload("wl-1")
        await transaction.put_workload(workload.model_copy(update={"lease_ids": ["lease-1", "lease-2"]}))
    controller.workload_log_page = _scripted_page(records=[{"message": "second cell", "rank": 3}])

    await controller._archive_terminal_logs(await _workload(controller))
    manifest = await _manifest(controller)

    assert manifest["topology"]["width"] == 5
    assert manifest["topology"]["cells"] == [
        {"cell": "cell-a", "generation": 7, "width": 2},
        {"cell": "cell-b", "generation": 2, "width": 3},
    ]
    assert [entry["status"] for entry in manifest["ranks"]] == ["quiet", "quiet", "quiet", "archived", "quiet"]
    assert manifest["complete"] is True


# --------------------------------------------------------------------------- manifest-driven reader


async def test_the_reader_replays_from_the_manifest_and_tells_quiet_from_unreachable(
    tmp_path: Path,
    manifest_v2: None,
) -> None:
    """The distinction the embedded index cannot make, now visible on replay.

    A rank that printed nothing and a rank whose host could not be reached are
    both simply absent from the v1 index, so replay reported the unreachable one
    as if it had been read and found empty. The manifest names both, and the
    reader keeps them apart: silence is not truncation.
    """

    controller = await _controller(tmp_path, width=3)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    controller.workload_log_page = _scripted_page(
        records=[{"message": "zero", "rank": 0}, {"message": "two", "rank": 2}],
    )
    await controller._archive_terminal_logs(await _workload(controller))
    statuses = {rank: entry["status"] for rank, entry in _by_rank(await _manifest(controller)).items()}
    assert statuses[1] == "quiet"

    replayed = await controller._archived_log_page(await _workload(controller))

    assert [(record.rank, record.message) for record in replayed.records] == [(0, "zero"), (2, "two")]
    assert replayed.truncated_ranks == ()
    assert replayed.unavailable is False


async def test_the_reader_names_every_rank_a_failed_read_never_reached(
    tmp_path: Path,
    manifest_v2: None,
) -> None:
    """An attempt whose live read reached nothing must not replay as empty-and-fine."""

    controller = await _controller(tmp_path, width=3)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    controller.workload_log_page = _scripted_page(unavailable=True, records=[])
    await controller._archive_terminal_logs(await _workload(controller))
    statuses = {rank: entry["status"] for rank, entry in _by_rank(await _manifest(controller)).items()}
    assert set(statuses.values()) == {"unreachable"}

    replayed = await controller._archived_log_page(await _workload(controller))

    assert replayed.records == ()
    assert replayed.truncated_ranks == (0, 1, 2)
    assert replayed.unavailable is True


async def test_the_reader_refuses_bytes_that_contradict_the_manifest(tmp_path: Path, manifest_v2: None) -> None:
    """An object that does not match its own manifest is not evidence.

    Replaying it would present rewritten bytes as the rank's output, which is
    worse than reporting the rank as a gap: the caller cannot tell it happened.
    """

    controller = await _controller(tmp_path, width=2)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    controller.workload_log_page = _scripted_page(
        records=[{"message": "zero", "rank": 0}, {"message": "one", "rank": 1}],
    )
    await controller._archive_terminal_logs(await _workload(controller))

    tampered = _object_path(tmp_path, log_uri("wl-1", 1, 1))
    os.unlink(tampered)
    tampered.write_bytes(json.dumps({"message": "rewritten", "rank": 1}).encode() + b"\n")

    replayed = await controller._archived_log_page(await _workload(controller))

    assert [record.message for record in replayed.records] == ["zero"]
    assert "rewritten" not in (replayed.detail or "")
    assert replayed.truncated_ranks == (1,)
    # Either the content-addressed store refuses the bytes on read or the reader's
    # own digest check does. Both are the same guarantee; what matters is that the
    # rank becomes a named gap and the rewritten bytes never reach the caller.
    assert "could not be replayed: 1" in (replayed.detail or "")


async def test_an_unreadable_manifest_never_falls_back_to_the_narrower_index(
    tmp_path: Path,
    manifest_v2: None,
) -> None:
    """Falling back would answer with fewer expected ranks than are known to exist."""

    controller = await _controller(tmp_path, width=3)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    controller.workload_log_page = _scripted_page(records=[{"message": "zero", "rank": 0}])
    await controller._archive_terminal_logs(await _workload(controller))

    manifest_object = _object_path(tmp_path, log_manifest_uri("wl-1", 1))
    os.unlink(manifest_object)
    manifest_object.write_bytes(b"{not json")

    replayed = await controller._archived_log_page(await _workload(controller))

    assert replayed.unavailable is True
    assert replayed.records == ()
    assert "manifest" in (replayed.detail or "")


async def test_an_archive_without_a_manifest_still_reads_through_the_embedded_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Archives already on disk predate the manifest and must stay readable."""

    monkeypatch.delenv("EYVAN_LOG_ARCHIVE_MANIFEST_V2_ENABLED", raising=False)
    controller = await _controller(tmp_path, width=2)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    controller.workload_log_page = _scripted_page(
        records=[{"message": "zero", "rank": 0}, {"message": "one", "rank": 1}],
    )
    await controller._archive_terminal_logs(await _workload(controller))
    assert not await _committed(controller, log_manifest_uri("wl-1", 1))

    replayed = await controller._archived_log_page(await _workload(controller))

    assert [(record.rank, record.message) for record in replayed.records] == [(0, "zero"), (1, "one")]


async def test_a_released_attempt_on_a_cell_that_outlived_it_reads_its_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read a released attempt's archive after its attached cell advances to a newer generation."""

    from eyvan.records import LeaseState

    monkeypatch.delenv("EYVAN_LOG_ARCHIVE_MANIFEST_V2_ENABLED", raising=False)
    controller = await _controller(tmp_path, width=2)
    _attach(controller, _artifact_access(tmp_path / "artifacts"))
    controller.workload_log_page = _scripted_page(
        records=[{"message": "zero", "rank": 0}, {"message": "one", "rank": 1}],
    )
    await controller._archive_terminal_logs(await _workload(controller))
    del controller.workload_log_page  # back to the real reader
    async with controller.repository.transaction() as transaction:
        lease = await transaction.get_lease("lease-1")
        await transaction.put_lease(lease.model_copy(update={"state": LeaseState.RELEASED}))
        cell = await transaction.get_cell("cell-a")
        await transaction.put_cell(cell.model_copy(update={"generation": cell.generation + 1}))
        workload = await transaction.get_workload("wl-1")
        await transaction.put_workload(workload.model_copy(update={"state": WorkloadState.SUCCEEDED}))

    page = await controller.workload_log_page("wl-1")

    assert page["unavailable"] is False, page.get("detail")
    assert sorted((record["rank"], record["message"]) for record in page["records"]) == [(0, "zero"), (1, "one")]
    # A writer asking for live lines is still told there are none, not handed the archive.
    live = await controller.workload_log_page("wl-1", live_only=True)
    assert live["unavailable"] is True
    assert live["records"] == []


class _InFlightStore(_RecordingStore):
    """Records the most rank writes that were in flight at once."""

    def __init__(self, backing: LocalManagedArtifactStore, *, hold_s: float) -> None:
        super().__init__(backing)
        self.hold_s = hold_s
        self.in_flight = 0
        self.peak = 0

    def put_versioned(self, uri, payload, **scope):
        with self._lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            time.sleep(self.hold_s)
            return super().put_versioned(uri, payload, **scope)
        finally:
            with self._lock:
                self.in_flight -= 1


@pytest.mark.parametrize(("width", "concurrency"), [(6, 2), (5, 1), (4, 8)])
async def test_rank_objects_archive_concurrently_within_the_configured_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    concurrency: int,
) -> None:
    """Archive rank objects concurrently within log_archive_concurrency so settlement scales to wide slices."""

    import dataclasses

    monkeypatch.delenv("EYVAN_LOG_ARCHIVE_MANIFEST_V2_ENABLED", raising=False)
    controller = await _controller(tmp_path, width=width)
    controller.settings = dataclasses.replace(controller.settings, log_archive_concurrency=concurrency)
    store = _InFlightStore(LocalManagedArtifactStore(tmp_path / "artifacts"), hold_s=0.05)
    _attach(controller, _artifact_access(tmp_path / "artifacts", store))
    controller.workload_log_page = _scripted_page(
        records=[{"message": f"rank {rank}", "rank": rank} for rank in range(width)],
    )

    await controller._archive_terminal_logs(await _workload(controller))

    assert sorted(store.put_uris) == sorted(log_uri("wl-1", 1, rank) for rank in range(width))
    assert store.peak <= concurrency
    assert store.peak == min(concurrency, width)
    replayed = await controller._archived_log_page(await _workload(controller))
    assert sorted((record.rank, record.message) for record in replayed.records) == [
        (rank, f"rank {rank}") for rank in range(width)
    ]


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow
