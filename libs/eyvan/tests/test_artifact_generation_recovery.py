# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unpublished partial writes must not poison a digest anchor.

Caught I/O errors and abrupt writer process death are distinct interruption
boundaries. Only fully written, durable staging bytes may be published under
the immutable anchor name. Recovery preserves published or ambiguous links.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_priority_preemption_properties import repository as _repository_fixture

from eyvan.artifacts.managed import (
    LocalManagedArtifactStore,
    ManagedArtifactConflictError,
    callable_uri,
    parse_managed_uri,
)
from eyvan.records import ManagedArtifactState
from eyvan.server.service import ControlPlaneService

repository = _repository_fixture

_PAYLOAD = b"callable-payload"
_DIGEST = hashlib.sha256(_PAYLOAD).hexdigest()


def test_a_failed_generation_write_does_not_poison_the_digest(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = LocalManagedArtifactStore(tmp_path / "artifacts")
    uri = callable_uri(_DIGEST)
    original = store._write_descriptor
    attempts = {"count": 0}

    def flaky(descriptor: int, payload: bytes) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("simulated ENOSPC mid-write")
        original(descriptor, payload)

    monkeypatch.setattr(store, "_write_descriptor", flaky)
    with pytest.raises(OSError, match="ENOSPC"):
        store.put(uri, _PAYLOAD, org_id="acme", project_id="research")

    monkeypatch.setattr(store, "_write_descriptor", original)
    store.put(uri, _PAYLOAD, org_id="acme", project_id="research")
    assert store.exists(uri, org_id="acme", project_id="research")


@pytest.mark.parametrize("boundary", ("partial-write", "anchor-publication"))
async def test_fresh_service_retries_after_generation_writer_process_dies(repository, tmp_path, boundary) -> None:
    root = tmp_path / "artifacts"
    store = LocalManagedArtifactStore(root)
    service = ControlPlaneService(repository, artifact_store=store)
    uri = callable_uri(_DIGEST)
    reserved = await service.reserve_managed_artifact_upload(
        org_id="acme",
        project_id="research",
        uri=uri,
        size_bytes=len(_PAYLOAD),
        sha256=_DIGEST,
        actor="alice",
    )
    assert reserved.state is ManagedArtifactState.PENDING_UPLOAD
    program = """
import os
from pathlib import Path
import sys
from eyvan.artifacts.managed import LocalManagedArtifactStore
root, uri, boundary, digest, payload_hex = sys.argv[1:]
store = LocalManagedArtifactStore(Path(root))
if boundary == "partial-write":
    def die_mid_write(descriptor, payload):
        os.write(descriptor, payload[:5])
        os.fsync(descriptor)
        os._exit(71)
    store._write_descriptor = die_mid_write
else:
    original_link = os.link
    def die_after_publication(source, destination, **kwargs):
        original_link(source, destination, **kwargs)
        if str(destination).endswith(".artifact") or str(destination).startswith(digest):
            os.fsync(kwargs["dst_dir_fd"])
            os._exit(71)
    os.link = die_after_publication
store.put_versioned(uri, bytes.fromhex(payload_hex), org_id="acme", project_id="research")
raise AssertionError("interruption boundary not reached")
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(root), uri, boundary, _DIGEST, _PAYLOAD.hex()],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 71, (result.stdout, result.stderr)
    async with repository.transaction() as tx:
        interrupted = await tx.get_managed_artifact("acme", "research", uri)
    assert interrupted.state is ManagedArtifactState.PENDING_UPLOAD
    assert interrupted.record_generation == reserved.record_generation

    fresh = ControlPlaneService(repository, artifact_store=LocalManagedArtifactStore(root))
    active = await fresh.upload_managed_artifact(
        org_id="acme",
        project_id="research",
        uri=uri,
        payload=_PAYLOAD,
        sha256=_DIGEST,
        actor="alice",
    )
    assert active.state is ManagedArtifactState.ACTIVE
    assert active.record_generation == reserved.record_generation
    assert active.object_version
    assert await fresh.read_managed_artifact(org_id="acme", project_id="research", uri=uri) == _PAYLOAD
    replayed = await fresh.upload_managed_artifact(
        org_id="acme",
        project_id="research",
        uri=uri,
        payload=_PAYLOAD,
        sha256=_DIGEST,
        actor="alice",
    )
    assert replayed.state is ManagedArtifactState.ACTIVE
    assert replayed.object_version == active.object_version
    assert replayed.record_generation == active.record_generation
    assert replayed.sha256 == active.sha256 == _DIGEST
    assert replayed.size_bytes == active.size_bytes == len(_PAYLOAD)
    assert not list((root / store._GENERATION_DIRECTORY).rglob("*.tmp"))


@pytest.mark.parametrize("links", (1, 2))
def test_a_conflicting_legacy_generation_is_preserved(tmp_path, links) -> None:
    store = LocalManagedArtifactStore(tmp_path / "artifacts")
    uri = callable_uri(_DIGEST)
    reference = parse_managed_uri(uri)
    relative = store._relative_object(reference, "acme", "research")
    with store._object_lock(reference, "acme", "research"):
        with store._open_generation_directory(relative, create=True) as generation_fd:
            descriptor = os.open(
                f"{_DIGEST}.artifact", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=generation_fd
            )
            os.write(descriptor, b"partial")
            os.close(descriptor)
    anchor = store.root / store._GENERATION_DIRECTORY / store._object_key(relative) / f"{_DIGEST}.artifact"
    alias = tmp_path / "unexpected-link"
    if links == 2:
        os.link(anchor, alias)
    before = anchor.stat()

    with pytest.raises(ManagedArtifactConflictError):
        store.put(uri, _PAYLOAD, org_id="acme", project_id="research")

    assert anchor.read_bytes() == b"partial"
    if links == 2:
        assert alias.read_bytes() == b"partial"
    assert anchor.stat().st_ino == before.st_ino
    assert anchor.stat().st_nlink == links


def test_a_published_generation_is_never_recreated_during_identical_replay(tmp_path) -> None:
    store = LocalManagedArtifactStore(tmp_path / "artifacts")
    uri = callable_uri(_DIGEST)
    first = store.put_versioned(uri, _PAYLOAD, org_id="acme", project_id="research")
    replayed = LocalManagedArtifactStore(store.root).put_versioned(uri, _PAYLOAD, org_id="acme", project_id="research")
    assert replayed.metadata == first.metadata
    assert replayed.created is False
    assert store.get(uri, org_id="acme", project_id="research") == _PAYLOAD
