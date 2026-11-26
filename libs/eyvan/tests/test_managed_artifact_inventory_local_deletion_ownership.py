# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regressions for managed artifact inventory and local deletion ownership."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest

from eyvan.artifacts.managed import (
    LocalManagedArtifactStore,
    ManagedArtifactDeleteInconclusiveError,
    ManagedArtifactDeleteStatus,
    result_uri,
)


@pytest.mark.parametrize("boundary", ("create", "file-fsync", "directory-fsync"))
def test_validation_marker_failure_closes_candidate_and_recovers(tmp_path: Path, monkeypatch, boundary: str) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    uri = result_uri("marker-failure")
    written = store.put_versioned(uri, b"generation", org_id="acme", project_id="research")
    original_open, original_close, original_fsync = os.open, os.close, os.fsync
    live: dict[int, tuple[str, int]] = {}
    marker_created = False
    injected = False

    def tracked_open(path, flags, *args, **kwargs):
        nonlocal marker_created, injected
        if path == "validated" and flags & os.O_CREAT:
            if boundary == "create":
                injected = True
                raise OSError(errno.ENOSPC, "injected validated marker creation failure")
            marker_created = True
        descriptor = original_open(path, flags, *args, **kwargs)
        live[descriptor] = (str(path), flags)
        return descriptor

    def tracked_close(descriptor):
        original_close(descriptor)
        live.pop(descriptor, None)

    def failing_fsync(descriptor):
        nonlocal injected
        if marker_created and not injected:
            name, _flags = live[descriptor]
            is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
            if (boundary == "file-fsync" and name == "validated") or (boundary == "directory-fsync" and is_directory):
                injected = True
                raise OSError(errno.EIO, "injected validated marker fsync failure")
        original_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", tracked_open)
        patch.setattr(os, "close", tracked_close)
        patch.setattr(os, "fsync", failing_fsync)
        try:
            with pytest.raises(ManagedArtifactDeleteInconclusiveError):
                store.delete_versioned(
                    uri, org_id="acme", project_id="research", object_version=written.metadata.object_version
                )
            assert injected
            assert not live, f"unclosed descriptors after inconclusive deletion: {live}"
        finally:
            # The RED implementation leaks descriptors; close only this test's measured leaks.
            for descriptor in tuple(live):
                tracked_close(descriptor)

    recovered = LocalManagedArtifactStore(store.root).delete_versioned(
        uri, org_id="acme", project_id="research", object_version=written.metadata.object_version
    )
    assert recovered.status is ManagedArtifactDeleteStatus.DELETED
    assert store.stat(uri, org_id="acme", project_id="research") is None


@pytest.mark.parametrize("boundary", ("scope", "kind", "child-open"))
def test_inventory_pins_parent_directories(tmp_path: Path, monkeypatch, boundary: str) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    written = store.put_versioned(result_uri("original"), b"original", org_id="acme", project_id="research")
    scope = next((store.root / "scopes").iterdir())
    kind = scope / "results"
    outside = tmp_path / "outside"
    (outside / "results").mkdir(parents=True)
    (outside / "results" / "foreign.result").write_bytes(b"outside-secret-metadata")
    victim = scope if boundary == "scope" else kind
    target = outside if boundary == "scope" else outside / "results"
    victim_identity = victim.stat()
    original_scandir, original_open = os.scandir, os.open
    swapped = False

    def swap():
        nonlocal swapped
        victim.rename(victim.with_name(victim.name + "-original"))
        victim.symlink_to(target, target_is_directory=True)
        swapped = True

    def racing_scandir(path):
        metadata = os.fstat(path) if isinstance(path, int) else Path(path).stat()
        if (
            boundary != "child-open"
            and not swapped
            and (metadata.st_dev, metadata.st_ino) == (victim_identity.st_dev, victim_identity.st_ino)
        ):
            swap()
        return original_scandir(path)

    def racing_open(path, flags, *args, **kwargs):
        if boundary == "child-open" and not swapped and str(path).endswith("original.result"):
            # DirEntry classification already succeeded; swap a parent immediately
            # before the metadata descriptor open, not just before enumeration.
            (outside / "results" / "original.result").write_bytes(b"outside-secret-metadata")
            swap()
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", racing_scandir)
    monkeypatch.setattr(os, "open", racing_open)
    page = store.list_scope(org_id="acme", project_id="research")
    assert swapped
    assert page.objects == (written.metadata,)
