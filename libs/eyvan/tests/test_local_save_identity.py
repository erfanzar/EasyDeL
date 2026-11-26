# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A save skips re-reading only the exact file this store itself last loaded or wrote."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from eyvan.server.persistence.local import LocalControlPlaneStore
from eyvan.server.persistence.state import ControlPlaneState

_V1_FIXTURE = Path(__file__).parent / "fixtures" / "local_control_plane_v1.json"


def _counting_store(path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[LocalControlPlaneStore, list[int]]:
    store = LocalControlPlaneStore(path)
    reads: list[int] = []
    original = store._read_document

    def counted():
        reads.append(1)
        return original()

    monkeypatch.setattr(store, "_read_document", counted)
    return store, reads


def test_consecutive_saves_do_not_reread_their_own_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "control-plane.json"
    LocalControlPlaneStore(path).save(ControlPlaneState())
    store, reads = _counting_store(path, monkeypatch)

    state = store.load()
    assert len(reads) == 1
    for revision in range(1, 4):
        state.revision = revision
        store.save(state)
    assert len(reads) == 1, "the store's own file is never re-parsed before a save"
    assert json.loads(path.read_text())["revision"] == 3
    assert store.load().revision == 3


def test_an_outside_rewrite_takes_the_full_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "control-plane.json"
    LocalControlPlaneStore(path).save(ControlPlaneState())
    store, reads = _counting_store(path, monkeypatch)
    state = store.load()
    store.save(state)
    assert len(reads) == 1

    # Same content written in place by someone else: mtime/ctime move.
    payload = path.read_bytes()
    stat_before = path.stat()
    os.utime(path, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns + 1_000_000_000))
    path.write_bytes(payload)
    store.save(state)
    assert len(reads) == 2, "a file this store did not write is read again before replacing it"


def test_a_v1_document_swapped_in_from_outside_is_still_backed_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "control-plane.json"
    store, _reads = _counting_store(path, monkeypatch)
    store.save(ControlPlaneState())
    state = store.load()

    replacement = tmp_path / "incoming.json"
    shutil.copyfile(_V1_FIXTURE, replacement)
    replacement.chmod(0o600)
    os.replace(replacement, path)

    store.save(state)
    assert store.v1_backup_path.read_bytes() == _V1_FIXTURE.read_bytes()
    assert json.loads(path.read_text())["version"] != 1


def test_a_failed_save_forgets_the_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "control-plane.json"
    LocalControlPlaneStore(path).save(ControlPlaneState())
    store, reads = _counting_store(path, monkeypatch)
    state = store.load()

    def boom(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_write_owner_only_json_temporary", boom)
        with pytest.raises(Exception, match="disk full"):
            store.save(state)
    store.save(state)
    assert len(reads) == 2, "after a failed save the next one reads the file again"
