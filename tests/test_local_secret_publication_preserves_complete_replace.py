# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Local secret publication preserves complete, no-replace profile identities."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from eyvan.server.local.auth import LocalDevelopmentCredentialError, LocalDevelopmentCredentialStore
from eyvan.server.local.profile import LocalArtifactTokenKeyStore, LocalProfileSecretError

STORE_TYPES = (LocalDevelopmentCredentialStore, LocalArtifactTokenKeyStore)
SECRET_ERRORS = (LocalDevelopmentCredentialError, LocalProfileSecretError)


@pytest.mark.parametrize("store_type", STORE_TYPES)
def test_secret_is_unpublished_and_private_until_file_fsync(tmp_path: Path, monkeypatch, store_type) -> None:
    path = tmp_path / "profile" / "secret"
    original_fsync = os.fsync
    persisted_files = []

    def fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if stat.S_ISREG(metadata.st_mode):
            assert not path.exists(), "secret became visible before file persistence"
            assert stat.S_IMODE(metadata.st_mode) == 0o600
            temporary_paths = list(path.parent.iterdir())
            assert len(temporary_paths) == 1
            persisted_files.append(temporary_paths[0].read_bytes())
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)
    store_type(path).load_or_create()
    assert persisted_files == [path.read_bytes()]
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("store_type", STORE_TYPES)
def test_publication_failure_cleans_only_unpublished_temporary(tmp_path: Path, monkeypatch, store_type) -> None:
    path = tmp_path / "profile" / "secret"

    def fail_link(*args, **kwargs) -> None:
        raise OSError("injected no-replace publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", fail_link)
        with pytest.raises(SECRET_ERRORS, match="could not create"):
            store_type(path).load_or_create()
    assert not path.exists()
    assert list(path.parent.iterdir()) == []
    created = store_type(path).load_or_create()
    assert store_type(path).load_or_create() == created


@pytest.mark.parametrize("store_type", STORE_TYPES)
def test_no_replace_publication_preserves_malformed_concurrent_winner(tmp_path: Path, monkeypatch, store_type) -> None:
    path = tmp_path / "profile" / "secret"
    original_link = os.link

    def competing_link(source, destination, **kwargs) -> None:
        path.write_bytes(b"malformed-concurrent-winner")
        path.chmod(0o600)
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(SECRET_ERRORS):
        store_type(path).load_or_create()
    assert path.read_bytes() == b"malformed-concurrent-winner"
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("store_type", STORE_TYPES)
def test_directory_fsync_failure_never_rolls_back_published_identity(tmp_path: Path, monkeypatch, store_type) -> None:
    path = tmp_path / "profile" / "secret"
    original_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory sync failure")
        original_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_directory_fsync)
        with pytest.raises(SECRET_ERRORS, match="could not sync"):
            store_type(path).load_or_create()
    published = path.read_bytes()
    loaded = store_type(path).load_or_create()
    assert store_type(path).load_or_create() == loaded
    assert path.read_bytes() == published
    assert list(path.parent.iterdir()) == [path]
