# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Stable local secrets must become visible only after complete persistence."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, get_ident

import pytest

from eyvan.server.local.auth import LocalDevelopmentCredentialError, LocalDevelopmentCredentialStore
from eyvan.server.local.profile import LocalArtifactTokenKeyStore, LocalProfileSecretError

Store = LocalDevelopmentCredentialStore | LocalArtifactTokenKeyStore
STORE_TYPES = (LocalDevelopmentCredentialStore, LocalArtifactTokenKeyStore)
SECRET_ERRORS = (LocalDevelopmentCredentialError, LocalProfileSecretError)


@pytest.mark.parametrize("store_type", STORE_TYPES)
@pytest.mark.parametrize("pause_after_partial_write", (False, True), ids=("before-first-write", "after-partial-write"))
def test_concurrent_creators_never_load_an_unpublished_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    store_type: type[Store],
    pause_after_partial_write: bool,
) -> None:
    """Pause a real creator's write while another initializer uses the same path."""

    path = tmp_path / "profile" / "secret"
    original_write = os.write
    paused = Event()
    resume = Event()
    creator_thread: list[int] = []
    writes = 0

    def paused_write(descriptor: int, data: bytes) -> int:
        nonlocal writes
        if creator_thread and get_ident() == creator_thread[0]:
            writes += 1
            if pause_after_partial_write and writes == 1:
                return original_write(descriptor, data[: len(data) // 2])
            if writes == (2 if pause_after_partial_write else 1):
                paused.set()
                if not resume.wait(10):
                    raise TimeoutError("test did not resume secret creation")
        return original_write(descriptor, data)

    def create_first():
        creator_thread.append(get_ident())
        return store_type(path).load_or_create()

    monkeypatch.setattr(os, "write", paused_write)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(create_first)
        try:
            assert paused.wait(10), "creator never reached the publication window"
            second = executor.submit(store_type(path).load_or_create)
            try:
                second_value = second.result(timeout=10)
                second_error = None
            except SECRET_ERRORS as error:
                second_error = error
                second_value = None
        finally:
            resume.set()
        first_value = first.result(timeout=10)

    assert second_error is None, f"a concurrent initializer read an incomplete final file: {second_error}"
    assert first_value == second_value == store_type(path).load_or_create()
    assert list(path.parent.iterdir()) == [path], "unpublished temporary secret files leaked"


@pytest.mark.parametrize("store_type", STORE_TYPES)
@pytest.mark.parametrize("failure_stage", ("before-first-write", "after-partial-write", "before-file-fsync"))
def test_unpublished_failed_initialization_allows_retry_without_rotating_a_published_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    store_type: type[Store],
    failure_stage: str,
) -> None:
    """Inject an I/O failure around real writes, then retry through the public store."""

    path = tmp_path / "profile" / "secret"
    original_write = os.write
    writes = 0

    def failed_write(descriptor: int, data: bytes) -> int:
        nonlocal writes
        writes += 1
        if failure_stage == "after-partial-write" and writes == 1:
            return original_write(descriptor, data[: len(data) // 2])
        raise OSError("injected secret write failure")

    def failed_fsync(descriptor: int) -> None:
        raise OSError("injected secret fsync failure")

    with monkeypatch.context() as patch:
        if failure_stage == "before-file-fsync":
            patch.setattr(os, "fsync", failed_fsync)
        else:
            patch.setattr(os, "write", failed_write)
        with pytest.raises(SECRET_ERRORS, match="could not persist"):
            store_type(path).load_or_create()

    assert not path.exists(), "failed initialization published an incomplete or unpersisted secret"
    assert list(path.parent.iterdir()) == [], "failed initialization left its private temporary file"
    created = store_type(path).load_or_create()
    published = path.read_bytes()
    assert store_type(path).load_or_create() == created
    assert path.read_bytes() == published
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("store_type", STORE_TYPES)
def test_existing_malformed_published_secrets_are_refused_not_rotated(tmp_path: Path, store_type: type[Store]) -> None:
    path = tmp_path / "profile" / "secret"
    path.parent.mkdir(mode=0o700)
    path.write_bytes(b"malformed-published-secret")
    path.chmod(0o600)

    for _ in range(2):
        with pytest.raises(SECRET_ERRORS):
            store_type(path).load_or_create()
        assert path.read_bytes() == b"malformed-published-secret"


@pytest.mark.parametrize("store_type", STORE_TYPES)
def test_existing_valid_published_secrets_remain_stable(tmp_path: Path, store_type: type[Store]) -> None:
    path = tmp_path / "profile" / "secret"
    created = store_type(path).load_or_create()
    published = path.read_bytes()

    for _ in range(2):
        assert store_type(path).load_or_create() == created
        assert path.read_bytes() == published
    assert list(path.parent.iterdir()) == [path]
