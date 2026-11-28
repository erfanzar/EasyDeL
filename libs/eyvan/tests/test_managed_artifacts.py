# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from eyvan.artifacts.managed import (
    ArtifactAction,
    ArtifactTokenSigner,
    ExpiredArtifactToken,
    GCSManagedArtifactStore,
    InvalidArtifactToken,
    InvalidManagedArtifactURI,
    LocalManagedArtifactStore,
    ManagedArtifactConflictError,
    ManagedArtifactDeleteInconclusiveError,
    ManagedArtifactDeleteStatus,
    ManagedArtifactInventoryError,
    ManagedArtifactKind,
    ManagedArtifactObjectMetadata,
    ManagedArtifactOperationInconclusiveError,
    ManagedArtifactVersionMismatchError,
    ManagedArtifactWriteInconclusiveError,
    callable_uri,
    parse_managed_uri,
    result_uri,
    storage_scope,
)
from eyvan.artifacts.transfers import ManagedArtifactAccess
from eyvan.identity.rotation import RotatingCredential


def test_managed_uri_helpers_are_canonical_and_reject_path_traversal() -> None:
    digest = hashlib.sha256(b"callable").hexdigest()

    callable_reference = parse_managed_uri(callable_uri(digest))
    result_reference = parse_managed_uri(result_uri("run-1.part_0"))

    assert callable_reference.kind is ManagedArtifactKind.CALLABLES
    assert callable_reference.object_id == digest
    assert result_reference.kind is ManagedArtifactKind.RESULTS
    assert result_reference.object_id == "run-1.part_0"

    invalid = (
        "eyvan://results/../secret",
        "eyvan://results/%2e%2e",
        "eyvan://results/a/b",
        "eyvan://results/run-1?download=true",
        "eyvan://results/run-1#fragment",
        "eyvan://unknown/run-1",
        "eyvan:///results/run-1",
        "EYVAN://results/run-1",
    )
    for uri in invalid:
        with pytest.raises(InvalidManagedArtifactURI):
            parse_managed_uri(uri)
    with pytest.raises(InvalidManagedArtifactURI):
        callable_uri("A" * 64)
    with pytest.raises(InvalidManagedArtifactURI):
        result_uri("../../secret")


def test_local_store_is_content_addressed_atomic_and_project_scoped(tmp_path: Path) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    payload = b"serialized callable"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())

    assert not store.exists(uri, org_id="acme", project_id="research")
    assert store.put(uri, payload, org_id="acme", project_id="research") == uri
    assert store.put(uri, payload, org_id="acme", project_id="research") == uri
    assert store.get(uri, org_id="acme", project_id="research") == payload
    assert not store.exists(uri, org_id="acme", project_id="other")

    expected = (
        store.root
        / "scopes"
        / storage_scope("acme", "research")
        / "callables"
        / f"{hashlib.sha256(payload).hexdigest()}.pkl"
    )
    assert expected.read_bytes() == payload
    assert "acme" not in expected.as_posix()
    assert "research" not in expected.as_posix()

    with pytest.raises(ValueError, match="content-addressed"):
        store.put(callable_uri("0" * 64), payload, org_id="acme", project_id="research")


def test_local_store_creates_a_missing_configured_root_before_first_write(tmp_path: Path) -> None:
    root = tmp_path / "missing" / "managed-artifacts"
    payload = b"first managed callable"
    uri = callable_uri(hashlib.sha256(payload).hexdigest())
    assert not root.exists()

    store = LocalManagedArtifactStore(root)
    stored_uri = store.put(uri, payload, org_id="acme", project_id="research")

    assert root.is_dir()
    assert stored_uri == uri
    assert store.get(uri, org_id="acme", project_id="research") == payload


def test_local_result_writes_are_immutable_and_idempotent(tmp_path: Path) -> None:
    store = LocalManagedArtifactStore(tmp_path)
    uri = result_uri("workload-1")

    store.put(uri, b"first", org_id="acme", project_id="research")
    store.put(uri, b"first", org_id="acme", project_id="research")

    with pytest.raises(ManagedArtifactConflictError, match="different bytes"):
        store.put(uri, b"second", org_id="acme", project_id="research")
    assert store.get(uri, org_id="acme", project_id="research") == b"first"


def test_local_result_race_commits_exactly_one_payload(tmp_path: Path) -> None:
    store = LocalManagedArtifactStore(tmp_path)
    uri = result_uri("workload-race")
    barrier = threading.Barrier(2)

    def write(payload: bytes) -> bytes | ManagedArtifactConflictError:
        barrier.wait()
        try:
            store.put(uri, payload, org_id="acme", project_id="research")
        except ManagedArtifactConflictError as exc:
            return exc
        return payload

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(write, (b"first", b"second")))

    assert sum(isinstance(outcome, ManagedArtifactConflictError) for outcome in outcomes) == 1
    assert store.get(uri, org_id="acme", project_id="research") in {b"first", b"second"}


def test_local_store_rejects_symlink_escape_before_writing(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    store = LocalManagedArtifactStore(root)
    (root / "scopes").symlink_to(outside, target_is_directory=True)

    with pytest.raises(InvalidManagedArtifactURI, match="symbolic links"):
        store.put(result_uri("run-1"), b"secret", org_id="acme", project_id="research")

    assert list(outside.iterdir()) == []


def test_local_versioned_store_retries_one_inode_and_deletes_only_that_version(tmp_path: Path) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    uri = result_uri("run-versioned")

    first = store.put_versioned(uri, b"first", org_id="acme", project_id="research")
    retried = store.put_versioned(uri, b"first", org_id="acme", project_id="research")
    exact = store.get_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=first.metadata.object_version,
    )

    assert first.created is True
    assert retried.created is False
    assert retried.metadata == first.metadata
    assert exact.payload == b"first"
    assert exact.metadata == first.metadata
    assert first.metadata.object_version.startswith("local:2:")
    assert first.metadata.sha256 == hashlib.sha256(b"first").hexdigest()
    assert list((store.root / ".locks").glob("*.lock"))
    assert list((store.root / ".generations").rglob("*.artifact"))
    assert not list((store.root / "scopes").rglob("*.lock"))

    target = store.root / "scopes" / storage_scope("acme", "research") / "results" / "run-versioned.result"
    replacement = target.with_name("replacement.result")
    replacement.write_bytes(b"replacement")
    replacement.replace(target)

    with pytest.raises(ManagedArtifactVersionMismatchError):
        store.get_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=first.metadata.object_version,
        )
    mismatch = store.delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=first.metadata.object_version,
    )
    assert mismatch.status is ManagedArtifactDeleteStatus.VERSION_MISMATCH
    assert mismatch.current_metadata is not None
    assert mismatch.current_metadata.object_version != first.metadata.object_version
    assert target.read_bytes() == b"replacement"

    deleted = store.delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=mismatch.current_metadata.object_version,
    )
    absent = store.delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=mismatch.current_metadata.object_version,
    )
    assert deleted.status is ManagedArtifactDeleteStatus.DELETED
    assert absent.status is ManagedArtifactDeleteStatus.ABSENT


def test_local_conditional_delete_never_unlinks_a_rename_at_detach_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    uri = result_uri("rename-at-delete")
    written = store.put_versioned(uri, b"generation-a", org_id="acme", project_id="research")
    target = store.root / "scopes" / storage_scope("acme", "research") / "results" / "rename-at-delete.result"
    replacement = target.with_name("replacement.result")
    replacement.write_bytes(b"generation-b")
    original_rename = os.rename
    replacement_installed = False

    def adversarial_rename(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal replacement_installed
        if source == target.name and not replacement_installed:
            replacement_installed = True
            assert src_dir_fd is not None
            original_rename(
                replacement.name,
                target.name,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=src_dir_fd,
            )
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "rename", adversarial_rename)
    outcome = store.delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=written.metadata.object_version,
    )

    assert replacement_installed
    assert outcome.status is ManagedArtifactDeleteStatus.VERSION_MISMATCH
    assert outcome.current_metadata is not None
    assert outcome.current_metadata.sha256 == hashlib.sha256(b"generation-b").hexdigest()
    assert target.read_bytes() == b"generation-b"


def test_local_delete_raw_detached_replacement_is_preserved_before_receipt_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    uri = result_uri("replace-detached-raw")
    written = store.put_versioned(uri, b"generation-a", org_id="acme", project_id="research")
    target = store.root / "scopes" / storage_scope("acme", "research") / "results" / "replace-detached-raw.result"
    replacement = target.with_name("replacement-detached-raw.result")
    replacement.write_bytes(b"generation-b")
    original_ftruncate = os.ftruncate
    original_rename = os.rename
    original_unlink = os.unlink
    unlinked_names: list[tuple[str, int | None]] = []
    installed = False

    def adversarial_ftruncate(descriptor: int, length: int) -> None:
        nonlocal installed
        if length == 0 and not installed:
            installed = True
            detached = next((store.root / ".generations").rglob("detached"))
            # Deliberately bypass every store helper: first install B at the
            # canonical name, then rename that raw path over the verified
            # staging entry. The truncate must still hit generation A's fd,
            # while this generation B entry survives.
            original_rename(replacement, target)
            generation_directory = detached.parent.parent
            os.link(target, generation_directory / f"{hashlib.sha256(b'generation-b').hexdigest()}.artifact")
            original_rename(target, detached)
        original_ftruncate(descriptor, length)

    def record_unlink(path: str, *, dir_fd: int | None = None) -> None:
        unlinked_names.append((os.fspath(path), dir_fd))
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "ftruncate", adversarial_ftruncate)
    monkeypatch.setattr(os, "unlink", record_unlink)
    outcome = store.delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=written.metadata.object_version,
    )

    assert installed
    detached_unlinks = [directory_fd for name, directory_fd in unlinked_names if name == "detached"]
    assert detached_unlinks
    assert all(directory_fd is not None for _name, directory_fd in unlinked_names)
    assert outcome.status is ManagedArtifactDeleteStatus.VERSION_MISMATCH
    assert outcome.current_metadata is not None
    replacement_digest = hashlib.sha256(b"generation-b").hexdigest()
    assert outcome.current_metadata.sha256 == replacement_digest
    assert target.read_bytes() == b"generation-b"
    generation_directory = next((store.root / ".generations").iterdir())
    replacement_anchor = generation_directory / f"{replacement_digest}.artifact"
    assert replacement_anchor.samefile(target)
    assert target.stat().st_nlink == 2
    assert not list(generation_directory.glob(".delete-*"))
    assert not list(generation_directory.glob(".trash-*"))


def test_local_delete_rehashes_same_inode_mutated_during_detach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    uri = result_uri("mutated-during-detach")
    written = store.put_versioned(uri, b"generation-a", org_id="acme", project_id="research")
    target = store.root / "scopes" / storage_scope("acme", "research") / "results" / "mutated-during-detach.result"
    original_rename = os.rename
    mutated = False

    def adversarial_rename(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal mutated
        if source == target.name and destination == "detached" and not mutated:
            mutated = True
            inode = target.stat().st_ino
            os.chmod(target, 0o600)
            target.write_bytes(b"generation-b-expanded")
            assert target.stat().st_ino == inode
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "rename", adversarial_rename)
    outcome = store.delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=written.metadata.object_version,
    )

    assert mutated
    assert outcome.status is ManagedArtifactDeleteStatus.VERSION_MISMATCH
    assert outcome.current_metadata is not None
    assert outcome.current_metadata.sha256 == hashlib.sha256(b"generation-b-expanded").hexdigest()
    assert target.read_bytes() == b"generation-b-expanded"
    assert target.stat().st_ino != int(written.metadata.object_version.split(":")[-2])


def test_local_delete_retains_an_invalid_content_addressed_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    original_payload = b"callable-generation-a"
    replacement_payload = b"callable-generation-b-invalid-for-uri"
    digest = hashlib.sha256(original_payload).hexdigest()
    uri = callable_uri(digest)
    written = store.put_versioned(uri, original_payload, org_id="acme", project_id="research")
    target = store.root / "scopes" / storage_scope("acme", "research") / "callables" / f"{digest}.pkl"
    original_rename = os.rename
    original_unlink = os.unlink
    mutated = False
    unlinked_receipt_entries: list[str] = []

    def adversarial_rename(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal mutated
        if source == target.name and destination == "detached" and not mutated:
            mutated = True
            inode = target.stat().st_ino
            os.chmod(target, 0o600)
            target.write_bytes(replacement_payload)
            assert target.stat().st_ino == inode
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    def record_unlink(path: str, *, dir_fd: int | None = None) -> None:
        unlinked_receipt_entries.append(os.fspath(path))
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "rename", adversarial_rename)
    monkeypatch.setattr(os, "unlink", record_unlink)
    with pytest.raises(ManagedArtifactDeleteInconclusiveError):
        store.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )

    assert mutated
    transactions = list((store.root / ".generations").glob("*/.delete-v1-*"))
    assert len(transactions) == 1
    retained_payloads = [target.read_bytes()] if target.exists() and target.is_file() else []
    retained_payloads.extend(
        path.read_bytes()
        for path in transactions[0].iterdir()
        if path.is_file() and not path.is_symlink() and path.stat().st_size
    )
    assert replacement_payload in retained_payloads
    assert not list((store.root / ".generations").glob("*/.trash-v1-*"))
    assert not {"detached", "preserved-anchor"}.intersection(unlinked_receipt_entries)


def test_local_delete_retains_detached_b_when_a_different_c_becomes_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    uri = result_uri("replacement-b-versus-current-c")
    written = store.put_versioned(uri, b"generation-a", org_id="acme", project_id="research")
    target = (
        store.root / "scopes" / storage_scope("acme", "research") / "results" / "replacement-b-versus-current-c.result"
    )
    staging_b = target.with_name("staging-generation-b.result")
    staging_b.write_bytes(b"generation-b")
    original_ftruncate = os.ftruncate
    original_rename = os.rename
    original_install = store._install_preserved_replacement
    installed_b = False
    installed_c = False

    def adversarial_ftruncate(descriptor: int, length: int) -> None:
        nonlocal installed_b
        if length == 0 and not installed_b:
            installed_b = True
            detached = next((store.root / ".generations").rglob("detached"))
            original_rename(staging_b, target)
            generation_directory = detached.parent.parent
            os.link(target, generation_directory / f"{hashlib.sha256(b'generation-b').hexdigest()}.artifact")
            original_rename(target, detached)
        original_ftruncate(descriptor, length)

    def install_after_c_appears(
        generation_fd: int,
        directory_fd: int,
        relative: Path,
        reference,
        payload: bytes,
    ):
        nonlocal installed_c
        if not installed_c:
            installed_c = True
            target.write_bytes(b"generation-c")
            target.chmod(0o400)
        return original_install(generation_fd, directory_fd, relative, reference, payload)

    monkeypatch.setattr(os, "ftruncate", adversarial_ftruncate)
    monkeypatch.setattr(store, "_install_preserved_replacement", install_after_c_appears)
    with pytest.raises(ManagedArtifactDeleteInconclusiveError):
        store.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )

    assert installed_b and installed_c
    assert target.read_bytes() == b"generation-c"
    transactions = list((store.root / ".generations").glob("*/.delete-v1-*"))
    assert len(transactions) == 1
    assert any(
        path.is_file() and not path.is_symlink() and path.read_bytes() == b"generation-b"
        for path in transactions[0].iterdir()
    )
    assert not list((store.root / ".generations").glob("*/.trash-v1-*"))


def test_local_delete_anchors_an_exact_current_b_before_receipt_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    uri = result_uri("replacement-b-is-current-b")
    written = store.put_versioned(uri, b"generation-a", org_id="acme", project_id="research")
    target = store.root / "scopes" / storage_scope("acme", "research") / "results" / "replacement-b-is-current-b.result"
    staging_b = target.with_name("staging-current-generation-b.result")
    staging_b.write_bytes(b"generation-b")
    original_ftruncate = os.ftruncate
    original_rename = os.rename
    original_install = store._install_preserved_replacement
    installed_b = False
    current_b = False

    def adversarial_ftruncate(descriptor: int, length: int) -> None:
        nonlocal installed_b
        if length == 0 and not installed_b:
            installed_b = True
            detached = next((store.root / ".generations").rglob("detached"))
            original_rename(staging_b, target)
            generation_directory = detached.parent.parent
            os.link(target, generation_directory / f"{hashlib.sha256(b'generation-b').hexdigest()}.artifact")
            original_rename(target, detached)
        original_ftruncate(descriptor, length)

    def install_after_same_b_appears(
        generation_fd: int,
        directory_fd: int,
        relative: Path,
        reference,
        payload: bytes,
    ):
        nonlocal current_b
        if not current_b:
            current_b = True
            target.write_bytes(b"generation-b")
            target.chmod(0o400)
        return original_install(generation_fd, directory_fd, relative, reference, payload)

    monkeypatch.setattr(os, "ftruncate", adversarial_ftruncate)
    monkeypatch.setattr(store, "_install_preserved_replacement", install_after_same_b_appears)
    outcome = store.delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=written.metadata.object_version,
    )

    assert installed_b and current_b
    assert outcome.status is ManagedArtifactDeleteStatus.VERSION_MISMATCH
    assert target.read_bytes() == b"generation-b"
    observed = store.stat(uri, org_id="acme", project_id="research")
    assert observed is not None
    replacement_digest = hashlib.sha256(b"generation-b").hexdigest()
    assert observed.object_version.startswith(f"local:2:{replacement_digest}:")
    anchor = next((store.root / ".generations").glob(f"*/{replacement_digest}.artifact"))
    assert anchor.samefile(target)
    assert not list((store.root / ".generations").glob("*/.delete-v1-*"))
    assert not list((store.root / ".generations").glob("*/.trash-v1-*"))


@pytest.mark.parametrize("crash_boundary", ("detached", "truncated", "anchor-retired"))
def test_local_delete_crash_residue_is_private_and_retry_put_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_boundary: str,
) -> None:
    class SimulatedCrash(BaseException):
        pass

    root = tmp_path / f"managed-{crash_boundary}"
    store = LocalManagedArtifactStore(root)
    uri = result_uri(f"crash-{crash_boundary}")
    written = store.put_versioned(uri, b"generation-a", org_id="acme", project_id="research")
    original_rename = os.rename
    original_ftruncate = os.ftruncate
    crashed = False

    def crash_after_rename(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal crashed
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        should_crash = (crash_boundary == "detached" and destination == "detached") or (
            crash_boundary == "anchor-retired" and destination == "destroyed-anchor"
        )
        if should_crash and not crashed:
            crashed = True
            raise SimulatedCrash

    def crash_after_truncate(descriptor: int, length: int) -> None:
        nonlocal crashed
        original_ftruncate(descriptor, length)
        if crash_boundary == "truncated" and length == 0 and not crashed:
            crashed = True
            raise SimulatedCrash

    monkeypatch.setattr(os, "rename", crash_after_rename)
    monkeypatch.setattr(os, "ftruncate", crash_after_truncate)
    with pytest.raises(SimulatedCrash):
        store.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )
    assert crashed

    target = root / "scopes" / storage_scope("acme", "research") / "results" / f"crash-{crash_boundary}.result"
    assert not target.exists()
    transactions = list((root / ".generations").glob("*/.delete-v1-*"))
    assert len(transactions) == 1
    assert (transactions[0] / "detached").stat().st_nlink == 2
    assert not list((root / "scopes").rglob(".eyvan-delete-*"))
    inventory = store.list_scope(org_id="acme", project_id="research")
    assert inventory.objects == ()

    monkeypatch.setattr(os, "rename", original_rename)
    monkeypatch.setattr(os, "ftruncate", original_ftruncate)
    restarted = LocalManagedArtifactStore(root)
    rewritten = restarted.put_versioned(uri, b"generation-b", org_id="acme", project_id="research")

    assert rewritten.created is True
    assert rewritten.metadata.object_version != written.metadata.object_version
    assert restarted.get(uri, org_id="acme", project_id="research") == b"generation-b"
    assert not list((root / ".generations").glob("*/.delete-v1-*"))
    assert not list((root / ".generations").glob("*/.trash-v1-*"))


def test_local_delete_retry_finishes_a_post_detach_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedCrash(BaseException):
        pass

    store = LocalManagedArtifactStore(tmp_path / "managed")
    uri = result_uri("retry-post-detach")
    written = store.put_versioned(uri, b"generation-a", org_id="acme", project_id="research")
    original_rename = os.rename
    crashed = False

    def crash_after_detach(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal crashed
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if destination == "detached" and not crashed:
            crashed = True
            raise SimulatedCrash

    monkeypatch.setattr(os, "rename", crash_after_detach)
    with pytest.raises(SimulatedCrash):
        store.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )

    monkeypatch.setattr(os, "rename", original_rename)
    outcome = LocalManagedArtifactStore(store.root).delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=written.metadata.object_version,
    )

    assert outcome.status is ManagedArtifactDeleteStatus.DELETED
    assert list((store.root / ".generations").iterdir()) == []


def test_local_delete_reuse_keeps_inode_consumption_bounded(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    store = LocalManagedArtifactStore(root)
    uri = result_uri("bounded-receipts")

    for cycle in range(100):
        written = store.put_versioned(
            uri,
            f"generation-{cycle}".encode(),
            org_id="acme",
            project_id="research",
        )
        deleted = store.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )
        assert deleted.status is ManagedArtifactDeleteStatus.DELETED

    assert list((root / ".generations").iterdir()) == []
    physical_entries = (root, *root.rglob("*"))
    allocated_inodes = {(entry.lstat().st_dev, entry.lstat().st_ino) for entry in physical_entries}
    assert len(allocated_inodes) <= 8


@pytest.mark.parametrize(
    "crash_boundary",
    (
        "archive-renamed",
        "archive-synced",
        "cleanup-safe-unlinked",
        "complete-unlinked",
        "destroyed-anchor-unlinked",
        "detached-unlinked",
        "validated-unlinked",
        "receipt-synced",
        "receipt-removed",
        "generation-synced",
    ),
)
@pytest.mark.parametrize("retry_operation", ("delete", "put"))
def test_local_delete_cleanup_recovers_after_every_durable_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_boundary: str,
    retry_operation: str,
) -> None:
    class SimulatedCrash(BaseException):
        pass

    root = tmp_path / f"managed-{crash_boundary}-{retry_operation}"
    store = LocalManagedArtifactStore(root)
    uri = result_uri("cleanup-crash")
    written = store.put_versioned(uri, b"generation-a", org_id="acme", project_id="research")
    original_rename = os.rename
    original_unlink = os.unlink
    original_rmdir = os.rmdir
    original_fsync = os.fsync
    archived = False
    archive_synced = False
    receipt_removed = False
    crashed = False

    def crash(event: str) -> None:
        nonlocal crashed
        if event == crash_boundary and not crashed:
            crashed = True
            raise SimulatedCrash

    def boundary_rename(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal archived
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if destination.startswith(".trash-"):
            archived = True
            crash("archive-renamed")

    def boundary_unlink(path: str, *, dir_fd: int | None = None) -> None:
        original_unlink(path, dir_fd=dir_fd)
        if archived:
            crash(f"{os.fspath(path)}-unlinked")

    def boundary_rmdir(path: str, *, dir_fd: int | None = None) -> None:
        nonlocal receipt_removed
        original_rmdir(path, dir_fd=dir_fd)
        if archived and os.fspath(path).startswith(".trash-"):
            receipt_removed = True
            crash("receipt-removed")

    def boundary_fsync(descriptor: int) -> None:
        nonlocal archive_synced
        original_fsync(descriptor)
        if receipt_removed:
            crash("generation-synced")
        elif archived and not archive_synced:
            archive_synced = True
            crash("archive-synced")
        elif archived:
            crash("receipt-synced")

    monkeypatch.setattr(os, "rename", boundary_rename)
    monkeypatch.setattr(os, "unlink", boundary_unlink)
    monkeypatch.setattr(os, "rmdir", boundary_rmdir)
    monkeypatch.setattr(os, "fsync", boundary_fsync)
    with pytest.raises(SimulatedCrash):
        store.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )
    assert crashed

    monkeypatch.setattr(os, "rename", original_rename)
    monkeypatch.setattr(os, "unlink", original_unlink)
    monkeypatch.setattr(os, "rmdir", original_rmdir)
    monkeypatch.setattr(os, "fsync", original_fsync)
    restarted = LocalManagedArtifactStore(root)
    if retry_operation == "delete":
        retried = restarted.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )
        expected = (
            ManagedArtifactDeleteStatus.ABSENT
            if crash_boundary in {"receipt-removed", "generation-synced"}
            else ManagedArtifactDeleteStatus.DELETED
        )
        assert retried.status is expected
    else:
        replacement = restarted.put_versioned(
            uri,
            b"generation-b",
            org_id="acme",
            project_id="research",
        )
        assert replacement.created is True
        stale_delete = restarted.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )
        assert stale_delete.status is ManagedArtifactDeleteStatus.VERSION_MISMATCH
        assert stale_delete.current_metadata == replacement.metadata
        assert restarted.get(uri, org_id="acme", project_id="research") == b"generation-b"

    assert not list((root / ".generations").glob("*/.delete-*"))
    assert not list((root / ".generations").glob("*/.trash-*"))
    if retry_operation == "delete":
        assert list((root / ".generations").iterdir()) == []
    else:
        assert list((root / ".generations").glob("*/*.artifact"))


class _SimulatedArchivedReceiptCrash(BaseException):
    pass


def _leave_archived_local_delete_receipt(
    store: LocalManagedArtifactStore,
    uri: str,
    object_version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    original_rename = os.rename

    def crash_after_archive(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if destination.startswith(".trash-"):
            raise _SimulatedArchivedReceiptCrash

    monkeypatch.setattr(os, "rename", crash_after_archive)
    with pytest.raises(_SimulatedArchivedReceiptCrash):
        store.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=object_version,
        )
    monkeypatch.setattr(os, "rename", original_rename)
    receipts = list((store.root / ".generations").glob("*/.trash-v1-*"))
    assert len(receipts) == 1
    return receipts[0]


def test_local_delete_cleanup_retains_payload_receipts_without_cleanup_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    uri = result_uri("legacy-unvalidated-receipt")
    written = store.put_versioned(uri, b"generation-a", org_id="acme", project_id="research")
    receipt = _leave_archived_local_delete_receipt(store, uri, written.metadata.object_version, monkeypatch)
    (receipt / "cleanup-safe").unlink()
    retained = receipt / "preserved-anchor"
    retained.write_bytes(b"unvalidated-scope-replacement")

    outcome = store.delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=written.metadata.object_version,
    )

    assert outcome.status is ManagedArtifactDeleteStatus.DELETED
    assert receipt.is_dir()
    assert retained.read_bytes() == b"unvalidated-scope-replacement"


@pytest.mark.parametrize(
    "unexpected_kind",
    ("unknown-file", "unknown-directory", "unknown-symlink", "known-directory"),
)
def test_local_delete_cleanup_retains_unexpected_receipt_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unexpected_kind: str,
) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    uri = result_uri(f"unexpected-receipt-{unexpected_kind}")
    written = store.put_versioned(uri, b"generation-a", org_id="acme", project_id="research")
    receipt = _leave_archived_local_delete_receipt(store, uri, written.metadata.object_version, monkeypatch)

    if unexpected_kind == "unknown-file":
        unexpected = receipt / "unknown"
        unexpected.write_bytes(b"retain")
    elif unexpected_kind == "unknown-directory":
        unexpected = receipt / "unknown"
        unexpected.mkdir()
        (unexpected / "do-not-traverse").write_bytes(b"retain")
    elif unexpected_kind == "unknown-symlink":
        outside = tmp_path / "outside"
        outside.write_bytes(b"retain")
        unexpected = receipt / "unknown"
        unexpected.symlink_to(outside)
    else:
        unexpected = receipt / "detached"
        unexpected.unlink()
        unexpected.mkdir()
        (unexpected / "do-not-traverse").write_bytes(b"retain")

    with pytest.raises(ManagedArtifactDeleteInconclusiveError, match=r"unexpected entry|nested directory"):
        store.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )

    assert receipt.is_dir()
    assert os.path.lexists(unexpected)
    if unexpected_kind == "unknown-symlink":
        assert outside.read_bytes() == b"retain"
    else:
        assert (receipt / "complete").is_file()


@pytest.mark.parametrize("operation", ("delete", "put"))
def test_local_generation_recovery_has_an_explicit_fixed_scan_cap(tmp_path: Path, operation: str) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    uri = result_uri("bounded-generation-scan")
    written = store.put_versioned(uri, b"generation-a", org_id="acme", project_id="research")
    generation_directory = next((store.root / ".generations").iterdir())
    for index in range(256):
        (generation_directory / f"ignored-{index:03d}").touch()

    expected_error = (
        ManagedArtifactDeleteInconclusiveError if operation == "delete" else ManagedArtifactWriteInconclusiveError
    )
    with pytest.raises(expected_error, match="bounded scan limit"):
        if operation == "delete":
            store.delete_versioned(
                uri,
                org_id="acme",
                project_id="research",
                object_version=written.metadata.object_version,
            )
        else:
            store.put_versioned(uri, b"generation-a", org_id="acme", project_id="research")

    assert store.get(uri, org_id="acme", project_id="research") == b"generation-a"


def test_local_delete_receipt_recovery_has_an_explicit_fixed_scan_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    uri = result_uri("bounded-receipt-scan")
    written = store.put_versioned(uri, b"generation-a", org_id="acme", project_id="research")
    receipt = _leave_archived_local_delete_receipt(store, uri, written.metadata.object_version, monkeypatch)
    for index in range(17 - len(list(receipt.iterdir()))):
        (receipt / f"unknown-{index:02d}").touch()

    with pytest.raises(ManagedArtifactDeleteInconclusiveError, match="bounded scan limit"):
        store.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )

    assert receipt.is_dir()
    assert len(list(receipt.iterdir())) == 17


def test_local_delete_recovery_never_removes_a_later_canonical_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedCrash(BaseException):
        pass

    store = LocalManagedArtifactStore(tmp_path / "managed")
    uri = result_uri("crash-then-replacement")
    written = store.put_versioned(uri, b"generation-a", org_id="acme", project_id="research")
    target = store.root / "scopes" / storage_scope("acme", "research") / "results" / "crash-then-replacement.result"
    original_rename = os.rename

    def crash_after_detach(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if destination == "detached":
            raise SimulatedCrash

    monkeypatch.setattr(os, "rename", crash_after_detach)
    with pytest.raises(SimulatedCrash):
        store.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )

    monkeypatch.setattr(os, "rename", original_rename)
    target.write_bytes(b"generation-b")
    outcome = LocalManagedArtifactStore(store.root).delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=written.metadata.object_version,
    )

    assert outcome.status is ManagedArtifactDeleteStatus.VERSION_MISMATCH
    assert outcome.current_metadata is not None
    assert outcome.current_metadata.sha256 == hashlib.sha256(b"generation-b").hexdigest()
    assert target.read_bytes() == b"generation-b"


def test_local_delete_receipt_special_entries_are_unlinked_nonblocking_in_a_bounded_subprocess(
    tmp_path: Path,
) -> None:
    script = r"""
import os
import socket
import sys
from pathlib import Path

from eyvan.artifacts.managed import LocalManagedArtifactStore, ManagedArtifactDeleteStatus, result_uri


class SimulatedCrash(BaseException):
    pass


root = Path(sys.argv[1])
store = LocalManagedArtifactStore(root)


def leave_receipt(name, payload):
    uri = result_uri(name)
    written = store.put_versioned(uri, payload, org_id="acme", project_id="research")
    original_rename = os.rename

    def crash_after_archive(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
        original_rename(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        if destination.startswith(".trash-"):
            raise SimulatedCrash

    os.rename = crash_after_archive
    try:
        try:
            store.delete_versioned(
                uri,
                org_id="acme",
                project_id="research",
                object_version=written.metadata.object_version,
            )
        except SimulatedCrash:
            pass
        else:
            raise AssertionError("delete reached past the archived receipt boundary")
    finally:
        os.rename = original_rename
    receipts = list(root.glob(".generations/*/.trash-v1-*"))
    if len(receipts) != 1:
        raise AssertionError(f"expected one receipt, found {len(receipts)}")
    return uri, written.metadata.object_version, receipts[0]


for kind in ("fifo", "socket", "symlink"):
    uri, object_version, receipt = leave_receipt(f"special-receipt-{kind}", kind.encode())
    detached = receipt / "detached"
    detached.unlink()
    listener = None
    outside = root / f"outside-{kind}"
    if kind == "fifo":
        os.mkfifo(detached)
    elif kind == "socket":
        listener = socket.socket(socket.AF_UNIX)
        previous = Path.cwd()
        os.chdir(receipt)
        try:
            listener.bind("detached")
        finally:
            os.chdir(previous)
    else:
        outside.write_bytes(b"retain")
        detached.symlink_to(outside)

    outcome = store.delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=object_version,
    )
    if outcome.status is not ManagedArtifactDeleteStatus.DELETED:
        raise AssertionError(f"unexpected retry outcome: {outcome.status}")
    if receipt.exists():
        raise AssertionError("special receipt was retained")
    if kind == "symlink" and outside.read_bytes() != b"retain":
        raise AssertionError("receipt symlink target was followed")
    if listener is not None:
        listener.close()

print("bounded")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "special-receipts")],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
        cwd=Path(__file__).parents[1],
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "bounded"


def test_local_special_file_opens_are_nonblocking_and_type_safe_in_a_bounded_subprocess(tmp_path: Path) -> None:
    script = r"""
import os
import socket
import stat
import sys
from pathlib import Path

from eyvan.artifacts.managed import (
    InvalidManagedArtifactURI,
    LocalManagedArtifactStore,
    ManagedArtifactDeleteInconclusiveError,
    result_uri,
    storage_scope,
)

root = Path(sys.argv[1])
store = LocalManagedArtifactStore(root)
uri = result_uri("special")
store.put(uri, b"payload", org_id="acme", project_id="research")
target = root / "scopes" / storage_scope("acme", "research") / "results" / "special.result"

target.unlink()
os.mkfifo(target)
try:
    store.stat(uri, org_id="acme", project_id="research")
except InvalidManagedArtifactURI:
    pass
else:
    raise AssertionError("FIFO was accepted")
target.unlink()

store.put(uri, b"payload", org_id="acme", project_id="research")
target.unlink()
previous = Path.cwd()
os.chdir(target.parent)
listener = socket.socket(socket.AF_UNIX)
try:
    listener.bind(target.name)
finally:
    os.chdir(previous)
try:
    store.stat(uri, org_id="acme", project_id="research")
except InvalidManagedArtifactURI:
    pass
else:
    raise AssertionError("socket was accepted")
listener.close()
target.unlink()

store.put(uri, b"payload", org_id="acme", project_id="research")
original_open = os.open

def replace_open_with_device(path, flags, *args, **kwargs):
    if path == target.name and kwargs.get("dir_fd") is not None:
        return original_open(os.devnull, flags)
    return original_open(path, flags, *args, **kwargs)

os.open = replace_open_with_device
try:
    try:
        store.stat(uri, org_id="acme", project_id="research")
    except InvalidManagedArtifactURI:
        pass
    else:
        raise AssertionError("device was accepted")
finally:
    os.open = original_open

race_uri = result_uri("race-special")
race_write = store.put_versioned(race_uri, b"payload", org_id="acme", project_id="research")
race_target = root / "scopes" / storage_scope("acme", "research") / "results" / "race-special.result"
race_fifo = race_target.with_name("race-special.fifo")
os.mkfifo(race_fifo)
original_rename = os.rename
replaced = False

def replace_before_detach(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
    global replaced
    if source == race_target.name and not replaced:
        replaced = True
        original_rename(
            race_fifo.name,
            race_target.name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=src_dir_fd,
        )
    original_rename(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

os.rename = replace_before_detach
try:
    try:
        store.delete_versioned(
            race_uri,
            org_id="acme",
            project_id="research",
            object_version=race_write.metadata.object_version,
        )
    except ManagedArtifactDeleteInconclusiveError:
        pass
    else:
        raise AssertionError("racing FIFO was accepted or deleted")
finally:
    os.rename = original_rename
assert replaced and stat.S_ISFIFO(os.lstat(race_target).st_mode)
print("bounded")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "special-root")],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
        cwd=Path(__file__).parents[1],
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "bounded"


def test_local_store_rejects_an_unrecognized_hardlink_without_deleting_it(tmp_path: Path) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    uri = result_uri("hardlink")
    written = store.put_versioned(uri, b"managed", org_id="acme", project_id="research")
    target = store.root / "scopes" / storage_scope("acme", "research") / "results" / "hardlink.result"
    outside = target.with_name("outside")
    target.unlink()
    outside.write_bytes(b"replacement")
    os.link(outside, target)

    with pytest.raises(InvalidManagedArtifactURI, match="unrecognized hard link"):
        store.stat(uri, org_id="acme", project_id="research")
    with pytest.raises(ManagedArtifactDeleteInconclusiveError):
        store.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )

    assert target.read_bytes() == b"replacement"
    assert outside.read_bytes() == b"replacement"


class _BlockingLocalManagedArtifactStore(LocalManagedArtifactStore):
    def __init__(self, root: Path):
        super().__init__(root)
        self.pause_next_lock = False
        self.lock_entered = threading.Event()
        self.release_lock = threading.Event()

    @contextlib.contextmanager
    def _object_lock(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Iterator[None]:
        with super()._object_lock(*args, **kwargs):
            if self.pause_next_lock:
                self.pause_next_lock = False
                self.lock_entered.set()
                assert self.release_lock.wait(timeout=5)
            yield


def test_local_cross_store_lock_serializes_conditional_delete_and_put(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    deleting_store = _BlockingLocalManagedArtifactStore(root)
    writing_store = LocalManagedArtifactStore(root)
    uri = result_uri("serialized")
    initial = deleting_store.put_versioned(uri, b"payload", org_id="acme", project_id="research")
    deleting_store.pause_next_lock = True
    write_started = threading.Event()

    def rewrite() -> tp.Any:
        write_started.set()
        return writing_store.put_versioned(uri, b"payload", org_id="acme", project_id="research")

    with ThreadPoolExecutor(max_workers=2) as executor:
        delete_future = executor.submit(
            deleting_store.delete_versioned,
            uri,
            org_id="acme",
            project_id="research",
            object_version=initial.metadata.object_version,
        )
        assert deleting_store.lock_entered.wait(timeout=5)
        write_future = executor.submit(rewrite)
        assert write_started.wait(timeout=5)
        assert not write_future.done()
        deleting_store.release_lock.set()
        deleted = delete_future.result(timeout=5)
        rewritten = write_future.result(timeout=5)

    assert deleted.status is ManagedArtifactDeleteStatus.DELETED
    assert rewritten.created is True
    assert rewritten.metadata.object_version != initial.metadata.object_version
    assert writing_store.get(uri, org_id="acme", project_id="research") == b"payload"


class FakeGcloudRunner:
    """Generation-aware gcloud and exact-current JSON API fake."""

    def __init__(self):
        self.objects: dict[str, tuple[bytes, int]] = {}
        self.calls: list[list[str]] = []
        self.timeouts: list[float] = []
        self.inventory_calls: list[str] = []
        self.inventory_failures: list[tuple[int, bytes] | Exception] = []
        self.cp_barrier: threading.Barrier | None = None
        self.cp_timeout_after_commit = False
        self.cp_post_stat_failure = False
        self.cat_observed: threading.Event | None = None
        self.cat_release: threading.Event | None = None
        self.cat_post_stat_failure = False
        self.cat_timeout = False
        self.rm_failure: str | None = None
        self.rm_timeout_after_delete = False
        self._next_generation = 40
        self._lock = threading.Lock()

    def replace(self, uri: str, payload: bytes) -> int:
        with self._lock:
            self._next_generation += 1
            self.objects[uri] = (payload, self._next_generation)
            return self._next_generation

    def __call__(
        self,
        args: tp.Sequence[str],
        *,
        check: bool,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        del check, capture_output
        command = list(args)
        self.calls.append(command)
        self.timeouts.append(timeout)
        uri = command[-1]
        if command[2] == "cp":
            assert "--if-generation-match=0" in command
            assert "--no-clobber" not in command
            if self.cp_barrier is not None:
                self.cp_barrier.wait()
            with self._lock:
                if uri in self.objects:
                    return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"provider-secret")
                self._next_generation += 1
                self.objects[uri] = (Path(command[-2]).read_bytes(), self._next_generation)
            if self.cp_post_stat_failure:
                self.inventory_failures.append((403, b'{"error":"provider-secret"}'))
            if self.cp_timeout_after_commit:
                raise subprocess.TimeoutExpired(command, timeout)
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        if command[2] == "cat":
            if self.cat_timeout:
                raise subprocess.TimeoutExpired(command, timeout, stderr=b"provider-secret")
            with self._lock:
                current = self.objects.get(uri)
            if current is None:
                return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"provider-secret")
            if self.cat_observed is not None:
                self.cat_observed.set()
                assert self.cat_release is not None and self.cat_release.wait(timeout=5)
            if self.cat_post_stat_failure:
                self.inventory_failures.append((403, b'{"error":"provider-secret"}'))
            return subprocess.CompletedProcess(command, 0, stdout=current[0], stderr=b"")
        if command[2] == "rm":
            assert any(argument.startswith("--if-generation-match=") for argument in command)
            if self.rm_failure == "permission":
                return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"provider-secret")
            if self.rm_failure == "timeout":
                raise subprocess.TimeoutExpired(command, timeout, stderr=b"provider-secret")
            expected = int(next(argument.split("=", 1)[1] for argument in command if argument.startswith("--if-")))
            with self._lock:
                current = self.objects.get(uri)
                if current is None or current[1] != expected:
                    return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"provider-secret")
                del self.objects[uri]
            if self.rm_timeout_after_delete:
                raise subprocess.TimeoutExpired(command, timeout, stderr=b"provider-secret")
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        raise AssertionError(command)

    def inventory(
        self,
        url: str,
        *,
        headers: tp.Mapping[str, str],
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> tuple[int, bytes]:
        del headers, timeout_seconds, maximum_response_bytes
        self.inventory_calls.append(url)
        if self.inventory_failures:
            failure = self.inventory_failures.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return failure
        parsed = urlsplit(url)
        encoded_bucket, separator, encoded_name = parsed.path.partition("/o/")
        assert separator
        bucket = unquote(encoded_bucket.rsplit("/b/", 1)[1])
        name = unquote(encoded_name)
        uri = f"gs://{bucket}/{name}"
        with self._lock:
            current = self.objects.get(uri)
        if current is None:
            return 404, b'{"error":{"message":"not found"}}'
        payload, generation = current
        return 200, json.dumps(
            {"name": name, "size": str(len(payload)), "generation": str(generation)},
            separators=(",", ":"),
        ).encode()

    def store(self) -> GCSManagedArtifactStore:
        return GCSManagedArtifactStore(
            "gs://eyvan-managed/artifacts",
            runner=self,
            inventory_transport=self.inventory,
            access_token_provider=lambda: "fake-access-token",
        )


def test_gcs_store_uses_conditional_create_and_returns_the_stable_generation() -> None:
    runner = FakeGcloudRunner()
    store = runner.store()
    uri = result_uri("run-1")

    created = store.put_versioned(uri, b"result", org_id="acme", project_id="research")
    retried = store.put_versioned(uri, b"result", org_id="acme", project_id="research")

    object_uri = next(iter(runner.objects))
    assert object_uri == f"gs://eyvan-managed/artifacts/scopes/{storage_scope('acme', 'research')}/results/run-1.result"
    assert created.created is True
    assert retried.created is False
    assert retried.metadata == created.metadata
    assert created.metadata.sha256 == hashlib.sha256(b"result").hexdigest()
    assert runner.timeouts and set(runner.timeouts) == {30.0}
    cp = next(command for command in runner.calls if command[2] == "cp")
    assert "--if-generation-match=0" in cp
    assert "--no-clobber" not in cp
    assert not store.exists(uri, org_id="acme", project_id="other")
    with pytest.raises(ManagedArtifactConflictError):
        store.put(uri, b"changed", org_id="acme", project_id="research")


class _CountingTokens:
    """An ambient credential source that hands out a new token per call."""

    def __init__(self) -> None:
        self.issued: list[str] = []
        self._lock = threading.Lock()

    def __call__(self) -> str:
        with self._lock:
            token = f"token-{len(self.issued)}"
            self.issued.append(token)
            return token


def _store_with_tokens(runner: FakeGcloudRunner, tokens: _CountingTokens, seen: list[str]) -> GCSManagedArtifactStore:
    def inventory(url: str, *, headers: tp.Mapping[str, str], **kwargs: tp.Any) -> tuple[int, bytes]:
        seen.append(headers["Authorization"])
        return runner.inventory(url, headers=headers, **kwargs)

    return GCSManagedArtifactStore(
        "gs://eyvan-managed/artifacts",
        runner=runner,
        inventory_transport=inventory,
        access_token_provider=tokens,
    )


def test_gcs_store_reuses_one_ambient_token_across_a_write_and_read() -> None:
    """Each stat used to spawn `gcloud auth print-access-token`: ~4 per write."""

    runner = FakeGcloudRunner()
    tokens = _CountingTokens()
    seen: list[str] = []
    store = _store_with_tokens(runner, tokens, seen)
    uri = result_uri("token-reuse")

    store.put_versioned(uri, b"payload", org_id="acme", project_id="research")
    assert store.get(uri, org_id="acme", project_id="research") == b"payload"

    assert len(seen) >= 4
    assert tokens.issued == ["token-0"]
    assert set(seen) == {"Bearer token-0"}


def test_gcs_store_refreshes_a_rejected_token_once_and_retries() -> None:
    runner = FakeGcloudRunner()
    tokens = _CountingTokens()
    seen: list[str] = []
    store = _store_with_tokens(runner, tokens, seen)
    uri = result_uri("token-refresh")
    store.put_versioned(uri, b"payload", org_id="acme", project_id="research")
    seen.clear()

    runner.inventory_failures.append((401, b'{"error":"provider-secret"}'))
    metadata = store.stat(uri, org_id="acme", project_id="research")

    assert metadata is not None and metadata.size_bytes == len(b"payload")
    assert seen == ["Bearer token-0", "Bearer token-1"]
    # The fresh token is what later requests reuse.
    store.stat(uri, org_id="acme", project_id="research")
    assert seen[-1] == "Bearer token-1"
    assert tokens.issued == ["token-0", "token-1"]


def test_gcs_store_reports_a_second_rejection_without_looping() -> None:
    from eyvan.artifacts.managed import ManagedArtifactInventoryError

    runner = FakeGcloudRunner()
    tokens = _CountingTokens()
    seen: list[str] = []
    store = _store_with_tokens(runner, tokens, seen)
    runner.inventory_failures.extend([(401, b'{"error":"provider-secret"}')] * 2)

    with pytest.raises(ManagedArtifactInventoryError) as rejected:
        store.stat(result_uri("token-rejected"), org_id="acme", project_id="research")

    assert str(rejected.value) == "GCS inventory request failed with HTTP status 401"
    assert "provider-secret" not in str(rejected.value)
    assert len(seen) == 2
    assert tokens.issued == ["token-0", "token-1"]


class _JsonUploads:
    """A JSON API upload endpoint backed by the fake runner's object table."""

    def __init__(self, runner: FakeGcloudRunner) -> None:
        self.runner = runner
        self.calls: list[tuple[str, str]] = []
        self.answers: list[tuple[int, bytes] | Exception] = []
        self.md5_override: str | None = None
        self.omit_md5 = False

    def __call__(
        self,
        url: str,
        *,
        headers: tp.Mapping[str, str],
        content: bytes,
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> tuple[int, bytes]:
        del timeout_seconds, maximum_response_bytes
        self.calls.append((url, headers["Authorization"]))
        if self.answers:
            answer = self.answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        parsed = urlsplit(url)
        assert parsed.path.startswith("/upload/storage/v1/b/") and parsed.path.endswith("/o")
        query = dict(part.split("=", 1) for part in parsed.query.split("&"))
        assert query["uploadType"] == "media" and query["ifGenerationMatch"] == "0"
        bucket = unquote(parsed.path.split("/b/", 1)[1].rsplit("/o", 1)[0])
        name = unquote(query["name"])
        uri = f"gs://{bucket}/{name}"
        with self.runner._lock:
            if uri in self.runner.objects:
                return 412, b'{"error":"provider-secret"}'
            self.runner._next_generation += 1
            self.runner.objects[uri] = (content, self.runner._next_generation)
            generation = self.runner._next_generation
        document: dict[str, str] = {"name": name, "size": str(len(content)), "generation": str(generation)}
        if not self.omit_md5:
            document["md5Hash"] = self.md5_override or base64.b64encode(hashlib.md5(content).digest()).decode()
        return 200, json.dumps(document).encode()


def _json_upload_store(runner: FakeGcloudRunner) -> tuple[GCSManagedArtifactStore, _JsonUploads]:
    uploads = _JsonUploads(runner)
    store = GCSManagedArtifactStore(
        "gs://eyvan-managed/artifacts",
        runner=runner,
        inventory_transport=runner.inventory,
        access_token_provider=lambda: "fake-access-token",
        upload_transport=uploads,
    )
    return store, uploads


def _gcloud_verbs(runner: FakeGcloudRunner) -> list[str]:
    return [command[2] for command in runner.calls]


def test_a_json_upload_with_a_matching_checksum_is_not_read_back() -> None:
    """One JSON request replaced `gcloud storage cp` and the full `cat` that verified it."""

    runner = FakeGcloudRunner()
    store, uploads = _json_upload_store(runner)
    uri = result_uri("json-upload")

    written = store.put_versioned(uri, b"payload", org_id="acme", project_id="research")

    assert written.created is True
    assert written.metadata.sha256 == hashlib.sha256(b"payload").hexdigest()
    assert written.metadata.size_bytes == len(b"payload")
    assert len(uploads.calls) == 1
    assert _gcloud_verbs(runner) == []  # no cp, no cat
    # The stored generation is the one reported, and reads still see the bytes.
    current = store.stat(uri, org_id="acme", project_id="research")
    assert current is not None and current.object_version == written.metadata.object_version
    assert store.get(uri, org_id="acme", project_id="research") == b"payload"
    # Retrying the identical write is still recognised as already present.
    assert store.put_versioned(uri, b"payload", org_id="acme", project_id="research").created is False


@pytest.mark.parametrize("disagreement", ["md5", "absent-md5"])
def test_a_json_upload_whose_checksum_does_not_prove_the_bytes_falls_back_to_an_exact_read(disagreement: str) -> None:
    runner = FakeGcloudRunner()
    store, uploads = _json_upload_store(runner)
    if disagreement == "md5":
        uploads.md5_override = base64.b64encode(b"\0" * 16).decode()
    else:
        uploads.omit_md5 = True

    written = store.put_versioned(result_uri(f"json-{disagreement}"), b"payload", org_id="acme", project_id="research")

    assert written.created is True
    assert written.metadata.sha256 == hashlib.sha256(b"payload").hexdigest()
    assert _gcloud_verbs(runner) == ["cat"]


def test_a_json_upload_that_lost_a_create_race_is_settled_by_the_bytes() -> None:
    runner = FakeGcloudRunner()
    store, uploads = _json_upload_store(runner)
    uri = result_uri("json-race")
    object_uri = f"gs://eyvan-managed/artifacts/scopes/{storage_scope('acme', 'research')}/results/json-race.result"
    real_upload = uploads.__call__

    def racing(url: str, **kwargs: tp.Any) -> tuple[int, bytes]:
        # Another writer lands identical bytes just before this create.
        runner.replace(object_uri, kwargs["content"])
        return real_upload(url, **kwargs)

    store._upload_transport = racing  # type: ignore[assignment]
    identical = store.put_versioned(uri, b"same", org_id="acme", project_id="research")
    assert identical.created is None
    assert identical.metadata.sha256 == hashlib.sha256(b"same").hexdigest()

    def conflicting(url: str, **kwargs: tp.Any) -> tuple[int, bytes]:
        runner.replace(object_uri.replace("json-race", "json-race-2"), b"other")
        return real_upload(url, **kwargs)

    store._upload_transport = conflicting  # type: ignore[assignment]
    with pytest.raises(ManagedArtifactConflictError):
        store.put_versioned(result_uri("json-race-2"), b"mine", org_id="acme", project_id="research")


def test_a_json_upload_transport_failure_is_inconclusive_not_trusted() -> None:
    runner = FakeGcloudRunner()
    store, uploads = _json_upload_store(runner)
    uploads.answers.append(TimeoutError())

    with pytest.raises(ManagedArtifactWriteInconclusiveError) as inconclusive:
        store.put_versioned(result_uri("json-timeout"), b"payload", org_id="acme", project_id="research")

    assert str(inconclusive.value) == "GCS managed artifact write was inconclusive"
    assert not runner.objects


def test_a_json_upload_retries_a_rejected_token_once() -> None:
    runner = FakeGcloudRunner()
    tokens = _CountingTokens()
    uploads = _JsonUploads(runner)
    store = GCSManagedArtifactStore(
        "gs://eyvan-managed/artifacts",
        runner=runner,
        inventory_transport=runner.inventory,
        access_token_provider=tokens,
        upload_transport=uploads,
    )
    uploads.answers.append((401, b'{"error":"provider-secret"}'))

    written = store.put_versioned(result_uri("json-401"), b"payload", org_id="acme", project_id="research")

    assert written.created is True
    assert [authorization for _url, authorization in uploads.calls] == ["Bearer token-0", "Bearer token-1"]
    assert _gcloud_verbs(runner) == []


class _JsonDownloads:
    """A generation-pinned JSON API media endpoint backed by the fake runner's objects."""

    def __init__(self, runner: FakeGcloudRunner) -> None:
        self.runner = runner
        self.calls: list[tuple[str, str]] = []
        self.answers: list[tuple[int, dict[str, str], bytes] | Exception] = []
        self.drop_headers: set[str] = set()
        self.override_headers: dict[str, str] = {}
        self.corrupt_body = False

    def __call__(
        self, url: str, *, headers: tp.Mapping[str, str], timeout_seconds: float
    ) -> tuple[int, dict[str, str], bytes]:
        del timeout_seconds
        self.calls.append((url, headers["Authorization"]))
        if self.answers:
            answer = self.answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        assert query["alt"] == ["media"]
        encoded_bucket, _separator, encoded_name = parsed.path.partition("/o/")
        uri = f"gs://{unquote(encoded_bucket.rsplit('/b/', 1)[1])}/{unquote(encoded_name)}"
        with self.runner._lock:
            current = self.runner.objects.get(uri)
        if current is None:
            return 404, {}, b'{"error":"provider-secret"}'
        payload, generation = current
        if query["ifGenerationMatch"] != [str(generation)]:
            return 412, {}, b'{"error":"provider-secret"}'
        response_headers = {
            "x-goog-generation": str(generation),
            "x-goog-stored-content-length": str(len(payload)),
            "x-goog-stored-content-encoding": "identity",
            "x-goog-hash": f"crc32c=AAAAAA==,md5={base64.b64encode(hashlib.md5(payload).digest()).decode()}",
        }
        for name in self.drop_headers:
            response_headers.pop(name, None)
        response_headers.update(self.override_headers)
        return 200, response_headers, (payload + b"!" if self.corrupt_body else payload)


def _json_read_store(runner: FakeGcloudRunner) -> tuple[GCSManagedArtifactStore, _JsonDownloads]:
    downloads = _JsonDownloads(runner)
    store = GCSManagedArtifactStore(
        "gs://eyvan-managed/artifacts",
        runner=runner,
        inventory_transport=runner.inventory,
        access_token_provider=lambda: "fake-access-token",
        download_transport=downloads,
    )
    return store, downloads


def test_an_exact_read_is_one_generation_pinned_request() -> None:
    """The stat / `gcloud storage cat` / stat sandwich cost seconds per archived rank."""

    runner = FakeGcloudRunner()
    store, downloads = _json_read_store(runner)
    uri = result_uri("json-read")
    written = store.put_versioned(uri, b"payload", org_id="acme", project_id="research")
    runner.calls.clear()
    runner.inventory_calls.clear()
    downloads.calls.clear()  # the write verified itself through the same read path

    read = store.get_versioned(uri, org_id="acme", project_id="research", object_version=written.metadata.object_version)

    assert read.payload == b"payload"
    assert read.metadata == written.metadata
    assert read.metadata.sha256 == hashlib.sha256(b"payload").hexdigest()
    assert len(downloads.calls) == 1
    assert _gcloud_verbs(runner) == []  # no cat
    assert runner.inventory_calls == []  # no stats


@pytest.mark.parametrize(
    "disagreement",
    ["no-md5", "wrong-md5", "no-length", "wrong-length", "wrong-generation", "gzip", "body", "error", "timeout"],
)
def test_an_exact_read_the_response_does_not_prove_falls_back_to_the_sandwich(disagreement: str) -> None:
    runner = FakeGcloudRunner()
    store, downloads = _json_read_store(runner)
    uri = result_uri(f"json-read-{disagreement}")
    written = store.put_versioned(uri, b"payload", org_id="acme", project_id="research")
    runner.calls.clear()
    if disagreement == "no-md5":
        downloads.override_headers["x-goog-hash"] = "crc32c=AAAAAA=="
    elif disagreement == "wrong-md5":
        downloads.override_headers["x-goog-hash"] = f"md5={base64.b64encode(b'0' * 16).decode()}"
    elif disagreement == "no-length":
        downloads.drop_headers.add("x-goog-stored-content-length")
    elif disagreement == "wrong-length":
        downloads.override_headers["x-goog-stored-content-length"] = "999"
    elif disagreement == "wrong-generation":
        downloads.override_headers["x-goog-generation"] = "1"
    elif disagreement == "gzip":
        downloads.override_headers["x-goog-stored-content-encoding"] = "gzip"
    elif disagreement == "body":
        downloads.corrupt_body = True
    elif disagreement == "error":
        downloads.answers.append((503, {}, b'{"error":"provider-secret"}'))
    else:
        downloads.answers.append(TimeoutError())

    read = store.get_versioned(uri, org_id="acme", project_id="research", object_version=written.metadata.object_version)

    assert read.payload == b"payload"
    assert read.metadata == written.metadata
    assert _gcloud_verbs(runner) == ["cat"]


def test_an_exact_read_of_a_replaced_or_absent_object_keeps_its_precise_error() -> None:
    from eyvan.artifacts.managed import ManagedArtifactVersionMismatchError

    runner = FakeGcloudRunner()
    store, _downloads = _json_read_store(runner)
    uri = result_uri("json-replaced")
    written = store.put_versioned(uri, b"payload", org_id="acme", project_id="research")
    object_uri = next(iter(runner.objects))
    runner.replace(object_uri, b"other")

    with pytest.raises(ManagedArtifactVersionMismatchError):
        store.get_versioned(uri, org_id="acme", project_id="research", object_version=written.metadata.object_version)

    del runner.objects[object_uri]
    with pytest.raises(FileNotFoundError):
        store.get_versioned(uri, org_id="acme", project_id="research", object_version=written.metadata.object_version)


def test_an_exact_read_still_rejects_content_addressed_bytes_that_disagree() -> None:
    runner = FakeGcloudRunner()
    store, _downloads = _json_read_store(runner)
    digest = hashlib.sha256(b"real").hexdigest()
    uri = callable_uri(digest)
    written = store.put_versioned(uri, b"real", org_id="acme", project_id="research")
    object_uri = next(iter(runner.objects))
    with runner._lock:
        runner.objects[object_uri] = (b"fake", runner.objects[object_uri][1])

    with pytest.raises(ManagedArtifactConflictError):
        store.get_versioned(uri, org_id="acme", project_id="research", object_version=written.metadata.object_version)


def test_an_exact_read_retries_a_rejected_token_once() -> None:
    runner = FakeGcloudRunner()
    tokens = _CountingTokens()
    downloads = _JsonDownloads(runner)
    store = GCSManagedArtifactStore(
        "gs://eyvan-managed/artifacts",
        runner=runner,
        inventory_transport=runner.inventory,
        access_token_provider=tokens,
        download_transport=downloads,
    )
    uri = result_uri("json-read-401")
    written = store.put_versioned(uri, b"payload", org_id="acme", project_id="research")
    runner.calls.clear()
    downloads.calls.clear()  # the write verified itself through the same read path
    downloads.answers.append((401, {}, b'{"error":"provider-secret"}'))

    read = store.get_versioned(uri, org_id="acme", project_id="research", object_version=written.metadata.object_version)

    assert read.payload == b"payload"
    assert [authorization for _url, authorization in downloads.calls] == ["Bearer token-0", "Bearer token-1"]
    assert _gcloud_verbs(runner) == []


def test_the_default_gcs_store_reads_through_the_json_api() -> None:
    from eyvan.artifacts.managed import _httpx_gcs_download_transport

    assert GCSManagedArtifactStore("gs://eyvan-managed/artifacts")._download_transport is _httpx_gcs_download_transport
    assert FakeGcloudRunner().store()._download_transport is None


def test_the_default_gcs_store_uploads_through_the_json_api() -> None:
    from eyvan.artifacts.managed import _httpx_gcs_upload_transport

    assert GCSManagedArtifactStore("gs://eyvan-managed/artifacts")._upload_transport is _httpx_gcs_upload_transport
    # An injected gcloud runner without an upload transport keeps the cp path.
    assert FakeGcloudRunner().store()._upload_transport is None


def test_gcs_cp_timeout_after_commit_is_resolved_by_an_exact_digest_read() -> None:
    runner = FakeGcloudRunner()
    runner.cp_timeout_after_commit = True
    store = runner.store()
    uri = result_uri("cp-timeout-after-commit")

    written = store.put_versioned(uri, b"payload", org_id="acme", project_id="research")

    assert written.created is None
    assert written.metadata.sha256 == hashlib.sha256(b"payload").hexdigest()
    assert 30.0 in runner.timeouts
    assert store.get(uri, org_id="acme", project_id="research") == b"payload"


def test_gcs_cat_timeout_is_bounded_and_sanitized() -> None:
    runner = FakeGcloudRunner()
    store = runner.store()
    uri = result_uri("cat-timeout")
    written = store.put_versioned(uri, b"payload", org_id="acme", project_id="research")
    runner.cat_timeout = True

    with pytest.raises(ManagedArtifactOperationInconclusiveError) as inconclusive:
        store.get_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )

    assert str(inconclusive.value) == "GCS managed artifact read failed"
    assert "provider-secret" not in str(inconclusive.value)
    assert runner.timeouts and set(runner.timeouts) == {30.0}


@pytest.mark.parametrize("failure_point", ("post_cp_stat", "post_cat_stat"))
def test_gcs_write_normalizes_every_post_cp_verification_failure(failure_point: str) -> None:
    runner = FakeGcloudRunner()
    if failure_point == "post_cp_stat":
        runner.cp_post_stat_failure = True
    else:
        runner.cat_post_stat_failure = True
    store = runner.store()
    uri = result_uri(f"write-{failure_point}")

    with pytest.raises(ManagedArtifactWriteInconclusiveError) as inconclusive:
        store.put_versioned(uri, b"payload", org_id="acme", project_id="research")

    assert str(inconclusive.value) == "GCS managed artifact write was inconclusive"
    assert "provider-secret" not in str(inconclusive.value)
    assert any(command[2] == "cp" for command in runner.calls)
    assert runner.timeouts and set(runner.timeouts) == {30.0}


def test_gcs_conditional_create_race_accepts_only_the_identical_current_generation() -> None:
    runner = FakeGcloudRunner()
    runner.cp_barrier = threading.Barrier(2)
    store = runner.store()
    uri = result_uri("race")

    def write() -> tp.Any:
        return store.put_versioned(uri, b"same", org_id="acme", project_id="research")

    with ThreadPoolExecutor(max_workers=2) as executor:
        writes = tuple(executor.map(lambda _index: write(), range(2)))

    assert {write.created for write in writes} == {True, None}
    assert writes[0].metadata == writes[1].metadata
    cp_calls = [command for command in runner.calls if command[2] == "cp"]
    assert len(cp_calls) == 2
    assert all("--if-generation-match=0" in command and "--no-clobber" not in command for command in cp_calls)


def test_gcs_exact_read_detects_a_replacement_during_the_byte_read() -> None:
    runner = FakeGcloudRunner()
    store = runner.store()
    uri = result_uri("read-race")
    written = store.put_versioned(uri, b"original", org_id="acme", project_id="research")
    object_uri = next(iter(runner.objects))
    runner.cat_observed = threading.Event()
    runner.cat_release = threading.Event()

    with ThreadPoolExecutor(max_workers=1) as executor:
        reading = executor.submit(
            store.get_versioned,
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )
        assert runner.cat_observed.wait(timeout=5)
        new_generation = runner.replace(object_uri, b"replacement")
        runner.cat_release.set()
        with pytest.raises(ManagedArtifactVersionMismatchError):
            reading.result(timeout=5)

    runner.cat_observed = None
    runner.cat_release = None
    current = store.get_versioned(uri, org_id="acme", project_id="research")
    assert current.payload == b"replacement"
    assert current.metadata.object_version == f"gcs:{new_generation}"
    assert current.metadata.sha256 == hashlib.sha256(b"replacement").hexdigest()


def test_gcs_conditional_delete_classifies_deleted_absent_and_version_mismatch() -> None:
    runner = FakeGcloudRunner()
    store = runner.store()
    uri = result_uri("delete")
    original = store.put_versioned(uri, b"first", org_id="acme", project_id="research")

    deleted = store.delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=original.metadata.object_version,
    )
    absent = store.delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=original.metadata.object_version,
    )
    assert deleted.status is ManagedArtifactDeleteStatus.DELETED
    assert absent.status is ManagedArtifactDeleteStatus.ABSENT

    replacement = store.put_versioned(uri, b"second", org_id="acme", project_id="research")
    mismatch = store.delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=original.metadata.object_version,
    )
    assert mismatch.status is ManagedArtifactDeleteStatus.VERSION_MISMATCH
    assert mismatch.current_metadata == replacement.metadata
    assert store.get(uri, org_id="acme", project_id="research") == b"second"
    assert store.delete(uri, org_id="acme", project_id="research") is True
    rm_calls = [command for command in runner.calls if command[2] == "rm"]
    assert rm_calls
    assert all(any(argument.startswith("--if-generation-match=") for argument in command) for command in rm_calls)


def test_gcs_rm_timeout_after_delete_is_resolved_as_absent() -> None:
    runner = FakeGcloudRunner()
    store = runner.store()
    uri = result_uri("rm-timeout-after-delete")
    written = store.put_versioned(uri, b"payload", org_id="acme", project_id="research")
    runner.rm_timeout_after_delete = True

    outcome = store.delete_versioned(
        uri,
        org_id="acme",
        project_id="research",
        object_version=written.metadata.object_version,
    )

    assert outcome.status is ManagedArtifactDeleteStatus.ABSENT
    assert runner.timeouts and set(runner.timeouts) == {30.0}


@pytest.mark.parametrize("failure", ("permission", "timeout"))
def test_gcs_conditional_delete_does_not_translate_permission_or_timeout_to_absent(failure: str) -> None:
    runner = FakeGcloudRunner()
    store = runner.store()
    uri = result_uri(f"delete-{failure}")
    written = store.put_versioned(uri, b"payload", org_id="acme", project_id="research")
    runner.rm_failure = failure

    with pytest.raises(ManagedArtifactDeleteInconclusiveError) as inconclusive:
        store.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )

    assert "provider-secret" not in str(inconclusive.value)
    assert store.get(uri, org_id="acme", project_id="research") == b"payload"
    rm = next(command for command in runner.calls if command[2] == "rm")
    assert any(argument.startswith("--if-generation-match=") for argument in rm)


def test_gcs_conditional_delete_normalizes_an_inconclusive_post_rm_stat() -> None:
    runner = FakeGcloudRunner()
    store = runner.store()
    uri = result_uri("delete-stat-permission")
    written = store.put_versioned(uri, b"payload", org_id="acme", project_id="research")
    runner.rm_failure = "permission"
    runner.inventory_failures.append((403, b'{"error":"provider-secret"}'))

    with pytest.raises(ManagedArtifactDeleteInconclusiveError) as inconclusive:
        store.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=written.metadata.object_version,
        )

    assert str(inconclusive.value) == "GCS managed artifact deletion was inconclusive"
    assert "provider-secret" not in str(inconclusive.value)
    rm = next(command for command in runner.calls if command[2] == "rm")
    assert any(argument.startswith("--if-generation-match=") for argument in rm)


@pytest.mark.parametrize(
    "object_version",
    (
        "local:1:2:3",
        "gcs:0",
        "gcs:01",
        "gcs:18446744073709551616",
    ),
)
def test_gcs_versioned_operations_reject_wrong_or_out_of_range_versions(object_version: str) -> None:
    runner = FakeGcloudRunner()
    store = runner.store()
    uri = result_uri("bad-version")

    with pytest.raises(ValueError):
        store.get_versioned(uri, org_id="acme", project_id="research", object_version=object_version)
    with pytest.raises(ValueError):
        store.delete_versioned(
            uri,
            org_id="acme",
            project_id="research",
            object_version=object_version,
        )

    assert runner.calls == []
    assert runner.inventory_calls == []


def test_transfer_token_is_url_safe_time_bounded_and_fully_scoped() -> None:
    now = [1_000.0]
    signer = ArtifactTokenSigner(b"s" * 32, default_ttl_seconds=60, clock=lambda: now[0])
    uri = result_uri("run-1")
    token = signer.issue(
        ArtifactAction.WRITE,
        uri,
        org_id="acme",
        project_id="research",
        workload_id="workload-1",
    )

    assert re.fullmatch(r"[A-Za-z0-9_.-]+", token)
    assert "=" not in token
    claims = signer.verify(
        token,
        "write",
        uri,
        org_id="acme",
        project_id="research",
        workload_id="workload-1",
    )
    assert claims.action is ArtifactAction.WRITE
    assert claims.artifact.uri == uri
    assert claims.issued_at == 1_000
    assert claims.expires_at == 1_060
    assert claims.workload_id == "workload-1"

    mismatched_scopes = (
        ("read", uri, "acme", "research", "workload-1"),
        ("write", result_uri("run-2"), "acme", "research", "workload-1"),
        ("write", uri, "other", "research", "workload-1"),
        ("write", uri, "acme", "other", "workload-1"),
        ("write", uri, "acme", "research", "workload-2"),
        ("write", uri, "acme", "research", None),
    )
    for action, expected_uri, org_id, project_id, workload_id in mismatched_scopes:
        with pytest.raises(InvalidArtifactToken, match="scope"):
            signer.verify(
                token,
                action,
                expected_uri,
                org_id=org_id,
                project_id=project_id,
                workload_id=workload_id,
            )

    now[0] = 1_060.0
    with pytest.raises(ExpiredArtifactToken, match="expired"):
        signer.verify(
            token,
            "write",
            uri,
            org_id="acme",
            project_id="research",
            workload_id="workload-1",
        )


def test_transfer_token_rejects_tampering_and_unsafe_lifetimes() -> None:
    signer = ArtifactTokenSigner(
        b"s" * 32,
        default_ttl_seconds=60,
        max_ttl_seconds=120,
        clock=lambda: 1_000,
    )
    uri = result_uri("run-1")
    token = signer.issue("read", uri, org_id="acme", project_id="research")
    replacement = "A" if token[-1] != "A" else "B"
    tampered = f"{token[:-1]}{replacement}"

    with pytest.raises(InvalidArtifactToken, match="invalid"):
        signer.verify(tampered, "read", uri, org_id="acme", project_id="research")
    with pytest.raises(ValueError, match="between 1 and 120"):
        signer.issue("read", uri, org_id="acme", project_id="research", ttl_seconds=121)
    with pytest.raises(ValueError, match="at least 32 bytes"):
        ArtifactTokenSigner(b"too-short")
    with pytest.raises(ValueError, match="workload-bound write"):
        signer.issue(
            "read",
            uri,
            org_id="acme",
            project_id="research",
            workload_id="workload-1",
            expires=False,
        )


@pytest.mark.parametrize(
    "public_url",
    (
        "https://token@eyvan.example.com",
        "https://eyvan.example.com?token=secret",
        "https://eyvan.example.com#fragment",
        "file:///tmp/eyvan.sock",
    ),
)
def test_managed_artifact_access_rejects_unsafe_public_urls(tmp_path: Path, public_url: str) -> None:
    with pytest.raises(ValueError):
        ManagedArtifactAccess(
            store=LocalManagedArtifactStore(tmp_path),
            signer=ArtifactTokenSigner(b"s" * 32),
            public_url=public_url,
            token_ttl_seconds=60,
        )


def test_managed_artifact_access_rejects_explicit_zero_ttl(tmp_path: Path) -> None:
    access = ManagedArtifactAccess(
        store=LocalManagedArtifactStore(tmp_path),
        signer=ArtifactTokenSigner(b"s" * 32),
        public_url="https://eyvan.example.com/",
        token_ttl_seconds=60,
    )

    with pytest.raises(ValueError, match="between"):
        access.callable_download_transfer(
            callable_uri("a" * 64),
            org_id="acme",
            project_id="research",
            workload_id="workload-1",
            ttl_seconds=0,
        )


def test_result_transfer_token_is_durable_but_generation_revocable(tmp_path: Path) -> None:
    now = [1_000.0]
    signer = ArtifactTokenSigner(b"s" * 32, default_ttl_seconds=60, clock=lambda: now[0])
    access = ManagedArtifactAccess(
        store=LocalManagedArtifactStore(tmp_path),
        signer=signer,
        public_url="https://eyvan.example.com",
        token_ttl_seconds=60,
    )
    uri = result_uri("workload-1-g1")
    transfer = access.result_upload_transfer(
        uri,
        org_id="acme",
        project_id="research",
        workload_id="workload-1",
    )

    now[0] += 365 * 24 * 60 * 60
    claims = signer.verify(
        transfer.token,
        ArtifactAction.WRITE,
        uri,
        org_id="acme",
        project_id="research",
        workload_id="workload-1",
    )
    assert claims.expires_at is None
    with pytest.raises(InvalidArtifactToken, match="scope"):
        signer.verify(
            transfer.token,
            ArtifactAction.WRITE,
            result_uri("workload-1-g2"),
            org_id="acme",
            project_id="research",
            workload_id="workload-1",
        )


class FakeGCSInventoryAPI:
    """Deterministic live/noncurrent model for the read-only GCS JSON API."""

    def __init__(self) -> None:
        self.objects: dict[str, list[tuple[int, int, bool]]] = {}
        self.calls: list[dict[str, tp.Any]] = []
        self.queued: list[tuple[int, bytes] | BaseException] = []
        self.server_page_size: int | None = None

    def add(self, name: str, *, size: int, generation: int, live: bool) -> None:
        versions = self.objects.setdefault(name, [])
        if live:
            assert not any(existing_live for _generation, _size, existing_live in versions)
        versions.append((generation, size, live))

    def queue(self, response: tuple[int, bytes] | BaseException) -> None:
        self.queued.append(response)

    def __call__(
        self,
        url: str,
        *,
        headers: tp.Mapping[str, str],
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> tuple[int, bytes]:
        parsed = urlsplit(url)
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
        assert parsed.scheme == "https"
        assert parsed.netloc == "storage.googleapis.com"
        assert headers == {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Authorization": "Bearer fake-access-token",
        }
        assert timeout_seconds == 30.0
        assert maximum_response_bytes == 2 * 1024 * 1024
        assert "fake-access-token" not in url
        self.calls.append(
            {
                "url": url,
                "path": parsed.path,
                "query": query,
                "timeout_seconds": timeout_seconds,
                "maximum_response_bytes": maximum_response_bytes,
            }
        )
        if self.queued:
            response = self.queued.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response

        marker = "/storage/v1/b/eyvan-managed/o"
        if parsed.path == marker:
            return self._list(query)
        if parsed.path.startswith(f"{marker}/"):
            return self._get(unquote(parsed.path[len(marker) + 1 :]), query)
        raise AssertionError(parsed.path)

    def _list(self, query: dict[str, list[str]]) -> tuple[int, bytes]:
        assert query["versions"] == ["false"]
        assert query["softDeleted"] == ["false"]
        assert query["projection"] == ["noAcl"]
        assert query["fields"] == ["items(name,size,generation),nextPageToken"]
        page_offset = int(query.get("pageToken", ["provider-page-0"])[0].removeprefix("provider-page-"))
        prefix = query["prefix"][0]
        maximum = int(query["maxResults"][0])
        live: list[tuple[str, int, int]] = []
        for name, versions in self.objects.items():
            if not name.startswith(prefix):
                continue
            current = next(
                ((generation, size) for generation, size, is_live in versions if is_live),
                None,
            )
            if current is not None:
                live.append((name, current[1], current[0]))
        live.sort(key=lambda item: item[0])
        start_offset = query.get("startOffset", [None])[0]
        if start_offset is not None:
            live = [item for item in live if item[0] >= start_offset]
        page_size = maximum if self.server_page_size is None else min(maximum, self.server_page_size)
        selected = live[page_offset : page_offset + page_size]
        document: dict[str, tp.Any] = {}
        if selected:
            document["items"] = [
                {"name": name, "size": str(size), "generation": str(generation)} for name, size, generation in selected
            ]
        if len(live) > page_offset + len(selected):
            document["nextPageToken"] = f"provider-page-{page_offset + len(selected)}"
        return 200, json.dumps(document, separators=(",", ":")).encode()

    def _get(self, name: str, query: dict[str, list[str]]) -> tuple[int, bytes]:
        assert query == {
            "projection": ["noAcl"],
            "softDeleted": ["false"],
            "fields": ["name,size,generation"],
        }
        current = next(
            ((generation, size) for generation, size, is_live in self.objects.get(name, []) if is_live),
            None,
        )
        if current is None:
            return 404, b'{"error":{"message":"not found"}}'
        generation, size = current
        return 200, json.dumps(
            {"name": name, "size": str(size), "generation": str(generation)},
            separators=(",", ":"),
        ).encode()


def _inventory_store(
    api: FakeGCSInventoryAPI,
    root_uri: str = "gs://eyvan-managed/artifacts",
) -> GCSManagedArtifactStore:
    return GCSManagedArtifactStore(
        root_uri,
        inventory_transport=api,
        access_token_provider=lambda: "fake-access-token",
    )


def test_local_inventory_is_bounded_scoped_and_detects_replacement_identity(tmp_path: Path) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    uris = tuple(result_uri(f"run-{index}") for index in range(3))
    for index, uri in enumerate(uris):
        store.put(uri, b"x" * (index + 1), org_id="acme", project_id="research")
    store.put(uris[0], b"foreign", org_id="acme", project_id="other")

    first = store.list_scope(org_id="acme", project_id="research", limit=2)
    second = store.list_scope(
        org_id="acme",
        project_id="research",
        cursor=first.next_cursor,
        limit=2,
    )

    assert [item.uri for item in first.objects] == list(uris[:2])
    assert [item.size_bytes for item in first.objects] == [1, 2]
    assert [item.sha256 for item in first.objects] == [
        hashlib.sha256(b"x").hexdigest(),
        hashlib.sha256(b"xx").hexdigest(),
    ]
    assert first.next_cursor == uris[1]
    assert [item.uri for item in second.objects] == [uris[2]]
    assert second.next_cursor is None
    assert all(item.object_version.startswith("local:") for item in (*first.objects, *second.objects))

    before = store.stat(uris[0], org_id="acme", project_id="research")
    assert before is not None
    target = store.root / "scopes" / storage_scope("acme", "research") / "results" / "run-0.result"
    replacement = target.with_name("replacement.result")
    replacement.write_bytes(b"replacement")
    replacement.replace(target)
    after = store.stat(uris[0], org_id="acme", project_id="research")

    assert after is not None
    assert after.size_bytes == len(b"replacement")
    assert after.object_version != before.object_version
    assert store.stat(uris[0], org_id="acme", project_id="missing") is None


def test_local_inventory_fails_closed_on_an_unparseable_physical_entry(tmp_path: Path) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    uri = result_uri("valid")
    store.put(uri, b"result", org_id="acme", project_id="research")
    kind_dir = store.root / "scopes" / storage_scope("acme", "research") / "results"
    (kind_dir / "not-a-managed-object.txt").write_bytes(b"unknown")

    with pytest.raises(ManagedArtifactInventoryError, match="unexpected file"):
        store.list_scope(org_id="acme", project_id="research")


def test_gcs_inventory_json_api_returns_only_the_current_live_generation() -> None:
    api = FakeGCSInventoryAPI()
    store = _inventory_store(api)
    first_uri, second_uri, deleted_uri = result_uri("run-1"), result_uri("run-2"), result_uri("run-deleted")
    prefix = f"artifacts/scopes/{storage_scope('acme', 'research')}/"
    first_name = f"{prefix}results/run-1.result"
    second_name = f"{prefix}results/run-2.result"
    deleted_name = f"{prefix}results/run-deleted.result"
    api.add(first_name, size=99, generation=9, live=False)
    api.add(first_name, size=3, generation=41, live=True)
    api.add(second_name, size=88, generation=10, live=False)
    api.add(second_name, size=6, generation=42, live=True)
    api.add(deleted_name, size=7, generation=11, live=False)
    other_prefix = f"artifacts/scopes/{storage_scope('acme', 'other')}/"
    api.add(f"{other_prefix}results/run-1.result", size=7, generation=99, live=True)

    before = len(api.calls)
    first = store.list_scope(org_id="acme", project_id="research", limit=1)
    assert len(api.calls) == before + 1
    second = store.list_scope(
        org_id="acme",
        project_id="research",
        cursor=first.next_cursor,
        limit=1,
    )
    described = store.stat(second_uri, org_id="acme", project_id="research")
    deleted = store.stat(deleted_uri, org_id="acme", project_id="research")

    assert first.objects == (ManagedArtifactObjectMetadata(uri=first_uri, size_bytes=3, object_version="gcs:41"),)
    assert first.next_cursor == first_uri
    assert second.objects == (ManagedArtifactObjectMetadata(uri=second_uri, size_bytes=6, object_version="gcs:42"),)
    assert second.next_cursor is None
    assert described == second.objects[0]
    assert deleted is None
    first_query = api.calls[0]["query"]
    assert first_query["prefix"] == [prefix]
    assert first_query["maxResults"] == ["1000"]
    assert first_query["versions"] == ["false"]
    assert first_query["softDeleted"] == ["false"]
    assert "startOffset" not in first_query
    assert "startOffset" not in api.calls[1]["query"]
    stat_calls = api.calls[2:]
    assert all(call["path"] != "/storage/v1/b/eyvan-managed/o" for call in stat_calls)
    assert all("versions" not in call["query"] and "generation" not in call["query"] for call in stat_calls)


def test_gcs_inventory_honors_bounded_page_tokens_and_resumes_with_logical_cursor() -> None:
    api = FakeGCSInventoryAPI()
    store = _inventory_store(api)
    prefix = f"artifacts/scopes/{storage_scope('acme', 'research')}/"
    uris = tuple(result_uri(f"run-{index}") for index in range(3))
    names = tuple(f"{prefix}results/run-{index}.result" for index in range(3))
    for index, name in enumerate(names):
        api.add(name, size=index + 1, generation=100 + index, live=True)

    # Short provider pages are consumed within the bounded logical scan.
    api.server_page_size = 1
    first = store.list_scope(org_id="acme", project_id="research", limit=2)
    assert [item.uri for item in first.objects] == list(uris[:2])
    assert first.next_cursor == uris[1]
    assert len(api.calls) == 3
    assert api.calls[1]["query"]["pageToken"] == ["provider-page-1"]
    assert api.calls[2]["query"]["pageToken"] == ["provider-page-2"]

    second = store.list_scope(org_id="acme", project_id="research", cursor=first.next_cursor, limit=2)
    assert [item.uri for item in second.objects] == [uris[2]]
    assert second.next_cursor is None
    assert len(api.calls) == 6
    assert all("startOffset" not in call["query"] for call in api.calls)


def test_gcs_inventory_cursor_does_not_skip_a_longer_canonical_filename() -> None:
    api = FakeGCSInventoryAPI()
    store = _inventory_store(api)
    prefix = f"artifacts/scopes/{storage_scope('acme', 'research')}/results/"
    uris = (result_uri("a"), result_uri("a.result-x"), result_uri("b"))
    for generation, uri in enumerate(uris, start=1):
        reference = parse_managed_uri(uri)
        api.add(
            f"{prefix}{reference.object_id}.result",
            size=generation,
            generation=generation,
            live=True,
        )
    api.server_page_size = 1

    observed: list[str] = []
    cursor: str | None = None
    while True:
        page = store.list_scope(org_id="acme", project_id="research", cursor=cursor, limit=2)
        observed.extend(item.uri for item in page.objects)
        cursor = page.next_cursor
        if cursor is None:
            break

    assert observed == list(uris)
    assert all("startOffset" not in call["query"] for call in api.calls)


def test_gcs_inventory_consumes_bounded_requests_for_a_large_prefix() -> None:
    api = FakeGCSInventoryAPI()
    store = _inventory_store(api)
    prefix = f"artifacts/scopes/{storage_scope('acme', 'research')}/results/"
    for index in range(5000):
        api.add(f"{prefix}run-{index:05d}.result", size=index, generation=index + 1, live=True)

    page = store.list_scope(org_id="acme", project_id="research", limit=4)

    assert len(page.objects) == 4
    assert page.next_cursor == result_uri("run-00003")
    assert len(api.calls) == 5
    assert all(call["query"]["maxResults"] == ["1000"] for call in api.calls)
    assert all(call["maximum_response_bytes"] == 2 * 1024 * 1024 for call in api.calls)


def test_gcs_inventory_percent_encodes_scope_prefix_and_exact_object_name() -> None:
    api = FakeGCSInventoryAPI()
    store = _inventory_store(api, "gs://eyvan-managed/root + blue=")
    uri = result_uri("run-1")
    prefix = f"root + blue=/scopes/{storage_scope('acme', 'research')}/"
    name = f"{prefix}results/run-1.result"
    api.add(name, size=3, generation=41, live=True)

    page = store.list_scope(org_id="acme", project_id="research", limit=1)
    metadata = store.stat(uri, org_id="acme", project_id="research")

    assert page.objects == (metadata,)
    assert api.calls[0]["query"]["prefix"] == [prefix]
    assert "root%20%2B%20blue%3D%2Fscopes%2F" in api.calls[0]["url"]
    assert "root%20%2B%20blue%3D%2Fscopes%2F" in api.calls[1]["path"]
    assert api.calls[1]["query"] == {
        "projection": ["noAcl"],
        "softDeleted": ["false"],
        "fields": ["name,size,generation"],
    }


def test_gcs_inventory_rejects_duplicate_items_and_malformed_or_unbounded_json() -> None:
    prefix = f"artifacts/scopes/{storage_scope('acme', 'research')}/"
    item = {"name": f"{prefix}results/run-1.result", "size": "3", "generation": "41"}
    cases = (
        (b"not-json", "malformed JSON"),
        (b"[]", "invalid JSON document"),
        (json.dumps({"items": [item], "unexpected": True}).encode(), "unexpected JSON fields"),
        (json.dumps({"items": [item, item]}).encode(), "duplicate object names"),
        (
            json.dumps({"items": [{**item, "generation": "9" * 21}]}).encode(),
            "invalid object generation",
        ),
        (json.dumps({"items": [item], "nextPageToken": "t" * (16 * 1024 + 1)}).encode(), "invalid page token"),
        (b"x" * (2 * 1024 * 1024 + 1), "bounded body size"),
    )
    for payload, message in cases:
        api = FakeGCSInventoryAPI()
        api.queue((200, payload))
        store = _inventory_store(api)
        with pytest.raises(ManagedArtifactInventoryError, match=message):
            store.list_scope(org_id="acme", project_id="research", limit=1)
        assert len(api.calls) == 1


def test_gcs_inventory_sanitizes_provider_failures_and_never_exposes_bearer_tokens() -> None:
    provider_secret = "provider-body-secret"
    api = FakeGCSInventoryAPI()
    api.queue((403, f'{{"error":"{provider_secret}","url":"https://private.invalid"}}'.encode()))
    store = _inventory_store(api)

    with pytest.raises(ManagedArtifactInventoryError) as failure:
        store.list_scope(org_id="acme", project_id="research")
    message = str(failure.value)
    assert message == "GCS inventory request failed with HTTP status 403"
    assert provider_secret not in message
    assert "private.invalid" not in message
    assert "fake-access-token" not in message

    timed_out = FakeGCSInventoryAPI()
    timed_out.queue(TimeoutError("fake-access-token https://private.invalid"))
    with pytest.raises(ManagedArtifactInventoryError) as timeout_failure:
        _inventory_store(timed_out).list_scope(org_id="acme", project_id="research")
    assert str(timeout_failure.value) == "GCS inventory request timed out"
    assert timeout_failure.value.__cause__ is None


def test_gcs_inventory_bounds_access_tokens_cursors_names_and_stalled_pages() -> None:
    api = FakeGCSInventoryAPI()
    oversized_token_store = GCSManagedArtifactStore(
        "gs://eyvan-managed/artifacts",
        inventory_transport=api,
        access_token_provider=lambda: "s" * (16 * 1024 + 1),
    )
    with pytest.raises(ManagedArtifactInventoryError, match="invalid access token"):
        oversized_token_store.list_scope(org_id="acme", project_id="research")
    assert api.calls == []

    store = _inventory_store(api)
    with pytest.raises(ValueError, match="cursor is too large"):
        store.list_scope(
            org_id="acme",
            project_id="research",
            cursor=f"eyvan://results/{'x' * 600}",
        )
    assert api.calls == []

    api.queue((200, json.dumps({"nextPageToken": "valid-provider-token"}).encode()))
    with pytest.raises(ManagedArtifactInventoryError, match="did not advance"):
        store.list_scope(org_id="acme", project_id="research")

    oversized_name = "x" * 1025
    api.queue(
        (
            200,
            json.dumps({"items": [{"name": oversized_name, "size": "1", "generation": "1"}]}).encode(),
        )
    )
    with pytest.raises(ManagedArtifactInventoryError, match="oversized object name"):
        store.list_scope(org_id="acme", project_id="research")

    oversized_root = _inventory_store(FakeGCSInventoryAPI(), f"gs://eyvan-managed/{'x' * 1100}")
    with pytest.raises(ManagedArtifactInventoryError, match="oversized scope prefix"):
        oversized_root.list_scope(org_id="acme", project_id="research")


def test_gcs_inventory_root_rejects_wildcards_that_could_escape_the_requested_prefix() -> None:
    with pytest.raises(ValueError, match="non-wildcard"):
        GCSManagedArtifactStore("gs://eyvan-managed/artifacts-*/tenant")
    with pytest.raises(ValueError, match="control characters"):
        GCSManagedArtifactStore("gs://eyvan-managed/artifacts/tenant\tother")


def test_an_artifact_token_keyring_rotates_without_orphaning_outstanding_tokens() -> None:
    """Tokens minted before a rotate keep verifying inside the overlap window."""

    old_secret = secrets.token_bytes(32)
    new_secret = secrets.token_bytes(32)
    keyring = RotatingCredential(old_secret)
    signer = ArtifactTokenSigner(keyring)

    uri = result_uri("obj-1")

    def issue(workload_id: str) -> str:
        return signer.issue("read", uri, org_id="acme", project_id="research", workload_id=workload_id)

    stale = issue("w-old")

    keyring.begin("rotate-artifacts")
    keyring.commit("rotate-artifacts", new_secret, overlap_s=600.0)

    fresh = issue("w-new")
    assert fresh != stale
    # The pre-rotation token still verifies against the retired material...
    claims = signer.verify(stale, "read", uri, org_id="acme", project_id="research", workload_id="w-old")
    assert claims.workload_id == "w-old"
    # ...and the new one mints under, and verifies against, the new material.
    verified = signer.verify(fresh, "read", uri, org_id="acme", project_id="research", workload_id="w-new")
    assert verified.workload_id == "w-new"
    # Scope mismatches still refuse under either material.
    with pytest.raises(InvalidArtifactToken):
        signer.verify(stale, "read", uri, org_id="acme", project_id="research", workload_id="w-other")
