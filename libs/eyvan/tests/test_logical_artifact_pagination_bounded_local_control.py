# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Logical artifact pagination and bounded local control-layout regressions."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_managed_artifacts import FakeGCSInventoryAPI

from eyvan.artifacts.managed import (
    GCSManagedArtifactStore,
    InvalidManagedArtifactURI,
    LocalManagedArtifactStore,
    ManagedArtifactDeleteStatus,
    ManagedArtifactInventoryError,
    parse_managed_uri,
    result_uri,
    storage_scope,
)
from eyvan.records import Principal, Role, RoleGrant
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service import ControlPlaneService


def inventory_store(api: FakeGCSInventoryAPI) -> GCSManagedArtifactStore:
    def no_provider_calls(*args, **kwargs):
        raise AssertionError("offline inventory must not invoke provider commands")

    return GCSManagedArtifactStore(
        "gs://eyvan-managed/artifacts",
        runner=no_provider_calls,
        inventory_transport=api,
        access_token_provider=lambda: "fake-access-token",
    )


@pytest.mark.parametrize("limit", (1, 2))
@pytest.mark.parametrize("suffix", ("-1", ".0", ".result-x", "_1"))
@pytest.mark.asyncio
async def test_inventory_union_uses_logical_order(limit: int, suffix: str) -> None:
    api = FakeGCSInventoryAPI()
    api.server_page_size = 1
    store = inventory_store(api)
    prefix = f"artifacts/scopes/{storage_scope('acme', 'research')}/results/"
    identifiers = ("a", "a" + suffix, "b")
    expected = sorted(result_uri(identifier) for identifier in identifiers)
    for generation, identifier in enumerate(identifiers, 1):
        api.add(prefix + identifier + ".result", size=1, generation=generation, live=True)
    service = ControlPlaneService(InMemoryRepository(), artifact_store=store)
    await service.index_managed_artifact(org_id="acme", project_id="research", uri=result_uri("a"), size_bytes=1)
    principal = Principal(
        subject="alice", grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="research"),)
    )
    cursor = None
    observed = []
    for _ in range(6):
        page = await service.inventory_managed_artifacts(
            principal, org_id="acme", project_id="research", limit=limit, cursor=cursor
        )
        observed.extend(entry.uri for entry in page.entries)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert cursor is None
    assert observed == expected
    assert len(set(observed)) == len(expected)


def test_gcs_inventory_scan_budget_exhaustion_is_not_a_partial_page() -> None:
    api = FakeGCSInventoryAPI()
    api.server_page_size = 1
    prefix = f"artifacts/scopes/{storage_scope('acme', 'research')}/results/"
    for generation in range(40):
        api.add(prefix + f"run-{generation:03d}.result", size=1, generation=generation + 1, live=True)
    with pytest.raises(ManagedArtifactInventoryError, match="scan budget"):
        inventory_store(api).list_scope(org_id="acme", project_id="research", limit=1)
    assert len(api.calls) == 32


def test_gcs_inventory_rejects_cross_page_nonprogress() -> None:
    api = FakeGCSInventoryAPI()
    prefix = f"artifacts/scopes/{storage_scope('acme', 'research')}/results/"
    item = {"name": prefix + "a.result", "size": "1", "generation": "1"}
    api.queue((200, json.dumps({"items": [item], "nextPageToken": "first"}).encode()))
    api.queue((200, json.dumps({"items": [item], "nextPageToken": "second"}).encode()))
    with pytest.raises(ManagedArtifactInventoryError, match="did not advance"):
        inventory_store(api).list_scope(org_id="acme", project_id="research", limit=1)
    assert len(api.calls) == 2


@pytest.mark.parametrize("count", (100, 200))
def test_unique_local_object_churn_bounds_control_inodes(tmp_path: Path, count: int) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    for index in range(count):
        uri = result_uri(f"unique-{index}")
        written = store.put_versioned(uri, b"payload", org_id="acme", project_id="research")
        outcome = store.delete_versioned(
            uri, org_id="acme", project_id="research", object_version=written.metadata.object_version
        )
        assert outcome.status is ManagedArtifactDeleteStatus.DELETED
    assert store.list_scope(org_id="acme", project_id="research").objects == ()
    assert len(list((store.root / ".locks").glob("*.lock"))) <= 16
    assert list((store.root / ".generations").iterdir()) == []
    restarted = LocalManagedArtifactStore(store.root)
    assert restarted.put_versioned(result_uri("next"), b"next", org_id="acme", project_id="research").created


@pytest.mark.parametrize("legacy_contents", ("empty", "lock", "payload"))
def test_legacy_local_layout_requires_explicit_maintenance(tmp_path: Path, legacy_contents: str) -> None:
    root = tmp_path / "legacy"
    root.mkdir(mode=0o700)
    if legacy_contents in {"empty", "lock"}:
        (root / ".locks").mkdir(mode=0o700)
    if legacy_contents == "lock":
        (root / ".locks" / ("a" * 64 + ".lock")).write_bytes(b"legacy-lock-preserved")
    if legacy_contents == "payload":
        (root / "unknown-operator-data").write_bytes(b"untouched")
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(InvalidManagedArtifactURI, match=r"quiesced.*maintenance"):
        LocalManagedArtifactStore(root)
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    assert not (root / ".locks" / "layout-v2").exists()


def test_distinct_objects_in_same_stripe_serialize_across_processes(tmp_path: Path) -> None:
    store = LocalManagedArtifactStore(tmp_path / "managed")
    first = result_uri("first")

    def stripe(uri: str) -> int:
        relative = f"scopes/{storage_scope('acme', 'research')}/results/{parse_managed_uri(uri).object_id}.result"
        return int(hashlib.sha256(relative.encode("ascii")).hexdigest(), 16) % 16

    second = next(
        result_uri(f"second-{index}") for index in range(1000) if stripe(result_uri(f"second-{index}")) == stripe(first)
    )
    script = """
import sys
from eyvan.artifacts.managed import LocalManagedArtifactStore
store = LocalManagedArtifactStore(sys.argv[1])
print("ready", flush=True)
store.put_versioned(sys.argv[2], b"child", org_id="acme", project_id="research")
print("written", flush=True)
"""
    process = None
    try:
        with store._object_lock(parse_managed_uri(first), "acme", "research"):
            process = subprocess.Popen(
                [sys.executable, "-B", "-c", script, str(store.root), second],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert process.stdout is not None
            assert process.stdout.readline().strip() == "ready"
            with pytest.raises(subprocess.TimeoutExpired):
                process.communicate(timeout=0.2)
            assert store.stat(second, org_id="acme", project_id="research") is None
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr
        assert stdout.strip() == "written"
        assert store.get(second, org_id="acme", project_id="research") == b"child"
        lock = store.root / ".locks" / f"stripe-{stripe(first):02x}.lock"
        original_inode = lock.stat().st_ino
        metadata = store.stat(second, org_id="acme", project_id="research")
        assert metadata is not None
        assert (
            store.delete_versioned(
                second, org_id="acme", project_id="research", object_version=metadata.object_version
            ).status
            is ManagedArtifactDeleteStatus.DELETED
        )
        assert lock.stat().st_ino == original_inode
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)
