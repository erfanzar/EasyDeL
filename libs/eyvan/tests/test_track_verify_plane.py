# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Executable regressions for the independent plane-wiring review findings."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import click
import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import CommandExecution, Principal, Role, RoleGrant, WorkloadDefinition, WorkloadRecord
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands import track as commands
from eyvan.sdk.commands.fleets import FleetCommandService
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.contexts import EyvanContext, EyvanContextConfig, FileContextStore
from eyvan.server.api import tracking_proxy
from eyvan.server.api.app import create_app
from eyvan.server.controller import EyvanController
from eyvan.server.local import tracker
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.track._format import SegmentName, encode_segment
from eyvan.track._store import store_for_uri
from eyvan.track.tracker.index import TrackingIndex


def _token_session(tmp_path, token):
    constructed = []
    original = SimpleNamespace()
    context = EyvanContext(fleet="saved", org_id="acme", project_id="research", base_url="http://127.0.0.1:18080")
    session = CliSession(
        client=original,
        identity=ClientIdentity(token=token),
        context_store=FileContextStore(tmp_path / "contexts.json"),
        context_config=EyvanContextConfig(contexts={"saved": context}),
        base_url="http://127.0.0.1:18081",
        org_id="acme",
        project_id="research",
        overrides=ContextOverrides(),
        client_factory=lambda url, headers: constructed.append((url, headers)) or original,
    )
    return session, original, constructed


@pytest.mark.parametrize("token", ["source-plane-only", ""])
def test_id0_token_identity_cannot_implicitly_follow_a_saved_local_plane(tmp_path, token) -> None:
    session, original, constructed = _token_session(tmp_path, token)
    identity = session.identity
    config = session.context_config
    with pytest.raises(click.ClickException, match="different saved control-plane endpoint"):
        FleetCommandService(session).require_apply_endpoint("saved")
    assert constructed == [], "no destination client may receive the source plane's token"
    assert session.client is original
    assert session.identity is identity
    assert session.context_config is config
    assert session.base_url == "http://127.0.0.1:18081"


@pytest.mark.parametrize("scope_field", ["org_id", "project_id"])
@pytest.mark.parametrize(
    "unsafe", ["%72esearch", "../research", "nested/project", "nested\\project", "scope%252fresearch"]
)
@pytest.mark.parametrize("root", ["gs://runs/tracking", "file:///tmp/tracking"])
def test_id9_invalid_tracking_scopes_do_not_alias_or_block_dispatch(scope_field, unsafe, root, monkeypatch) -> None:
    monkeypatch.delenv("EYVAN_CONTROL_PLANE_CA_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    scope = {"org_id": "acme", "project_id": "research", scope_field: unsafe}
    workload = WorkloadRecord(
        id="wl-1",
        **scope,
        created_by="alice",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py")),
    )
    controller = EyvanController(repository, service, settings=EyvanSettings())
    baseline = controller._render_execution(workload, runtime_environment={"RUNTIME": "unchanged"})
    controller.settings = EyvanSettings(tracking_store_uri=root)
    assert controller._render_execution(workload, runtime_environment={"RUNTIME": "unchanged"}) == baseline
    assert "EYVAN_TRACKING_URI" not in baseline[1]


def _indexing_supervisor(directory):
    live = set()
    arguments = {}
    indexes = []

    def launch(argv, environment, log):
        store = store_for_uri(argv[argv.index("--store") + 1])
        index = TrackingIndex(Path(argv[argv.index("--db") + 1]))
        index.scan(store)
        indexes.append(index)
        pid = 98765
        arguments[pid] = argv
        live.add(pid)
        return pid

    instance = tracker.LocalTrackerSupervisor(
        directory=directory,
        launcher=launch,
        health_probe=lambda url: bool(live),
        process_liveness=lambda pid: pid in live,
        argument_reader=arguments.get,
        process_terminator=lambda pid: live.discard(pid),
        process_killer=lambda pid: live.discard(pid),
    )
    return instance, indexes


@pytest.mark.parametrize("replacement_has_same_key", [False, True])
def test_id32_local_tracker_indexes_are_bound_to_the_canonical_store(tmp_path, replacement_has_same_key) -> None:
    store_a = store_for_uri(str(tmp_path / "store-a"))
    store_b = store_for_uri(str(tmp_path / "store-b"))
    key = f"acme/research/run-1/meta/{SegmentName(1, 0, 'bbbbbbbbbbbb', 0).filename}"
    store_a.put(key, encode_segment([{"op": "start", "t": 1000, "name": "from-a"}]))
    if replacement_has_same_key:
        store_b.put(key, encode_segment([{"op": "start", "t": 1000, "name": "from-b"}]))
    instance, indexes = _indexing_supervisor(tmp_path / "tracker")
    first = instance.up(store_uri=str(tmp_path / "store-a"))
    assert indexes[-1].list_runs("acme", "research")[0]["name"] == "from-a"
    instance.down()
    second = instance.up(store_uri=store_b.root)
    runs = indexes[-1].list_runs("acme", "research")
    assert [run["name"] for run in runs] == (["from-b"] if replacement_has_same_key else [])
    assert second.db != first.db
    assert tracker.read_tracker_record(instance.directory) == second
    instance.down()
    again = instance.up(store_uri=store_a.root)
    assert again.db == first.db, "equivalent path and URI spellings must share an index"
    assert indexes[-1].list_runs("acme", "research")[0]["name"] == "from-a"
    instance.down()


def _blob_session(data, before_return=lambda: None):
    def request_bytes(*args, **kwargs):
        before_return()
        return data, "image/png"

    return SimpleNamespace(
        client=SimpleNamespace(request_bytes=request_bytes),
        workload_scope=lambda: SimpleNamespace(org_id="acme", project_id="research"),
    )


def _download(session, output, *, force=False):
    argv = ["blob", "run-1", hashlib.sha256(b"blob-bytes").hexdigest(), "-o", str(output)]
    if force:
        argv.append("--force")
    return CliRunner().invoke(commands.runs, argv, obj=session)


def test_id33_no_force_never_overwrites_a_destination_created_during_download(tmp_path) -> None:
    output = tmp_path / "download.png"
    session = _blob_session(b"blob-bytes", lambda: output.write_bytes(b"important"))
    result = _download(session, output)
    assert result.exit_code == 1, result.output
    assert "already exists" in result.output
    assert output.read_bytes() == b"important"


def test_id33_no_force_rejects_a_dangling_output_symlink(tmp_path) -> None:
    output = tmp_path / "download.png"
    target = tmp_path / "absent-target"
    output.symlink_to(target)
    result = _download(_blob_session(b"blob-bytes"), output)
    assert result.exit_code == 1, result.output
    assert "already exists" in result.output
    assert output.is_symlink()
    assert not target.exists()


def test_id33_force_replaces_the_output_entry_not_a_symlink_target(tmp_path) -> None:
    output = tmp_path / "download.png"
    target = tmp_path / "important"
    target.write_bytes(b"important")
    output.symlink_to(target)
    result = _download(_blob_session(b"blob-bytes"), output, force=True)
    assert result.exit_code == 0, result.output
    assert not output.is_symlink()
    assert output.read_bytes() == b"blob-bytes"
    assert target.read_bytes() == b"important"


def test_id33_failed_atomic_force_keeps_original_and_cleans_temporary_file(tmp_path, monkeypatch) -> None:
    output = tmp_path / "download.png"
    output.write_bytes(b"important")

    def fail_replace(source, destination):
        assert Path(source).read_bytes() == b"blob-bytes"
        assert Path(destination) == output
        assert output.read_bytes() == b"important"
        raise OSError("replace denied")

    monkeypatch.setattr(os, "replace", fail_replace)
    result = _download(_blob_session(b"blob-bytes"), output, force=True)
    assert result.exit_code == 1, result.output
    assert output.read_bytes() == b"important"
    assert list(tmp_path.iterdir()) == [output]


class _Verifier:
    async def verify(self, token):
        return Principal(
            subject="alice", grants=(RoleGrant(role=Role.PROJECT_VIEWER, org_id="acme", project_id="research"),)
        )


@pytest.fixture
def proxy_app(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return create_app(
        settings=EyvanSettings(
            controller_enabled=False,
            artifact_store_uri=str(tmp_path / "artifacts"),
            tracker_url="https://tracker.example",
            tracker_token="upstream-only",
        ),
        verifier=_Verifier(),
    )


def _get_blob(app):
    with TestClient(app) as client:
        return client.get(
            f"/api/v1/tracking/runs/run-1/blobs/{'a' * 64}?org_id=acme&project_id=research",
            headers={"Authorization": "Bearer reader"},
        )


def test_id34_authenticated_proxy_blobs_cannot_be_cached_across_sessions(proxy_app) -> None:
    proxy_app.state.tracking_transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"blob-bytes", headers={"Content-Type": "image/png"})
    )
    response = _get_blob(proxy_app)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-security-policy"] == "default-src 'none'; sandbox"
    assert response.headers["content-disposition"] == "inline"


class _CountingStream(httpx.AsyncByteStream):
    def __init__(self, chunks=80):
        self.chunks = chunks
        self.consumed = 0
        self.closed = False

    async def __aiter__(self):
        for _ in range(self.chunks):
            self.consumed += 1
            yield b"x" * (1024 * 1024)

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("status_code", [200, 500])
def test_id35_proxy_aborts_and_closes_an_oversized_stream_before_consuming_it_all(proxy_app, status_code) -> None:
    stream = _CountingStream()
    proxy_app.state.tracking_transport = httpx.MockTransport(
        lambda request: httpx.Response(status_code, stream=stream, headers={"Content-Type": "image/png"})
    )
    response = _get_blob(proxy_app)
    assert response.status_code == 502
    assert "oversized" in response.json()["detail"]
    assert stream.consumed == tracking_proxy.MAX_BLOB_BYTES // (1024 * 1024) + 1
    assert stream.consumed < stream.chunks
    assert stream.closed


def test_id35_proxy_rejects_oversized_content_length_before_reading(proxy_app) -> None:
    stream = _CountingStream(chunks=1)
    proxy_app.state.tracking_transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            stream=stream,
            headers={"Content-Type": "image/png", "Content-Length": str(tracking_proxy.MAX_BLOB_BYTES + 1)},
        )
    )
    response = _get_blob(proxy_app)
    assert response.status_code == 502
    assert "oversized" in response.json()["detail"]
    assert stream.consumed == 0
    assert stream.closed


def test_id43_tracker_rejects_symlink_before_changing_target_permissions(tmp_path) -> None:
    target = tmp_path / "shared"
    target.mkdir(mode=0o755)
    directory = tmp_path / "tracker"
    directory.symlink_to(target)
    instance = tracker.LocalTrackerSupervisor(
        directory=directory,
        launcher=lambda *args: pytest.fail("must reject before launch"),
        health_probe=lambda *args: pytest.fail("must reject before network probing"),
    )
    with pytest.raises(tracker.LocalTrackerError, match="directory"):
        instance.up(store_uri=str(tmp_path / "store"))
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert list(target.iterdir()) == []


def test_id43_tracker_rejects_foreign_owned_directory_before_chmod(tmp_path, monkeypatch) -> None:
    directory = tmp_path / "tracker"
    directory.mkdir(mode=0o755)
    monkeypatch.setattr(tracker.os, "geteuid", lambda: directory.stat().st_uid + 1)
    instance = tracker.LocalTrackerSupervisor(directory=directory)
    with pytest.raises(tracker.LocalTrackerError, match="directory"):
        instance.up(store_uri=str(tmp_path / "store"))
    assert stat.S_IMODE(directory.stat().st_mode) == 0o755
    assert list(directory.iterdir()) == []
