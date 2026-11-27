# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end tests for collaborative runs, speedruns, multi-run compare, notes, feed, region check, and CLI."""

from __future__ import annotations

import asyncio
import getpass
import hashlib
import json
import sqlite3
import time
import typing as tp
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

import eyvan.track as track
from eyvan.config import EyvanSettings
from eyvan.records import CommandExecution, Principal, Role, RoleGrant, WorkloadDefinition, WorkloadRecord
from eyvan.sdk.commands import storage as storage_cli
from eyvan.sdk.commands import track as track_cli
from eyvan.server.api import collab_routes
from eyvan.server.api.app import create_app
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.collaboration import CollaborationStore, can_manage, can_view
from eyvan.server.service.errors import InputError
from eyvan.storage import BucketLocation, GcpContext
from eyvan.track._format import SegmentName, blob_key, encode_segment
from eyvan.track._store import LocalTrackingStore
from eyvan.track.tracker import TrackingIndex, create_tracker_app

SCOPE = {"org_id": "acme", "project_id": "research"}


class _MultiUserVerifier:
    """Resolve static test tokens into distinct researcher, viewer, admin, and outsider principals."""

    _PRINCIPALS: tp.ClassVar[dict[str, Principal]] = {
        "alice": Principal(
            subject="alice",
            grants=(RoleGrant(role=Role.PROJECT_SUBMITTER, org_id="acme", project_id="research"),),
        ),
        "bob": Principal(
            subject="bob",
            grants=(RoleGrant(role=Role.PROJECT_SUBMITTER, org_id="acme", project_id="research"),),
        ),
        "carol": Principal(
            subject="carol",
            grants=(RoleGrant(role=Role.PROJECT_VIEWER, org_id="acme", project_id="research"),),
        ),
        "admin": Principal(
            subject="admin",
            grants=(RoleGrant(role=Role.FLEET_ADMIN),),
        ),
        "outsider": Principal(
            subject="outsider",
            grants=(RoleGrant(role=Role.PROJECT_VIEWER, org_id="acme", project_id="other"),),
        ),
        "grantless": Principal(
            subject="grantless",
            grants=(),
        ),
    }

    async def verify(self, token: str) -> Principal:
        if token not in self._PRINCIPALS:
            raise ValueError("unknown token")
        return self._PRINCIPALS[token]


def _auth(user: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {user}"}


def _put_segment(
    store: LocalTrackingStore,
    stream: str,
    records: list[dict[str, object]],
    *,
    run: str,
    org: str = "acme",
    project: str = "research",
    attempt: int = 1,
    rank: int = 0,
    seq: int = 0,
    writer: str = "aaaaaaaaaaaa",
) -> None:
    seg = SegmentName(attempt, rank, writer, seq)
    store.put(f"{org}/{project}/{run}/{stream}/{seg.filename}", encode_segment(records))


def _build_collab_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tracker_store: LocalTrackingStore | None = None,
    tracker_enabled: bool = True,
) -> tuple[TestClient, object, TrackingIndex | None]:
    monkeypatch.setenv("HOME", str(tmp_path))
    index: TrackingIndex | None = None
    settings_kwargs: dict[str, object] = {
        "controller_enabled": False,
        "artifact_store_uri": str(tmp_path / "artifacts"),
        "artifact_token_key": "k" * 32,
    }
    if tracker_enabled:
        settings_kwargs["tracker_url"] = "https://tracker.test"
        settings_kwargs["tracker_token"] = "tracker-secret"

    collab_store = CollaborationStore(tmp_path / "collaboration.sqlite3")
    app = create_app(
        settings=EyvanSettings(**settings_kwargs),  # type: ignore[arg-type]
        verifier=_MultiUserVerifier(),
        collaboration_store=collab_store,
    )

    if tracker_enabled and tracker_store is not None:
        index = TrackingIndex(tmp_path / "tracker-index.sqlite3")
        index.scan(tracker_store)
        tracker_app = create_tracker_app(tracker_store, index, token="tracker-secret", scan_interval_s=3600.0)
        app.state.tracking_transport = httpx.ASGITransport(app=tracker_app)

    return TestClient(app), app, index


# --------------------------------------------------------------------------- 1. Tracker SDK & Index


def test_track_start_attribution_and_index_enrichment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = LocalTrackingStore(tmp_path / "store" / "acme" / "research")
    now = time.time() - 10.0

    # Explicit owner, visibility, experiment
    run_a = track.start(
        "gpt-fast",
        run_id="run-explicit",
        owner="alice",
        visibility="private",
        experiment="nanogpt-speedrun",
        group="sweep-1",
        tags=("fp8", "muon"),
        config={"lr": 3e-4, "model": {"layers": 12}},
        store=store,
        environment={},
        flush_interval_s=3600.0,
        spill_dir=tmp_path / "spill",
        clock=lambda: now,
    )
    assert (run_a.owner, run_a.visibility, run_a.experiment) == ("alice", "private", "nanogpt-speedrun")
    run_a.log({"val/loss": 4.2, "train/loss": 4.5, "nan_metric": float("nan")}, step=0)
    run_a.log({"val/loss": 3.15, "train/loss": 3.4}, step=10)
    run_a.summary(best_val=3.15)
    assert run_a.finish() is True

    # Auto-default owner from EYVAN_WORKLOAD_OWNER
    run_b = track.start(
        "from-env",
        run_id="run-env-owner",
        store=store,
        environment={"EYVAN_WORKLOAD_OWNER": "bob"},
        flush_interval_s=3600.0,
        spill_dir=tmp_path / "spill",
        clock=lambda: now + 5.0,
    )
    assert run_b.owner == "bob"
    assert run_b.visibility is None
    run_b.log({"val/loss": 3.5}, step=5)
    assert run_b.finish() is True

    # Auto-default owner from getpass.getuser()
    monkeypatch.setattr(getpass, "getuser", lambda: "carol")
    run_c = track.start(
        "from-getpass",
        run_id="run-getpass-owner",
        store=store,
        environment={},
        flush_interval_s=3600.0,
        spill_dir=tmp_path / "spill",
        clock=lambda: now + 8.0,
    )
    assert run_c.owner == "carol"
    assert run_c.finish() is True

    # Validation rejects unknown visibility
    with pytest.raises(ValueError, match="visibility"):
        track.start("bad", store=store, visibility="public", environment={}, spill_dir=tmp_path / "spill")

    # Add a run with >256 metrics and explicit timestamps to verify capping & duration_s
    root_store = LocalTrackingStore(tmp_path / "store")
    many_metrics = {f"m/{i:03d}": float(i) for i in range(270)}
    _put_segment(
        root_store,
        "meta",
        [
            {"op": "start", "t": 1000.0, "name": "wide", "owner": "alice"},
            {"op": "finish", "t": 1042.5, "status": "finished", "summary": {"from_finish": 99}},
        ],
        run="run-wide",
    )
    _put_segment(
        root_store,
        "scalars",
        [{"step": 0, "t": 1010.0, "v": many_metrics, "x": {}}],
        run="run-wide",
    )

    index = TrackingIndex(tmp_path / "index.sqlite3")
    index.scan(root_store)

    runs = {r["run_id"]: r for r in index.list_runs("acme", "research")}
    assert runs["run-explicit"]["owner"] == "alice"
    assert runs["run-explicit"]["visibility"] == "private"
    assert runs["run-explicit"]["experiment"] == "nanogpt-speedrun"
    assert runs["run-explicit"]["summary"] == {"best_val": 3.15}
    assert runs["run-explicit"]["latest_metrics"] == {"train/loss": 3.4, "val/loss": 3.15}
    assert isinstance(runs["run-explicit"]["duration_s"], float)

    assert runs["run-env-owner"]["owner"] == "bob"
    assert runs["run-env-owner"]["visibility"] == "team"
    assert runs["run-getpass-owner"]["owner"] == "carol"

    wide_summary = runs["run-wide"]
    assert wide_summary["duration_s"] == 42.5
    assert wide_summary["summary"] == {"from_finish": 99}
    assert len(wide_summary["latest_metrics"]) == 256
    assert list(wide_summary["latest_metrics"]) == [f"m/{i:03d}" for i in range(256)]

    wide_detail = index.run_detail("acme", "research", "run-wide")
    assert wide_detail is not None
    assert wide_detail["latest_metrics"] == wide_summary["latest_metrics"]
    assert wide_detail["duration_s"] == 42.5

    # Verify backfill when reopening a pre-existing DB without latest_scalars rows
    with index._connection() as db:
        db.execute("DELETE FROM latest_scalars")
    reopened = TrackingIndex(tmp_path / "index.sqlite3")
    assert reopened.run_detail("acme", "research", "run-explicit")["latest_metrics"] == {  # type: ignore[index]
        "train/loss": 3.4,
        "val/loss": 3.15,
    }


# --------------------------------------------------------------------------- 2. Workload Owner Env Injection


def test_render_execution_injects_workload_owner_and_strips_spoofed_value() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    controller = EyvanController(repository, service, settings=EyvanSettings())

    workload = WorkloadRecord(
        id="wl-owner-test",
        org_id="acme",
        project_id="research",
        created_by="alice",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            environment={"EYVAN_WORKLOAD_OWNER": "spoofed-user", "KEEP": "yes"},
        ),
    )
    _, env = controller._render_execution(
        workload,
        runtime_environment={"EYVAN_WORKLOAD_OWNER": "spoofed-runtime", "RUNTIME": "ok"},
    )
    assert env["EYVAN_WORKLOAD_ID"] == "wl-owner-test"
    assert env["EYVAN_WORKLOAD_OWNER"] == "alice"
    assert env["KEEP"] == "yes"
    assert env["RUNTIME"] == "ok"
    keys = list(env)
    assert keys.index("EYVAN_WORKLOAD_OWNER") == keys.index("EYVAN_WORKLOAD_ID") + 1

    # When created_by is blank, spoofed owner must be stripped completely
    anon = workload.model_copy(update={"created_by": ""})
    _, anon_env = controller._render_execution(anon, runtime_environment={"EYVAN_WORKLOAD_OWNER": "spoofed"})
    assert "EYVAN_WORKLOAD_OWNER" not in anon_env


# --------------------------------------------------------------------------- 3. CollaborationStore & Access Policy


def test_collaboration_store_persistence_and_validation(tmp_path: Path) -> None:
    db_path = tmp_path / "collab.sqlite3"
    store = CollaborationStore(db_path)

    ov = store.upsert_run_overlay(
        "acme",
        "research",
        "run-1",
        updated_by="alice",
        owner="alice",
        visibility="shared",
        shared_with=["carol", "bob", "bob", " "],
        experiment_id="exp_abc",
        pinned=True,
    )
    assert ov["shared_with"] == ["bob", "carol"]
    assert ov["pinned"] is True
    assert ov["experiment_id"] == "exp_abc"

    # Partial update preserves omitted fields while explicit None clears experiment_id
    ov2 = store.upsert_run_overlay(
        "acme",
        "research",
        "run-1",
        updated_by="alice",
        experiment_id=None,
    )
    assert ov2["visibility"] == "shared"
    assert ov2["shared_with"] == ["bob", "carol"]
    assert ov2["pinned"] is True
    assert ov2["experiment_id"] is None

    # Durable across reopen
    reopened = CollaborationStore(db_path)
    assert reopened.get_run_overlay("acme", "research", "run-1") == ov2

    # Validation checks
    with pytest.raises(InputError):
        store.create_experiment("acme", "research", name="bad", owner="alice", visibility="public")
    with pytest.raises(InputError):
        store.create_experiment("acme", "research", name="bad", owner="alice", kind="invalid")
    with pytest.raises(InputError):
        store.create_experiment("acme", "research", name="bad", owner="alice", goal="median")
    with pytest.raises(InputError):
        store.create_experiment("acme", "research", name="bad", owner="alice", target_value=float("nan"))

    # Access helper truth table
    assert can_view("bob", owner="alice", visibility="team", shared_with=[], is_admin=False) is True
    assert can_view("bob", owner="alice", visibility="private", shared_with=["bob"], is_admin=False) is False
    assert can_view("alice", owner="alice", visibility="private", shared_with=[], is_admin=False) is True
    assert can_view("admin", owner="alice", visibility="private", shared_with=[], is_admin=True) is True
    assert can_view("bob", owner="alice", visibility="shared", shared_with=["bob"], is_admin=False) is True
    assert can_view("carol", owner="alice", visibility="shared", shared_with=["bob"], is_admin=False) is False
    assert can_manage("alice", owner="alice", is_admin=False) is True
    assert can_manage("bob", owner="alice", is_admin=False) is False
    assert can_manage("admin", owner="alice", is_admin=True) is True
    assert can_manage("bob", owner=None, is_admin=False) is False
    assert can_manage("admin", owner=None, is_admin=True) is True


# --------------------------------------------------------------------------- 4. Visibility, Sharing & Multi-Run Compare


def test_visibility_selective_sharing_rbac_and_compare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    t0 = time.time() - 120.0
    store = LocalTrackingStore(tmp_path / "runs")
    # Private run by alice (set directly in tracker start metadata)
    _put_segment(
        store,
        "meta",
        [
            {"op": "start", "t": t0, "name": "alice-secret", "owner": "alice", "visibility": "private", "group": "g1"},
            {"op": "config", "t": t0 + 1, "config": {"lr": 1e-3, "model": {"heads": 8}, "seed": 1}},
            {"op": "finish", "t": t0 + 30, "status": "finished"},
        ],
        run="run-private",
    )
    _put_segment(
        store,
        "scalars",
        [
            {"step": 0, "t": t0 + 5, "v": {"val/loss": 4.0, "train/loss": 4.2}, "x": {}},
            {"step": 100, "t": t0 + 25, "v": {"val/loss": 3.1, "train/loss": 3.0}, "x": {}},
        ],
        run="run-private",
    )
    # Team run by alice (will be switched to shared with bob via PATCH)
    _put_segment(
        store,
        "meta",
        [
            {"op": "start", "t": t0 + 10, "name": "alice-shared", "owner": "alice", "group": "g1"},
            {"op": "config", "t": t0 + 11, "config": {"lr": 2e-3, "model": {"heads": 8}, "seed": 2}},
            {"op": "finish", "t": t0 + 60, "status": "finished"},
        ],
        run="run-shared",
    )
    _put_segment(
        store,
        "scalars",
        [
            {"step": 0, "t": t0 + 15, "v": {"val/loss": 4.5, "train/loss": 4.6, "extra/lr": 0.002}, "x": {}},
            {"step": 100, "t": t0 + 55, "v": {"val/loss": 3.4, "train/loss": 3.2, "extra/lr": 0.001}, "x": {}},
        ],
        run="run-shared",
    )
    # Team run by bob
    _put_segment(
        store,
        "meta",
        [
            {"op": "start", "t": t0 + 20, "name": "bob-team", "owner": "bob", "group": "g2"},
            {"op": "config", "t": t0 + 21, "config": {"lr": 5e-4, "model": {"heads": 8}, "seed": 2}},
            {"op": "finish", "t": t0 + 70, "status": "finished"},
        ],
        run="run-team",
    )
    _put_segment(
        store,
        "scalars",
        [
            {"step": 0, "t": t0 + 25, "v": {"val/loss": 4.1, "train/loss": 4.3}, "x": {}},
            {"step": 100, "t": t0 + 65, "v": {"val/loss": 3.2, "train/loss": 3.1}, "x": {}},
        ],
        run="run-team",
    )

    client, _app, _index = _build_collab_app(tmp_path, monkeypatch, tracker_store=store)
    with client:
        # Alice marks run-shared as shared with bob only, and pins it
        patched = client.patch(
            "/api/v1/collaboration/runs/run-shared",
            params=SCOPE,
            headers=_auth("alice"),
            json={"visibility": "shared", "shared_with": ["bob"], "pinned": True},
        )
        assert patched.status_code == 200, patched.text
        assert patched.json()["visibility"] == "shared"
        assert patched.json()["shared_with"] == ["bob"]
        assert patched.json()["pinned"] is True

        # Bob cannot edit alice's run overlay
        assert (
            client.patch(
                "/api/v1/collaboration/runs/run-shared",
                params=SCOPE,
                headers=_auth("bob"),
                json={"visibility": "team"},
            ).status_code
            == 403
        )

        # Also create an overlay on run-private so /series is blocked pre-network
        assert (
            client.patch(
                "/api/v1/collaboration/runs/run-private",
                params=SCOPE,
                headers=_auth("alice"),
                json={"visibility": "private"},
            ).status_code
            == 200
        )

        # Non-member outsider gets 403 immediately
        assert client.get("/api/v1/tracking/runs", params=SCOPE, headers=_auth("outsider")).status_code == 403

        # Visibility matrix on GET /runs:
        alice_runs = {
            r["run_id"] for r in client.get("/api/v1/tracking/runs", params=SCOPE, headers=_auth("alice")).json()["runs"]
        }
        bob_runs = {
            r["run_id"] for r in client.get("/api/v1/tracking/runs", params=SCOPE, headers=_auth("bob")).json()["runs"]
        }
        carol_runs = {
            r["run_id"] for r in client.get("/api/v1/tracking/runs", params=SCOPE, headers=_auth("carol")).json()["runs"]
        }
        admin_runs = {
            r["run_id"] for r in client.get("/api/v1/tracking/runs", params=SCOPE, headers=_auth("admin")).json()["runs"]
        }

        assert alice_runs == {"run-private", "run-shared", "run-team"}
        assert bob_runs == {"run-shared", "run-team"}
        assert carol_runs == {"run-team"}
        assert admin_runs == {"run-private", "run-shared", "run-team"}

        # Single-run access enforcement
        assert client.get("/api/v1/tracking/runs/run-private", params=SCOPE, headers=_auth("bob")).status_code == 403
        assert (
            client.get(
                "/api/v1/tracking/runs/run-private/series",
                params={**SCOPE, "key": "val/loss"},
                headers=_auth("bob"),
            ).status_code
            == 403
        )
        assert client.get("/api/v1/tracking/runs/run-shared", params=SCOPE, headers=_auth("bob")).status_code == 200
        assert client.get("/api/v1/tracking/runs/run-shared", params=SCOPE, headers=_auth("carol")).status_code == 403
        assert client.get("/api/v1/tracking/runs/run-private", params=SCOPE, headers=_auth("admin")).status_code == 200

        # Control-plane filter params work without leaking to tracker
        filtered = client.get(
            "/api/v1/tracking/runs",
            params={**SCOPE, "owner": "alice", "visibility": "shared", "group": "g1"},
            headers=_auth("bob"),
        )
        assert filtered.status_code == 200
        assert [r["run_id"] for r in filtered.json()["runs"]] == ["run-shared"]

        # Multi-run compare: bob can compare run-shared and run-team
        compared = client.post(
            "/api/v1/tracking/compare",
            headers=_auth("bob"),
            json={**SCOPE, "run_ids": ["run-shared", "run-team"], "x": "step", "points": 200},
        )
        assert compared.status_code == 200, compared.text
        cmp_body = compared.json()
        assert cmp_body["common_keys"] == ["train/loss", "val/loss"]
        assert cmp_body["all_keys"] == ["extra/lr", "train/loss", "val/loss"]
        assert cmp_body["selected_keys"] == ["train/loss", "val/loss"]
        assert cmp_body["series_by_key"]["val/loss"]["run-shared"]["x"] == [0, 100]
        assert cmp_body["series_by_key"]["val/loss"]["run-shared"]["y"] == [4.5, 3.4]
        assert cmp_body["series_by_key"]["val/loss"]["run-team"]["y"] == [4.1, 3.2]

        # Config diff: differs=True sorted before differs=False
        diffs = cmp_body["config_diff"]
        assert diffs[0]["key"] == "lr" and diffs[0]["differs"] is True
        assert diffs[0]["values"] == {"run-shared": 2e-3, "run-team": 5e-4}
        non_diff = {d["key"]: d for d in diffs if not d["differs"]}
        assert non_diff["model.heads"]["values"] == {"run-shared": 8, "run-team": 8}
        assert non_diff["seed"]["values"] == {"run-shared": 2, "run-team": 2}

        # Metric summary
        summary_by_key = {m["key"]: m["values"] for m in cmp_body["metric_summary"]}
        assert summary_by_key["val/loss"]["run-shared"]["latest"] == 3.4
        assert summary_by_key["val/loss"]["run-shared"]["min"] == 3.4
        assert summary_by_key["val/loss"]["run-shared"]["max"] == 4.5
        assert summary_by_key["val/loss"]["run-team"]["last_step"] == 100

        # Compare refuses 403 if ANY requested run is private to another researcher
        forbidden_cmp = client.post(
            "/api/v1/tracking/compare",
            headers=_auth("bob"),
            json={**SCOPE, "run_ids": ["run-team", "run-private"]},
        )
        assert forbidden_cmp.status_code == 403

        # Compare returns 404 for missing run
        missing_cmp = client.post(
            "/api/v1/tracking/compare",
            headers=_auth("alice"),
            json={**SCOPE, "run_ids": ["run-team", "run-does-not-exist"]},
        )
        assert missing_cmp.status_code == 404


# --------------------------------------------------------------------------- 5. Speedruns, Leaderboards, Notes & Feed


def test_speedruns_leaderboards_notes_and_team_feed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = time.time()
    store = LocalTrackingStore(tmp_path / "runs")

    # Baseline run (alice): reaches val/loss <= 3.28 at step 200, t = +200s
    _put_segment(
        store,
        "meta",
        [
            {"op": "start", "t": 1000.0, "name": "baseline-v1", "owner": "alice", "group": "nanogpt", "tags": ["gpt"]},
            {"op": "finish", "t": 1300.0, "status": "finished"},
        ],
        run="run-base",
    )
    _put_segment(
        store,
        "scalars",
        [
            {"step": 0, "t": 1000.0, "v": {"val/loss": 5.0}, "x": {}},
            {"step": 100, "t": 1100.0, "v": {"val/loss": 3.5}, "x": {}},
            {"step": 200, "t": 1200.0, "v": {"val/loss": 3.25}, "x": {}},
            {"step": 300, "t": 1300.0, "v": {"val/loss": 3.20}, "x": {}},
        ],
        run="run-base",
    )

    # Faster record run (bob): reaches val/loss <= 3.28 at step 150, t = +90s (matched via experiment name!)
    _put_segment(
        store,
        "meta",
        [
            {"op": "start", "t": 2000.0, "name": "muon-fast", "owner": "bob", "experiment": "NanoGPT 3.28"},
            {"op": "finish", "t": 2150.0, "status": "finished"},
        ],
        run="run-fast",
    )
    _put_segment(
        store,
        "scalars",
        [
            {"step": 0, "t": 2000.0, "v": {"val/loss": 4.9}, "x": {}},
            {"step": 150, "t": 2090.0, "v": {"val/loss": 3.26}, "x": {}},
            {"step": 200, "t": 2150.0, "v": {"val/loss": 3.12}, "x": {}},
        ],
        run="run-fast",
    )

    # In-progress active run (alice): still running, hasn't hit 3.28 yet
    _put_segment(
        store,
        "meta",
        [{"op": "start", "t": now - 30.0, "name": "live-exp", "owner": "alice", "group": "nanogpt"}],
        run="run-live",
    )
    _put_segment(
        store,
        "scalars",
        [
            {"step": 0, "t": now - 25.0, "v": {"val/loss": 4.8}, "x": {}},
            {"step": 50, "t": now - 5.0, "v": {"val/loss": 3.6}, "x": {}},
        ],
        run="run-live",
    )

    # RL max-reward speedrun runs
    _put_segment(
        store,
        "meta",
        [
            {"op": "start", "t": 3000.0, "name": "grpo-low", "owner": "alice", "tags": ["grpo-math"]},
            {"op": "finish", "t": 3100.0, "status": "finished"},
        ],
        run="run-rl-1",
    )
    _put_segment(
        store,
        "scalars",
        [{"step": 10, "t": 3050.0, "v": {"eval/reward": 0.62}, "x": {}}],
        run="run-rl-1",
    )
    _put_segment(
        store,
        "meta",
        [
            {"op": "start", "t": 3000.0, "name": "grpo-high", "owner": "bob", "tags": ["grpo-math"]},
            {"op": "finish", "t": 3120.0, "status": "finished"},
        ],
        run="run-rl-2",
    )
    _put_segment(
        store,
        "scalars",
        [{"step": 10, "t": 3060.0, "v": {"eval/reward": 0.89}, "x": {}}],
        run="run-rl-2",
    )

    client, _app, _index = _build_collab_app(tmp_path, monkeypatch, tracker_store=store)
    with client:
        # Create min-loss speedrun
        created = client.post(
            "/api/v1/collaboration/experiments",
            headers=_auth("alice"),
            json={
                **SCOPE,
                "name": "NanoGPT 3.28",
                "subject": "pretraining",
                "kind": "speedrun",
                "metric_key": "val/loss",
                "goal": "min",
                "target_value": 3.28,
                "group_filter": "nanogpt",
                "pinned_baseline_run_id": "run-base",
            },
        )
        assert created.status_code == 201, created.text
        exp_id = created.json()["id"]
        assert exp_id.startswith("exp_")
        assert created.json()["owner"] == "alice"

        # Leaderboard ranks fastest time_to_target_s first!
        lb = client.get(f"/api/v1/collaboration/experiments/{exp_id}/leaderboard", params=SCOPE, headers=_auth("bob"))
        assert lb.status_code == 200, lb.text
        body = lb.json()
        assert body["tracker_available"] is True
        assert body["baseline_run_id"] == "run-base"
        assert body["baseline_value"] == 3.20
        entries = body["entries"]
        assert [e["run_id"] for e in entries] == ["run-fast", "run-base", "run-live"]

        fast_entry, base_entry, live_entry = entries
        assert fast_entry["rank"] == 1 and fast_entry["is_record"] is True
        assert fast_entry["reached_target"] is True
        assert fast_entry["steps_to_target"] == 150
        assert fast_entry["time_to_target_s"] == 90.0
        assert fast_entry["delta_vs_baseline"] == pytest.approx(3.12 - 3.20)

        assert base_entry["rank"] == 2 and base_entry["is_baseline"] is True
        assert base_entry["reached_target"] is True
        assert base_entry["steps_to_target"] == 200
        assert base_entry["time_to_target_s"] == 200.0
        assert base_entry["delta_vs_baseline"] == 0.0

        assert live_entry["rank"] == 3 and live_entry["reached_target"] is False

        # Researcher breakdown
        by_researcher = {r["owner"]: r for r in body["researcher_breakdown"]}
        assert by_researcher["bob"]["run_count"] == 1
        assert by_researcher["bob"]["best_value"] == 3.12
        assert by_researcher["bob"]["reached_target_count"] == 1
        assert by_researcher["alice"]["run_count"] == 2
        assert by_researcher["alice"]["running_count"] == 1

        # Create max-reward speedrun matched by tag_filter
        rl_exp = client.post(
            "/api/v1/collaboration/experiments",
            headers=_auth("bob"),
            json={
                **SCOPE,
                "name": "Math RL",
                "subject": "rl",
                "kind": "speedrun",
                "metric_key": "eval/reward",
                "goal": "max",
                "tag_filter": ["grpo-math"],
            },
        )
        assert rl_exp.status_code == 201
        rl_lb = client.get(
            f"/api/v1/collaboration/experiments/{rl_exp.json()['id']}/leaderboard",
            params=SCOPE,
            headers=_auth("carol"),
        ).json()
        assert [e["run_id"] for e in rl_lb["entries"]] == ["run-rl-2", "run-rl-1"]
        assert rl_lb["entries"][0]["is_record"] is True
        assert rl_lb["entries"][0]["best_value"] == 0.89

        # Notes CRUD + permissions
        note_resp = client.post(
            "/api/v1/collaboration/notes",
            headers=_auth("alice"),
            json={
                **SCOPE,
                "target_type": "experiment",
                "target_id": exp_id,
                "body": "Muon optimizer cut time to 3.28 by more than half.",
                "metric_ref": "val/loss",
                "step_ref": 150,
            },
        )
        assert note_resp.status_code == 201, note_resp.text
        note_id = note_resp.json()["id"]
        assert note_resp.json()["author"] == "alice"

        notes_list = client.get(
            "/api/v1/collaboration/notes",
            params={**SCOPE, "target_type": "experiment", "target_id": exp_id},
            headers=_auth("bob"),
        )
        assert notes_list.status_code == 200
        assert [n["id"] for n in notes_list.json()["notes"]] == [note_id]

        # Peer cannot delete alice's note; admin can
        assert (
            client.delete(f"/api/v1/collaboration/notes/{note_id}", params=SCOPE, headers=_auth("bob")).status_code
            == 403
        )

        # Also add a note on a run and check live feed
        run_note = client.post(
            "/api/v1/collaboration/notes",
            headers=_auth("bob"),
            json={**SCOPE, "target_type": "run", "target_id": "run-fast", "body": "New record!"},
        )
        assert run_note.status_code == 201

        feed = client.get("/api/v1/collaboration/feed", params=SCOPE, headers=_auth("carol"))
        assert feed.status_code == 200, feed.text
        feed_data = feed.json()
        assert feed_data["tracker_available"] is True
        assert [r["run_id"] for r in feed_data["active_runs"]] == ["run-live"]
        assert len(feed_data["recent_runs"]) == 5
        assert len(feed_data["recent_notes"]) == 2
        exps_by_id = {e["id"]: e for e in feed_data["experiments"]}
        assert exps_by_id[exp_id]["run_count"] == 3
        assert exps_by_id[exp_id]["running_count"] == 1
        assert exps_by_id[exp_id]["leader_run_id"] == "run-fast"
        assert exps_by_id[exp_id]["leader_owner"] == "bob"
        assert exps_by_id[exp_id]["leader_value"] == 3.12
        researchers_by_owner = {r["owner"]: r for r in feed_data["researchers"]}
        assert researchers_by_owner["alice"]["running_runs"] == 1
        assert researchers_by_owner["bob"]["total_runs"] == 2

        # Delete note as author, delete other note as admin
        assert (
            client.delete(
                f"/api/v1/collaboration/notes/{run_note.json()['id']}", params=SCOPE, headers=_auth("bob")
            ).status_code
            == 204
        )
        assert (
            client.delete(f"/api/v1/collaboration/notes/{note_id}", params=SCOPE, headers=_auth("admin")).status_code
            == 204
        )

        # Peer cannot patch or delete alice's experiment; owner can patch & delete
        assert (
            client.patch(
                f"/api/v1/collaboration/experiments/{exp_id}",
                params=SCOPE,
                headers=_auth("bob"),
                json={"name": "hijacked"},
            ).status_code
            == 403
        )
        patched_exp = client.patch(
            f"/api/v1/collaboration/experiments/{exp_id}",
            params=SCOPE,
            headers=_auth("alice"),
            json={"target_value": 3.10, "pinned_baseline_run_id": None},
        )
        assert patched_exp.status_code == 200
        assert patched_exp.json()["target_value"] == 3.10
        assert patched_exp.json()["pinned_baseline_run_id"] is None

        assert (
            client.delete(
                f"/api/v1/collaboration/experiments/{exp_id}", params=SCOPE, headers=_auth("alice")
            ).status_code
            == 204
        )
        assert (
            client.get(f"/api/v1/collaboration/experiments/{exp_id}", params=SCOPE, headers=_auth("alice")).status_code
            == 404
        )


def test_leaderboard_and_feed_handle_offline_tracker_gracefully(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _app, _index = _build_collab_app(tmp_path, monkeypatch, tracker_enabled=False)
    with client:
        exp = client.post(
            "/api/v1/collaboration/experiments",
            headers=_auth("alice"),
            json={**SCOPE, "name": "Offline Check", "kind": "speedrun", "metric_key": "loss"},
        )
        assert exp.status_code == 201
        exp_id = exp.json()["id"]

        lb = client.get(f"/api/v1/collaboration/experiments/{exp_id}/leaderboard", params=SCOPE, headers=_auth("alice"))
        assert lb.status_code == 200
        assert lb.json()["tracker_available"] is False
        assert lb.json()["entries"] == []
        assert lb.json()["researcher_breakdown"] == []

        feed = client.get("/api/v1/collaboration/feed", params=SCOPE, headers=_auth("alice"))
        assert feed.status_code == 200
        assert feed.json()["tracker_available"] is False
        assert feed.json()["active_runs"] == []
        assert len(feed.json()["experiments"]) == 1


# --------------------------------------------------------------------------- 6. Storage Region Check Route


def test_storage_check_region_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, app, _index = _build_collab_app(tmp_path, monkeypatch, tracker_enabled=False)
    locations = {
        "example-bucket-central": BucketLocation("example-bucket-central", "us-central1", "region"),
        "example-bucket-europe": BucketLocation("example-bucket-europe", "europe-west4", "region"),
        "example-bucket-us": BucketLocation("example-bucket-us", "us", "multi-region"),
    }
    app.state.storage_location_resolver = locations.get

    with client:
        # Unauthenticated rejected
        assert (
            client.post("/api/v1/storage/check-region", json={"uri": "gs://example-bucket-central/ckpt"}).status_code
            == 401
        )

        # Same region -> allowed
        same = client.post(
            "/api/v1/storage/check-region",
            headers=_auth("alice"),
            json={"uri": "gs://example-bucket-central/ckpt", "region": "us-central1-a"},
        )
        assert same.status_code == 200, same.text
        assert same.json()["allowed"] is True
        assert same.json()["allowlisted"] is False
        assert same.json()["host_region"] == "us-central1"
        assert same.json()["bucket_location"] == "us-central1"

        # Cross region -> blocked
        cross = client.post(
            "/api/v1/storage/check-region",
            headers=_auth("alice"),
            json={"uri": "gs://example-bucket-europe/ckpt", "region": "us-central1"},
        )
        assert cross.status_code == 200
        assert cross.json()["allowed"] is False
        assert cross.json()["allowlisted"] is False
        assert cross.json()["bucket_location"] == "europe-west4"

        # Multi-region: allowed by default, blocked under strict=True
        multi_ok = client.post(
            "/api/v1/storage/check-region",
            headers=_auth("alice"),
            json={"uri": "gs://example-bucket-us/data", "region": "us-central1", "strict": False},
        )
        assert multi_ok.status_code == 200 and multi_ok.json()["allowed"] is True

        multi_strict = client.post(
            "/api/v1/storage/check-region",
            headers=_auth("alice"),
            json={"uri": "gs://example-bucket-us/data", "region": "us-central1", "strict": True},
        )
        assert multi_strict.status_code == 200 and multi_strict.json()["allowed"] is False

        # Allowlisted cross-region bucket -> allowed & allowlisted=True
        allowlisted = client.post(
            "/api/v1/storage/check-region",
            headers=_auth("alice"),
            json={
                "uri": "gs://example-bucket-europe/data",
                "region": "us-central1",
                "allowed_buckets": ["example-bucket-europe"],
            },
        )
        assert allowlisted.status_code == 200
        assert allowlisted.json()["allowed"] is True
        assert allowlisted.json()["allowlisted"] is True

        # Fallback to detect_gcp_context when region omitted
        monkeypatch.setattr(
            collab_routes,
            "detect_gcp_context",
            lambda config_path=None: GcpContext(on_gcp=True, region="us-central1", source="metadata"),
        )
        detected = client.post(
            "/api/v1/storage/check-region",
            headers=_auth("alice"),
            json={"uri": "gs://example-bucket-central/data"},
        )
        assert detected.status_code == 200 and detected.json()["allowed"] is True

        # Missing region with no detected context -> 400
        monkeypatch.setattr(
            collab_routes,
            "detect_gcp_context",
            lambda config_path=None: GcpContext(on_gcp=False, region=None, source="none"),
        )
        no_region = client.post(
            "/api/v1/storage/check-region",
            headers=_auth("alice"),
            json={"uri": "gs://example-bucket-central/data"},
        )
        assert no_region.status_code == 400

        # Non-GCS URI -> 400
        bad_uri = client.post(
            "/api/v1/storage/check-region",
            headers=_auth("alice"),
            json={"uri": "/tmp/local/file", "region": "us-central1"},
        )
        assert bad_uri.status_code == 400


# --------------------------------------------------------------------------- 7. CLI Collaboration & Experiments Commands


class _CollabRecordingClient:
    def __init__(self, responses: dict[tuple[str, str], object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, object], object]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
        json: object = None,
        headers: dict[str, str] | None = None,
    ) -> object:
        del headers
        self.calls.append((method, path, dict(params or {}), json))
        return self.responses[(method, path)]


def _cli_session(responses: dict[tuple[str, str], object]) -> tuple[SimpleNamespace, _CollabRecordingClient]:
    client = _CollabRecordingClient(responses)
    scope = SimpleNamespace(org_id="acme", project_id="research")
    return SimpleNamespace(client=client, workload_scope=lambda: scope), client


def test_cli_track_share_compare_note_and_experiments() -> None:
    session, client = _cli_session(
        {
            ("PATCH", "/api/v1/collaboration/runs/run-1"): {
                "run_id": "run-1",
                "visibility": "shared",
                "shared_with": ["bob", "carol"],
                "experiment_id": "exp_123",
                "pinned": True,
            },
            ("POST", "/api/v1/tracking/compare"): {
                "org_id": "acme",
                "project_id": "research",
                "x": "step",
                "runs": [
                    {"run_id": "run-1", "status": "finished", "owner": "alice", "name": "a"},
                    {"run_id": "run-2", "status": "finished", "owner": "bob", "name": "b"},
                ],
                "common_keys": ["train/loss"],
                "all_keys": ["train/loss"],
                "selected_keys": ["train/loss"],
                "series_by_key": {},
                "config_diff": [{"key": "lr", "values": {"run-1": 1e-3, "run-2": 3e-4}, "differs": True}],
                "metric_summary": [
                    {
                        "key": "train/loss",
                        "values": {
                            "run-1": {"latest": 3.1, "min": 3.1, "max": 4.0, "last_step": 100},
                            "run-2": {"latest": 3.4, "min": 3.4, "max": 4.2, "last_step": 100},
                        },
                    }
                ],
            },
            ("POST", "/api/v1/collaboration/notes"): {
                "id": "note_1",
                "target_type": "run",
                "target_id": "run-1",
                "author": "alice",
                "body": "Converged smoothly",
                "metric_ref": "train/loss",
                "step_ref": 100,
            },
            ("POST", "/api/v1/collaboration/experiments"): {
                "id": "exp_123",
                "name": "NanoGPT Speedrun",
                "kind": "speedrun",
                "visibility": "team",
                "owner": "alice",
            },
            ("GET", "/api/v1/collaboration/experiments"): {
                "experiments": [
                    {
                        "id": "exp_123",
                        "name": "NanoGPT Speedrun",
                        "kind": "speedrun",
                        "visibility": "team",
                        "owner": "alice",
                        "metric_key": "val/loss",
                        "goal": "min",
                        "target_value": 3.28,
                    }
                ]
            },
            ("GET", "/api/v1/collaboration/experiments/exp_123/leaderboard"): {
                "experiment": {"id": "exp_123", "name": "NanoGPT Speedrun", "kind": "speedrun"},
                "metric_key": "val/loss",
                "goal": "min",
                "target_value": 3.28,
                "tracker_available": True,
                "entries": [
                    {
                        "rank": 1,
                        "is_record": True,
                        "run_id": "run-1",
                        "owner": "alice",
                        "status": "finished",
                        "best_value": 3.12,
                        "delta_vs_baseline": -0.08,
                        "reached_target": True,
                        "time_to_target_s": 90.0,
                        "steps_to_target": 150,
                    }
                ],
                "researcher_breakdown": [],
                "notes_count": 1,
            },
        }
    )
    runner = CliRunner()

    # track share
    res = runner.invoke(
        track_cli.track,
        [
            "share",
            "run-1",
            "--visibility",
            "shared",
            "--with",
            "bob",
            "--with",
            "carol",
            "--experiment",
            "exp_123",
            "--pin",
        ],
        obj=session,
    )
    assert res.exit_code == 0, res.output
    assert "shared" in res.output and "bob" in res.output and "pinned" in res.output
    assert client.calls[0] == (
        "PATCH",
        "/api/v1/collaboration/runs/run-1",
        SCOPE,
        {"visibility": "shared", "shared_with": ["bob", "carol"], "experiment_id": "exp_123", "pinned": True},
    )

    # track compare
    cmp_res = runner.invoke(
        track_cli.track,
        ["compare", "run-1", "run-2", "--key", "train/loss", "--x", "step"],
        obj=session,
    )
    assert cmp_res.exit_code == 0, cmp_res.output
    assert "train/loss" in cmp_res.output and "Config differences" in cmp_res.output
    assert client.calls[1][0:2] == ("POST", "/api/v1/tracking/compare")
    assert client.calls[1][3] == {
        **SCOPE,
        "run_ids": ["run-1", "run-2"],
        "keys": ["train/loss"],
        "x": "step",
        "points": 300,
    }

    # track note
    note_res = runner.invoke(
        track_cli.track,
        ["note", "run-1", "Converged smoothly", "--metric", "train/loss", "--step", "100"],
        obj=session,
    )
    assert note_res.exit_code == 0, note_res.output
    assert "note_1" in note_res.output and "metric=train/loss" in note_res.output
    assert client.calls[2][3] == {
        **SCOPE,
        "target_type": "run",
        "target_id": "run-1",
        "body": "Converged smoothly",
        "metric_ref": "train/loss",
        "step_ref": 100,
    }

    # experiments create
    exp_create = runner.invoke(
        track_cli.experiments,
        [
            "create",
            "NanoGPT Speedrun",
            "--subject",
            "pretraining",
            "--speedrun",
            "--metric",
            "val/loss",
            "--goal",
            "min",
            "--target",
            "3.28",
            "--group",
            "nanogpt",
            "--visibility",
            "team",
        ],
        obj=session,
    )
    assert exp_create.exit_code == 0, exp_create.output
    assert "exp_123" in exp_create.output
    assert client.calls[3][3]["kind"] == "speedrun"  # type: ignore[index]
    assert client.calls[3][3]["target_value"] == 3.28  # type: ignore[index]

    # experiments list
    exp_list = runner.invoke(track_cli.experiments, ["list", "--kind", "speedrun"], obj=session)
    assert exp_list.exit_code == 0, exp_list.output
    assert "NanoGPT Speedrun" in exp_list.output
    assert client.calls[4][2] == {**SCOPE, "kind": "speedrun"}

    # experiments leaderboard (text & JSON)
    lb_res = runner.invoke(track_cli.experiments, ["leaderboard", "exp_123"], obj=session)
    assert lb_res.exit_code == 0, lb_res.output
    assert "*RECORD" in lb_res.output and "REACHED(90.0s/150steps)" in lb_res.output

    lb_json = runner.invoke(track_cli.experiments, ["leaderboard", "exp_123", "--json"], obj=session)
    assert lb_json.exit_code == 0, lb_json.output
    assert json.loads(lb_json.output)["entries"][0]["run_id"] == "run-1"

    # CLI registered on main
    from eyvan.sdk.cli import main

    assert "experiments" in main.commands
    assert {"share", "compare", "note"} <= set(track_cli.track.commands)


# --------------------------------------------------------------------------- 8. RBAC & Hardening Regressions


def _seed_rich_private_run(
    store: LocalTrackingStore,
    *,
    run_id: str,
    owner: str | None,
    workload_id: str | None = None,
) -> tuple[str, str, str]:
    """Seed a tracker-private run with scalars, hist, episodes, rollouts, completions, tables, media, and blob."""

    blob_bytes = b"\x89PNG\r\n\x1a\nprivate-image-payload"
    digest = hashlib.sha256(blob_bytes).hexdigest()
    blob_ref = {"sha256": digest, "content_type": "image/png", "bytes": len(blob_bytes)}
    store.put(f"acme/research/{blob_key(run_id, digest)}", blob_bytes)

    start_record: dict[str, object] = {
        "op": "start",
        "t": 1000.0,
        "name": f"secret-{run_id}",
        "visibility": "private",
    }
    if owner is not None:
        start_record["owner"] = owner
    if workload_id is not None:
        start_record["workload_id"] = workload_id

    _put_segment(store, "meta", [start_record, {"op": "finish", "t": 1050.0, "status": "finished"}], run=run_id)
    _put_segment(store, "scalars", [{"step": 1, "t": 1010.0, "v": {"loss": 1.25}, "x": {}}], run=run_id)
    _put_segment(
        store,
        "hist",
        [
            {
                "key": "weights",
                "step": 1,
                "t": 1011.0,
                "edges": [0.0, 1.0, 2.0],
                "counts": [3, 5],
                "n": 8,
                "under": 0,
                "over": 0,
                "nonfinite": 0,
                "min": 0.1,
                "max": 1.9,
                "mean": 1.0,
                "std": 0.5,
                "x": {},
            }
        ],
        run=run_id,
    )
    _put_segment(
        store,
        "episodes",
        [
            {
                "step": 1,
                "t": 1012.0,
                "ret": 42.0,
                "length": 10,
                "success": True,
                "kind": "train",
                "actor": "a0",
                "env": "cartpole",
                "seed": 1,
                "policy_version": 1,
                "learner_version": 1,
                "checkpoint": None,
                "x": {},
                "extra": {},
            }
        ],
        run=run_id,
    )
    _put_segment(
        store,
        "rollouts",
        [
            {
                "step": 1,
                "t": 1013.0,
                "episode_id": "ep-1",
                "kind": "train",
                "actor": "a0",
                "env": "cartpole",
                "policy_version": None,
                "ret": 42.0,
                "length": 2,
                "tracks": {"reward": [20.0, 22.0]},
                "actions": [0, 1],
                "action_names": None,
                "video": blob_ref,
                "fps": 24,
                "x": {},
            }
        ],
        run=run_id,
    )
    _put_segment(
        store,
        "completions",
        [
            {
                "step": 1,
                "t": 1014.0,
                "group": "g1",
                "prompt": "2+2?",
                "answer": "4",
                "policy_version": None,
                "responses": [
                    {
                        "text": "4",
                        "rewards": {"correct": 1.0},
                        "reward": 1.0,
                        "advantage": 0.0,
                        "tokens": 1,
                        "kl": None,
                        "truncated": False,
                    }
                ],
            }
        ],
        run=run_id,
    )
    _put_segment(
        store,
        "tables",
        [{"step": 1, "t": 1015.0, "key": "eval_table", "columns": ["q", "a"], "rows": [["2+2", "4"]]}],
        run=run_id,
    )
    _put_segment(
        store,
        "media",
        [{"step": 1, "t": 1016.0, "key": "sample_img", "caption": None, "blob": blob_ref}],
        run=run_id,
    )
    return "a1-r0-aaaaaaaaaaaa-0-0", "a1-r0-aaaaaaaaaaaa-0-0", digest


def test_all_13_tracking_subroutes_block_pre_overlay_tracker_private_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every per-run tracking sub-route must enforce tracker-declared privacy even with zero overlay."""

    store = LocalTrackingStore(tmp_path / "runs")
    rollout_id, completion_id, digest = _seed_rich_private_run(store, run_id="alice-priv-raw", owner="alice")
    # Also seed a private run owned only via workload_id in the control-plane repository (no tracker owner!)
    _, _, wl_digest = _seed_rich_private_run(store, run_id="wl-priv-raw", owner=None, workload_id="wl-alice-owned")

    client, app, _index = _build_collab_app(tmp_path, monkeypatch, tracker_store=store)
    # Put workload record owned by alice in the repository
    service = app.state.service  # type: ignore[attr-defined]

    async def _seed_workload() -> None:
        async with service.repository.transaction() as tx:
            await tx.put_workload(
                WorkloadRecord(
                    id="wl-alice-owned",
                    org_id="acme",
                    project_id="research",
                    created_by="alice",
                    spec=WorkloadDefinition(name="priv", execution=CommandExecution(command="echo hi")),
                )
            )

    asyncio.run(_seed_workload())

    subroutes = [
        ("series", "/api/v1/tracking/runs/alice-priv-raw/series", {**SCOPE, "key": "loss"}),
        ("histograms", "/api/v1/tracking/runs/alice-priv-raw/histograms", {**SCOPE, "key": "weights"}),
        ("episodes", "/api/v1/tracking/runs/alice-priv-raw/episodes", SCOPE),
        ("episode-curve", "/api/v1/tracking/runs/alice-priv-raw/episodes/curve", SCOPE),
        ("episode-breakdown", "/api/v1/tracking/runs/alice-priv-raw/episodes/breakdown", SCOPE),
        ("rollouts", "/api/v1/tracking/runs/alice-priv-raw/rollouts", SCOPE),
        ("rollout-detail", f"/api/v1/tracking/runs/alice-priv-raw/rollouts/{rollout_id}", SCOPE),
        ("completions", "/api/v1/tracking/runs/alice-priv-raw/completions", SCOPE),
        ("completion-stats", "/api/v1/tracking/runs/alice-priv-raw/completions/stats", SCOPE),
        ("completion-detail", f"/api/v1/tracking/runs/alice-priv-raw/completions/{completion_id}", SCOPE),
        ("tables", "/api/v1/tracking/runs/alice-priv-raw/tables", SCOPE),
        ("media", "/api/v1/tracking/runs/alice-priv-raw/media", SCOPE),
        ("blob", f"/api/v1/tracking/runs/alice-priv-raw/blobs/{digest}", SCOPE),
    ]
    assert len(subroutes) == 13

    with client:
        for name, path, params in subroutes:
            bob_resp = client.get(path, params=params, headers=_auth("bob"))
            assert bob_resp.status_code == 403, f"sub-route {name} leaked private run to bob: {bob_resp.status_code}"
            alice_resp = client.get(path, params=params, headers=_auth("alice"))
            assert alice_resp.status_code == 200, f"sub-route {name} blocked owner alice: {alice_resp.status_code}"
            admin_resp = client.get(path, params=params, headers=_auth("admin"))
            assert admin_resp.status_code == 200, f"sub-route {name} blocked admin: {admin_resp.status_code}"

        # Workload-owned private run with no tracker owner: alice allowed, bob blocked
        assert (
            client.get(
                "/api/v1/tracking/runs/wl-priv-raw/series",
                params={**SCOPE, "key": "loss"},
                headers=_auth("bob"),
            ).status_code
            == 403
        )
        assert (
            client.get(
                f"/api/v1/tracking/runs/wl-priv-raw/blobs/{wl_digest}",
                params=SCOPE,
                headers=_auth("bob"),
            ).status_code
            == 403
        )
        assert (
            client.get(
                "/api/v1/tracking/runs/wl-priv-raw/series",
                params={**SCOPE, "key": "loss"},
                headers=_auth("alice"),
            ).status_code
            == 200
        )

        # Older headerless tracker fallback: strip X-Eyvan-Run-* headers and verify BOTH JSON and blob still block bob
        real_transport = app.state.tracking_transport  # type: ignore[attr-defined]

        class _HeaderStrippingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                resp = await real_transport.handle_async_request(request)
                body = await resp.aread()
                cleaned_headers = {k: v for k, v in resp.headers.items() if not k.lower().startswith("x-eyvan-")}
                return httpx.Response(resp.status_code, content=body, headers=cleaned_headers, request=request)

        app.state.tracking_transport = _HeaderStrippingTransport()  # type: ignore[attr-defined]
        assert (
            client.get(
                "/api/v1/tracking/runs/alice-priv-raw/series",
                params={**SCOPE, "key": "loss"},
                headers=_auth("bob"),
            ).status_code
            == 403
        )
        assert (
            client.get(
                f"/api/v1/tracking/runs/alice-priv-raw/blobs/{digest}",
                params=SCOPE,
                headers=_auth("bob"),
            ).status_code
            == 403
        )
        assert (
            client.get(
                f"/api/v1/tracking/runs/alice-priv-raw/blobs/{digest}",
                params=SCOPE,
                headers=_auth("alice"),
            ).status_code
            == 200
        )


def test_ownership_hijack_refused_and_viewer_cannot_mutate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-owners cannot hijack pre-overlay/unowned runs; PROJECT_VIEWER cannot mutate collaboration state."""

    store = LocalTrackingStore(tmp_path / "runs")
    _put_segment(
        store,
        "meta",
        [{"op": "start", "t": 1000.0, "name": "alice-raw", "owner": "alice", "visibility": "private"}],
        run="alice-raw",
    )
    _put_segment(
        store,
        "meta",
        [{"op": "start", "t": 1001.0, "name": "wl-raw", "workload_id": "wl-owned-by-alice"}],
        run="wl-raw",
    )
    _put_segment(
        store,
        "meta",
        [{"op": "start", "t": 1002.0, "name": "unowned-raw"}],
        run="unowned-raw",
    )

    client, app, _index = _build_collab_app(tmp_path, monkeypatch, tracker_store=store)
    service = app.state.service  # type: ignore[attr-defined]

    async def _seed_workload() -> None:
        async with service.repository.transaction() as tx:
            await tx.put_workload(
                WorkloadRecord(
                    id="wl-owned-by-alice",
                    org_id="acme",
                    project_id="research",
                    created_by="alice",
                    spec=WorkloadDefinition(name="w", execution=CommandExecution(command="echo hi")),
                )
            )

    asyncio.run(_seed_workload())

    with client:
        # 1. Bob cannot hijack alice's pre-overlay tracker-owned run
        assert (
            client.patch(
                "/api/v1/collaboration/runs/alice-raw",
                params=SCOPE,
                headers=_auth("bob"),
                json={"visibility": "team"},
            ).status_code
            == 403
        )
        # 2. Bob cannot hijack alice's workload-owned run (resolved via tracker workload_id)
        assert (
            client.patch(
                "/api/v1/collaboration/runs/wl-raw",
                params=SCOPE,
                headers=_auth("bob"),
                json={"visibility": "team"},
            ).status_code
            == 403
        )
        # 3. Bob cannot claim/hijack a truly unowned run; admin can manage it
        assert (
            client.patch(
                "/api/v1/collaboration/runs/unowned-raw",
                params=SCOPE,
                headers=_auth("bob"),
                json={"pinned": True},
            ).status_code
            == 403
        )
        admin_patch = client.patch(
            "/api/v1/collaboration/runs/unowned-raw",
            params=SCOPE,
            headers=_auth("admin"),
            json={"pinned": True},
        )
        assert admin_patch.status_code == 200 and admin_patch.json()["owner"] is None

        # Alice CAN manage her own tracker-owned and workload-owned runs
        assert (
            client.patch(
                "/api/v1/collaboration/runs/alice-raw",
                params=SCOPE,
                headers=_auth("alice"),
                json={"pinned": True},
            ).status_code
            == 200
        )
        assert (
            client.patch(
                "/api/v1/collaboration/runs/wl-raw",
                params=SCOPE,
                headers=_auth("alice"),
                json={"visibility": "private"},
            ).status_code
            == 200
        )

        # Create an experiment and note as alice so we can test carol (PROJECT_VIEWER) against all 6 mutators
        exp = client.post(
            "/api/v1/collaboration/experiments",
            headers=_auth("alice"),
            json={**SCOPE, "name": "Team Benchmark", "visibility": "team"},
        )
        assert exp.status_code == 201
        exp_id = exp.json()["id"]
        note = client.post(
            "/api/v1/collaboration/notes",
            headers=_auth("alice"),
            json={**SCOPE, "target_type": "experiment", "target_id": exp_id, "body": "Baseline note"},
        )
        assert note.status_code == 201
        note_id = note.json()["id"]

        # Read-only PROJECT_VIEWER (carol) gets 403 on ALL 6 mutating collaboration endpoints
        assert (
            client.patch(
                "/api/v1/collaboration/runs/unowned-raw",
                params=SCOPE,
                headers=_auth("carol"),
                json={"pinned": False},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/v1/collaboration/experiments",
                headers=_auth("carol"),
                json={**SCOPE, "name": "Viewer Attempt"},
            ).status_code
            == 403
        )
        assert (
            client.patch(
                f"/api/v1/collaboration/experiments/{exp_id}",
                params=SCOPE,
                headers=_auth("carol"),
                json={"name": "Viewer Edit"},
            ).status_code
            == 403
        )
        assert (
            client.delete(
                f"/api/v1/collaboration/experiments/{exp_id}",
                params=SCOPE,
                headers=_auth("carol"),
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/v1/collaboration/notes",
                headers=_auth("carol"),
                json={**SCOPE, "target_type": "experiment", "target_id": exp_id, "body": "Viewer comment"},
            ).status_code
            == 403
        )
        assert (
            client.delete(
                f"/api/v1/collaboration/notes/{note_id}",
                params=SCOPE,
                headers=_auth("carol"),
            ).status_code
            == 403
        )
        # Carol still retains read-only access
        assert client.get("/api/v1/collaboration/experiments", params=SCOPE, headers=_auth("carol")).status_code == 200
        assert client.get("/api/v1/collaboration/notes", params=SCOPE, headers=_auth("carol")).status_code == 200
        assert client.get("/api/v1/collaboration/feed", params=SCOPE, headers=_auth("carol")).status_code == 200


def test_notes_and_feed_rbac_bypasses_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Case/whitespace target_type, missing target_type, and tracker-private run notes cannot bypass RBAC."""

    store = LocalTrackingStore(tmp_path / "runs")
    _put_segment(
        store,
        "meta",
        [{"op": "start", "t": 1000.0, "name": "priv-raw", "owner": "alice", "visibility": "private"}],
        run="priv-raw",
    )
    _put_segment(
        store,
        "meta",
        [{"op": "start", "t": 1005.0, "name": "pub-raw", "owner": "bob", "visibility": "team"}],
        run="pub-raw",
    )

    client, _app, _index = _build_collab_app(tmp_path, monkeypatch, tracker_store=store)
    with client:
        # Alice creates a private experiment and attaches notes to both her private run (no overlay!) and private exp
        priv_exp = client.post(
            "/api/v1/collaboration/experiments",
            headers=_auth("alice"),
            json={**SCOPE, "name": "Secret Exp", "visibility": "private"},
        )
        assert priv_exp.status_code == 201
        priv_exp_id = priv_exp.json()["id"]

        n_priv_run = client.post(
            "/api/v1/collaboration/notes",
            headers=_auth("alice"),
            json={**SCOPE, "target_type": "run", "target_id": "priv-raw", "body": "Secret run note"},
        )
        assert n_priv_run.status_code == 201

        n_priv_exp = client.post(
            "/api/v1/collaboration/notes",
            headers=_auth("alice"),
            json={**SCOPE, "target_type": "experiment", "target_id": priv_exp_id, "body": "Secret exp note"},
        )
        assert n_priv_exp.status_code == 201

        n_pub_run = client.post(
            "/api/v1/collaboration/notes",
            headers=_auth("bob"),
            json={**SCOPE, "target_type": "run", "target_id": "pub-raw", "body": "Public run note"},
        )
        assert n_pub_run.status_code == 201
        pub_note_id = n_pub_run.json()["id"]

        # Bob cannot comment on alice's tracker-private run (no overlay exists!)
        assert (
            client.post(
                "/api/v1/collaboration/notes",
                headers=_auth("bob"),
                json={**SCOPE, "target_type": "run", "target_id": "priv-raw", "body": "Sneaky comment"},
            ).status_code
            == 403
        )

        # Uppercase / padded target_type cannot bypass authorization
        assert (
            client.get(
                "/api/v1/collaboration/notes",
                params={**SCOPE, "target_type": "RUN", "target_id": "priv-raw"},
                headers=_auth("bob"),
            ).status_code
            == 403
        )
        assert (
            client.get(
                "/api/v1/collaboration/notes",
                params={**SCOPE, "target_type": "  EXPERIMENT  ", "target_id": priv_exp_id},
                headers=_auth("bob"),
            ).status_code
            == 403
        )

        # target_id without target_type and unknown target_type return 422
        assert (
            client.get(
                "/api/v1/collaboration/notes",
                params={**SCOPE, "target_id": "priv-raw"},
                headers=_auth("bob"),
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/api/v1/collaboration/notes",
                params={**SCOPE, "target_type": "bogus", "target_id": "priv-raw"},
                headers=_auth("bob"),
            ).status_code
            == 422
        )

        # Unfiltered GET /notes and GET /feed hide notes on BOTH tracker-private runs AND private experiments
        bob_all_notes = client.get("/api/v1/collaboration/notes", params=SCOPE, headers=_auth("bob"))
        assert bob_all_notes.status_code == 200
        assert [n["id"] for n in bob_all_notes.json()["notes"]] == [pub_note_id]

        bob_feed = client.get("/api/v1/collaboration/feed", params=SCOPE, headers=_auth("bob"))
        assert bob_feed.status_code == 200
        assert [n["id"] for n in bob_feed.json()["recent_notes"]] == [pub_note_id]
        assert [r["run_id"] for r in bob_feed.json()["recent_runs"]] == ["pub-raw"]

        # Owner alice sees all 3 notes
        alice_all_notes = client.get("/api/v1/collaboration/notes", params=SCOPE, headers=_auth("alice"))
        assert len(alice_all_notes.json()["notes"]) == 3


def test_storage_check_region_rbac_and_token_provider_instantiation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty-grant principals get 403; both API and CLI instantiate default_gcp_token_provider()."""

    client, _app, _index = _build_collab_app(tmp_path, monkeypatch, tracker_enabled=False)
    captured_tokens: list[object] = []

    def fake_factory() -> tp.Callable[[], str | None]:
        return lambda: "ya29.fake-token"

    def fake_resolve(bucket: str, *, token_provider: tp.Callable[[], str | None] | None = None) -> BucketLocation:
        assert token_provider is not None
        token = token_provider()
        captured_tokens.append(token)
        assert isinstance(token, str), f"expected string token from instantiated provider, got {type(token)}"
        return BucketLocation(bucket, "us-central1", "region")

    monkeypatch.setattr(collab_routes, "default_gcp_token_provider", fake_factory)
    monkeypatch.setattr(collab_routes, "resolve_bucket_location", fake_resolve)
    monkeypatch.setattr(storage_cli, "default_gcp_token_provider", fake_factory)
    monkeypatch.setattr(storage_cli, "resolve_bucket_location", fake_resolve)

    with client:
        # Empty-grant principal is refused with 403
        assert (
            client.post(
                "/api/v1/storage/check-region",
                headers=_auth("grantless"),
                json={"uri": "gs://example-bucket-central/ckpt", "region": "us-central1"},
            ).status_code
            == 403
        )
        # Active member invokes instantiated token provider and receives string token
        ok = client.post(
            "/api/v1/storage/check-region",
            headers=_auth("carol"),
            json={"uri": "gs://example-bucket-central/ckpt", "region": "us-central1"},
        )
        assert ok.status_code == 200 and ok.json()["allowed"] is True

    # CLI default resolver also instantiates default_gcp_token_provider()
    loc = storage_cli._default_location_resolver("example-bucket-central")
    assert loc is not None and loc.location == "us-central1"
    assert captured_tokens == ["ya29.fake-token", "ya29.fake-token"]


def test_sqlite_wal_scope_trimming_atomic_delete_and_tracker_index_hardening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify WAL mode, scope trimming/422, atomic delete, partial backfill, and immutable owner/visibility."""

    collab_db = tmp_path / "collab-wal.sqlite3"
    store = CollaborationStore(collab_db)
    with sqlite3.connect(collab_db) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    # Consistent trimming on write & read
    exp = store.create_experiment(" acme ", " research ", name="Atomic Exp", owner="alice")
    assert store.get_experiment("acme", "research", f" {exp['id']} ") is not None
    store.upsert_run_overlay("acme", "research", "run-1", updated_by="alice", owner="alice", experiment_id=exp["id"])
    store.create_note("acme", "research", target_type="experiment", target_id=exp["id"], author="alice", body="exp note")
    store.create_note("acme", "research", target_type="run", target_id="run-1", author="alice", body="run note")

    assert store.delete_experiment(" acme ", " research ", f" {exp['id']} ") is True
    assert store.get_experiment("acme", "research", exp["id"]) is None
    assert store.get_run_overlay("acme", "research", "run-1")["experiment_id"] is None  # type: ignore[index]
    assert store.list_notes("acme", "research", target_type="experiment", target_id=exp["id"]) == []
    assert len(store.list_notes("acme", "research", target_type="run", target_id="run-1")) == 1

    # Whitespace-only scope query/body params rejected with 422
    client, _app, _index = _build_collab_app(tmp_path, monkeypatch, tracker_enabled=False)
    with client:
        assert (
            client.get(
                "/api/v1/collaboration/experiments",
                params={"org_id": "   ", "project_id": "research"},
                headers=_auth("alice"),
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/api/v1/tracking/runs",
                params={"org_id": "acme", "project_id": "\t "},
                headers=_auth("alice"),
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v1/collaboration/experiments",
                headers=_auth("alice"),
                json={"org_id": "  ", "project_id": "research", "name": "Bad"},
            ).status_code
            == 422
        )

    # TrackingIndex WAL mode, partial latest_scalars backfill, and immutable earliest owner/visibility
    t_store = LocalTrackingStore(tmp_path / "tstore")
    _put_segment(
        t_store,
        "meta",
        [{"op": "start", "t": 1000.0, "name": "orig", "owner": "alice", "visibility": "private"}],
        run="run-immut",
        attempt=1,
    )
    # Later rogue attempt (even with clock rollback t=200 < 1000!) attempts to flip owner & visibility
    _put_segment(
        t_store,
        "meta",
        [{"op": "start", "t": 200.0, "name": "renamed", "owner": "mallory", "visibility": "team"}],
        run="run-immut",
        attempt=2,
    )
    _put_segment(t_store, "scalars", [{"step": 1, "t": 1010.0, "v": {"loss": 2.5}, "x": {}}], run="run-immut")
    _put_segment(t_store, "meta", [{"op": "start", "t": 1005.0, "name": "other", "owner": "bob"}], run="run-other")
    _put_segment(t_store, "scalars", [{"step": 1, "t": 1015.0, "v": {"loss": 1.5}, "x": {}}], run="run-other")

    idx_db = tmp_path / "tindex.sqlite3"
    index = TrackingIndex(idx_db)
    index.scan(t_store)
    with sqlite3.connect(idx_db) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    detail = index.run_detail("acme", "research", "run-immut")
    assert detail is not None
    assert detail["owner"] == "alice"
    assert detail["visibility"] == "private"
    assert detail["name"] == "renamed"

    # Partial backfill: delete latest_scalars for ONLY run-immut while run-other still exists
    with index._connection() as db:
        db.execute("DELETE FROM latest_scalars WHERE run = 'run-immut'")
    reopened = TrackingIndex(idx_db)
    assert reopened.run_detail("acme", "research", "run-immut")["latest_metrics"] == {"loss": 2.5}  # type: ignore[index]

    # Rebuild preserves WAL and exact data
    reopened.rebuild(t_store)
    with sqlite3.connect(idx_db) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert reopened.run_detail("acme", "research", "run-immut")["owner"] == "alice"  # type: ignore[index]


def test_compare_leaderboard_runs_hardening_and_webhook_400(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify NaN/Inf sanitization, compare warnings, leaderboard bool/NaN sort, runs over-fetch, and webhook 400."""

    # 1. /tracking/runs over-fetch: 5 newer private alice runs + 2 older bob runs with limit=2 returns bob's 2 runs
    t_store = LocalTrackingStore(tmp_path / "runs")
    for i in range(2):
        _put_segment(
            t_store,
            "meta",
            [{"op": "start", "t": 1000.0 + i, "name": f"bob-{i}", "owner": "bob", "visibility": "team"}],
            run=f"bob-run-{i}",
        )
    for i in range(5):
        _put_segment(
            t_store,
            "meta",
            [{"op": "start", "t": 2000.0 + i, "name": f"alice-{i}", "owner": "alice", "visibility": "private"}],
            run=f"alice-priv-{i}",
        )

    client, app, _index = _build_collab_app(tmp_path, monkeypatch, tracker_store=t_store)
    with client:
        bob_listed = client.get(
            "/api/v1/tracking/runs",
            params={**SCOPE, "limit": 2},
            headers=_auth("bob"),
        )
        assert bob_listed.status_code == 200
        assert [r["run_id"] for r in bob_listed.json()["runs"]] == ["bob-run-1", "bob-run-0"]

        # 2. Webhook private targets return 400 (never unhandled 500)
        wh_local = client.post(
            "/api/v1/webhooks",
            headers=_auth("admin"),
            json={
                **SCOPE,
                "url": "http://127.0.0.1:9000/hook",
                "secret": "s" * 32,
            },
        )
        assert wh_local.status_code == 400, wh_local.text
        wh_meta = client.post(
            "/api/v1/webhooks",
            headers=_auth("admin"),
            json={
                **SCOPE,
                "url": "http://169.254.169.254/latest/meta-data",
                "secret": "s" * 32,
            },
        )
        assert wh_meta.status_code == 400, wh_meta.text

        # 3. Compare NaN/Inf sanitization + partial series failure warnings + total series failure 502
        mode = {"series_r2": "ok"}

        def mock_tracker(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/api/v1/tracking/runs/r1":
                # Emit raw JSON containing NaN/Infinity (which Python json.loads accepts)
                raw = (
                    '{"run_id":"r1","owner":"alice","visibility":"team","keys":["loss"],'
                    '"config":{"lr":NaN,"scale":Infinity},"latest_metrics":{"loss":1.5},"last_step":10}'
                )
                return httpx.Response(200, content=raw.encode(), headers={"Content-Type": "application/json"})
            if path == "/api/v1/tracking/runs/r2":
                return httpx.Response(
                    200,
                    json={
                        "run_id": "r2",
                        "owner": "bob",
                        "visibility": "team",
                        "keys": ["loss"],
                        "config": {"lr": 0.001},
                        "latest_metrics": {"loss": 2.0},
                        "last_step": 10,
                    },
                )
            if path == "/api/v1/tracking/runs/r1/series":
                if mode["series_r2"] == "all_fail":
                    return httpx.Response(500, json={"detail": "r1 series exploded"})
                raw = (
                    '{"run_id":"r1","x":"step","series":'
                    '[{"key":"loss","points":[[0,NaN,NaN,NaN],[10,1.5,1.5,Infinity]]}]}'
                )
                return httpx.Response(200, content=raw.encode(), headers={"Content-Type": "application/json"})
            if path == "/api/v1/tracking/runs/r2/series":
                if mode["series_r2"] == "500":
                    return httpx.Response(500, json={"detail": "r2 upstream boom"})
                if mode["series_r2"] in {"timeout", "all_fail"}:
                    raise httpx.ReadTimeout("r2 timed out", request=request)
                return httpx.Response(
                    200,
                    json={"run_id": "r2", "x": "step", "series": [{"key": "loss", "points": [[10, 2.0, 2.0, 2.0]]}]},
                )
            if path == "/api/v1/tracking/runs":
                # Include NaN / bool updated_at to verify leaderboard sort & JSON sanitization
                raw = (
                    '{"runs":['
                    '{"run_id":"r1","owner":"alice","visibility":"team","group":"g-nan","status":"finished",'
                    '"updated_at":true,"duration_s":NaN,"latest_metrics":{"loss":1.5}},'
                    '{"run_id":"r2","owner":"bob","visibility":"team","group":"g-nan","status":"finished",'
                    '"updated_at":NaN,"duration_s":Infinity,"latest_metrics":{"loss":2.0}}'
                    "]}"
                )
                return httpx.Response(200, content=raw.encode(), headers={"Content-Type": "application/json"})
            return httpx.Response(404, json={"detail": "not found"})

        app.state.tracking_transport = httpx.MockTransport(mock_tracker)  # type: ignore[attr-defined]

        # NaN/Inf sanitized in compare response
        cmp_ok = client.post("/api/v1/tracking/compare", headers=_auth("alice"), json={**SCOPE, "run_ids": ["r1", "r2"]})
        assert cmp_ok.status_code == 200, cmp_ok.text
        body = cmp_ok.json()
        assert body["warnings"] == []
        assert body["series_by_key"]["loss"]["r1"]["y"] == [None, 1.5]
        assert body["series_by_key"]["loss"]["r1"]["max"] == [None, None]

        # Partial 500 surfaces warning
        mode["series_r2"] = "500"
        cmp_warn_500 = client.post(
            "/api/v1/tracking/compare", headers=_auth("alice"), json={**SCOPE, "run_ids": ["r1", "r2"]}
        )
        assert cmp_warn_500.status_code == 200
        assert len(cmp_warn_500.json()["warnings"]) == 1
        assert "r2" in cmp_warn_500.json()["warnings"][0]

        # Partial timeout surfaces warning
        mode["series_r2"] = "timeout"
        cmp_warn_to = client.post(
            "/api/v1/tracking/compare", headers=_auth("alice"), json={**SCOPE, "run_ids": ["r1", "r2"]}
        )
        assert cmp_warn_to.status_code == 200
        assert len(cmp_warn_to.json()["warnings"]) == 1
        assert "r2" in cmp_warn_to.json()["warnings"][0]

        # All runs failing returns 502
        mode["series_r2"] = "all_fail"
        cmp_all_fail = client.post(
            "/api/v1/tracking/compare", headers=_auth("alice"), json={**SCOPE, "run_ids": ["r1", "r2"]}
        )
        assert cmp_all_fail.status_code == 502

        # Leaderboard & feed handle NaN / Inf / bool updated_at without crashing
        mode["series_r2"] = "ok"
        exp = client.post(
            "/api/v1/collaboration/experiments",
            headers=_auth("alice"),
            json={**SCOPE, "name": "NaN Safe", "metric_key": "loss", "goal": "min", "group_filter": "g-nan"},
        )
        assert exp.status_code == 201
        exp_id = exp.json()["id"]
        lb = client.get(f"/api/v1/collaboration/experiments/{exp_id}/leaderboard", params=SCOPE, headers=_auth("alice"))
        assert lb.status_code == 200, lb.text
        assert [e["run_id"] for e in lb.json()["entries"]] == ["r1", "r2"]
        assert lb.json()["entries"][0]["duration_s"] is None

        feed = client.get("/api/v1/collaboration/feed", params=SCOPE, headers=_auth("alice"))
        assert feed.status_code == 200, feed.text
