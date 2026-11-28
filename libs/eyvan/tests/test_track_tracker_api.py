# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tracker API authentication, lifecycle scans and response contracts."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eyvan.track._format import SegmentName, encode_segment
from eyvan.track._store import LocalTrackingStore
from eyvan.track.tracker import TrackingIndex, create_tracker_app


@pytest.fixture
def tracker(tmp_path: Path) -> tuple[LocalTrackingStore, TrackingIndex]:
    store = LocalTrackingStore(tmp_path / "store")
    store.put(
        f"org/project/run/meta/{SegmentName(1, 0, 'bbbbbbbbbbbb', 0).filename}",
        encode_segment(
            [
                {"op": "start", "t": time.time(), "name": "PPO", "group": "g", "tags": ["rl"]},
                {"op": "config", "t": time.time(), "config": {"lr": 0.01}},
            ]
        ),
    )
    store.put(
        f"org/project/run/scalars/{SegmentName(1, 0, 'bbbbbbbbbbbb', 0).filename}",
        encode_segment([{"step": 1, "t": time.time(), "v": {"loss": 0.25}}]),
    )
    return store, TrackingIndex(tmp_path / "index.sqlite3")


def test_routes_shapes_and_initial_scan(tracker: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = tracker
    with TestClient(create_tracker_app(store, index, scan_interval_s=3600)) as client:
        health = client.get("/api/v1/tracking/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "service": "eyvan-tracker",
            "store": store.root,
            "indexed_segments": 2,
            "last_scan_at": health.json()["last_scan_at"],
        }
        assert isinstance(health.json()["last_scan_at"], float)
        params = {"org_id": "org", "project_id": "project"}
        response = client.get("/api/v1/tracking/runs", params=params)
        assert response.status_code == 200
        summary = response.json()["runs"][0]
        assert set(summary) == {
            "run_id",
            "org_id",
            "project_id",
            "name",
            "group",
            "tags",
            "owner",
            "visibility",
            "experiment",
            "status",
            "workload_id",
            "attempts",
            "started_at",
            "updated_at",
            "duration_s",
            "last_step",
            "keys_count",
            "summary",
            "latest_metrics",
        }
        assert summary["name"] == "PPO"
        assert summary["status"] == "running"
        detail = client.get("/api/v1/tracking/runs/run", params=params).json()
        assert detail["keys"] == ["loss"]
        assert detail["config"] == {"lr": 0.01}
        assert set(detail) == set(summary) | {
            "config",
            "keys",
            "axes",
            "attempt_list",
            "stats",
            "errors",
            "hist_keys",
            "episodes",
            "episode_axes",
            "envs",
            "actors",
            "rollouts",
            "completion_groups",
            "table_keys",
            "media_keys",
        }
        series_response = client.get("/api/v1/tracking/runs/run/series", params={**params, "key": "loss"})
        assert series_response.status_code == 200
        series = series_response.json()
        assert set(series) == {"run_id", "x", "series"}
        assert series["series"][0]["points"] == [[1, 0.25, 0.25, 0.25]]
        assert set(series["series"][0]) == {
            "key",
            "rank",
            "points",
            "downsampled",
            "total",
            "nonfinite",
            "attempts",
            "replayed",
        }
        assert client.get("/api/v1/tracking/runs/missing", params=params).status_code == 404
        assert client.get("/api/v1/tracking/runs/missing/series", params={**params, "key": "loss"}).status_code == 404
        assert client.get("/api/v1/tracking/runs", params={**params, "org_id": "other"}).json() == {"runs": []}


def test_auth_except_health_and_unicode_invalid_token(tracker: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = tracker
    with TestClient(create_tracker_app(store, index, token="private-secret")) as client:
        assert client.get("/api/v1/tracking/health").status_code == 200
        params = {"org_id": "org", "project_id": "project", "key": "loss"}
        for path in ("runs", "runs/run", "runs/run/series"):
            url = f"/api/v1/tracking/{path}"
            assert client.get(url, params=params).status_code == 401
            rejected = client.get(url, params=params, headers={"Authorization": "Bearer wrong"})
            assert rejected.status_code == 401
            assert "private-secret" not in rejected.text
            assert rejected.headers["www-authenticate"] == "Bearer"
            assert client.get(url, params=params, headers={"Authorization": "Bearer private-secret"}).status_code == 200


@pytest.mark.parametrize(
    "params",
    [
        {"org_id": "../escape", "project_id": "project"},
        {"org_id": "org", "project_id": "bad id"},
        {"org_id": "org", "project_id": "project", "points": "5001"},
        {"org_id": "org", "project_id": "project", "points": "0"},
    ],
)
def test_invalid_query_values_return_400(
    tracker: tuple[LocalTrackingStore, TrackingIndex], params: dict[str, str]
) -> None:
    store, index = tracker
    with TestClient(create_tracker_app(store, index)) as client:
        assert client.get("/api/v1/tracking/runs/run/series", params={**params, "key": "loss"}).status_code == 400


def test_invalid_run_and_too_many_keys(tracker: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = tracker
    with TestClient(create_tracker_app(store, index)) as client:
        params = [("org_id", "org"), ("project_id", "project")]
        assert client.get("/api/v1/tracking/runs/bad%20id", params=params).status_code == 400
        keys = [("key", f"key{i}") for i in range(33)]
        assert client.get("/api/v1/tracking/runs/run/series", params=params + keys).status_code == 400
        accepted = client.get("/api/v1/tracking/runs/run/series", params=params + keys[:32])
        assert accepted.status_code == 200
        assert len(accepted.json()["series"]) == 32


def test_background_scan_indexes_new_segments(tracker: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = tracker
    with TestClient(create_tracker_app(store, index, scan_interval_s=0.01)) as client:
        store.put(
            f"org/project/run/scalars/{SegmentName(1, 0, 'bbbbbbbbbbbb', 1).filename}",
            encode_segment([{"step": 2, "t": time.time(), "v": {"loss": 0.125}}]),
        )
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            response = client.get("/api/v1/tracking/runs/run", params={"org_id": "org", "project_id": "project"})
            if response.json()["last_step"] == 2:
                break
            time.sleep(0.01)
        else:
            pytest.fail("background scan did not ingest the new segment")


def test_module_cli_help_and_store_required() -> None:
    result = subprocess.run([sys.executable, "-m", "eyvan.track.tracker", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    for option in ("--store", "--db", "--host", "--port", "--scan-interval", "--log-level"):
        assert option in result.stdout
    missing = subprocess.run([sys.executable, "-m", "eyvan.track.tracker"], capture_output=True, text=True)
    assert missing.returncode == 2
    assert "--store" in missing.stderr
