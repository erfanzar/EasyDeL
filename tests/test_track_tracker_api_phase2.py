# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Real ASGI queries for authenticated histogram and RL episode tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eyvan.track._format import SegmentName, encode_segment
from eyvan.track._store import LocalTrackingStore
from eyvan.track.tracker import TrackingIndex, create_tracker_app

_BASE = "/api/v1/tracking/runs"
_SCOPE = {"org_id": "org", "project_id": "project"}
_AUTH = {"Authorization": "Bearer secret"}
_ROUTES = ("histograms", "episodes", "episodes/curve", "episodes/breakdown")


@pytest.fixture
def api(tmp_path: Path) -> Any:
    store = LocalTrackingStore(tmp_path / "store")
    name = SegmentName(1, 0, "bbbbbbbbbbbb", 0).filename
    store.put(
        f"org/project/run/hist/{name}",
        encode_segment(
            [
                {
                    "key": "advantages",
                    "step": 1,
                    "t": 100,
                    "edges": [0, 1, 2],
                    "counts": [2, 4],
                    "n": 6,
                    "under": 1,
                    "over": 2,
                    "nonfinite": 1,
                    "min": -1,
                    "max": 3,
                    "mean": 1,
                    "std": 0.5,
                    "x": {"frames": 50},
                }
            ]
        ),
    )
    store.put(
        f"org/project/run/episodes/{name}",
        encode_segment(
            [
                {
                    "t": 101,
                    "step": 2,
                    "ret": 12,
                    "length": 4,
                    "success": True,
                    "kind": "train",
                    "actor": "actor 2",
                    "env": "Pong",
                    "seed": 7,
                    "policy_version": 1,
                    "learner_version": 3,
                    "checkpoint": None,
                    "x": {"frames": 100},
                    "extra": {"reward": 3},
                }
            ]
        ),
    )
    index = TrackingIndex(tmp_path / "index.sqlite3")
    with TestClient(create_tracker_app(store, index, token="secret", scan_interval_s=3600)) as client:
        yield client


@pytest.mark.parametrize("route", _ROUTES)
def test_phase_two_routes_auth_and_missing_run(api: TestClient, route: str) -> None:
    params = {**_SCOPE, "key": "advantages"}
    path = f"{_BASE}/run/{route}"
    assert api.get(path, params=params).status_code == 401
    assert api.get(path, params=params, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert api.get(path, params=params, headers=_AUTH).status_code == 200
    assert api.get(f"{_BASE}/missing/{route}", params=params, headers=_AUTH).status_code == 404
    assert api.get(f"{_BASE}/bad%20id/{route}", params=params, headers=_AUTH).status_code == 400
    assert api.get(path, params={**params, "org_id": "bad id"}, headers=_AUTH).status_code == 400
    assert api.get(path, params={"key": "advantages"}, headers=_AUTH).status_code == 400


def test_route_data_and_custom_axes(api: TestClient) -> None:
    result = api.get(
        f"{_BASE}/run/histograms", params={**_SCOPE, "key": "advantages", "bins": 4, "x": "frames"}, headers=_AUTH
    )
    assert result.status_code == 200
    histogram = result.json()
    assert histogram["edges"] == [0, 0.5, 1, 1.5, 2]
    assert histogram["columns"][0]["counts"] == [1, 1, 2, 2]
    assert histogram["columns"][0]["x"] == 50 and histogram["columns"][0]["under"] == 1
    unknown = api.get(f"{_BASE}/run/histograms", params={**_SCOPE, "key": "absent"}, headers=_AUTH)
    assert unknown.status_code == 200 and unknown.json()["edges"] == []
    episodes = api.get(
        f"{_BASE}/run/episodes", params={**_SCOPE, "actor": "actor 2", "env": "Pong"}, headers=_AUTH
    ).json()
    assert episodes["total"] == 1 and episodes["episodes"][0]["extra"] == {"reward": 3}
    curve = api.get(f"{_BASE}/run/episodes/curve", params={**_SCOPE, "x": "frames"}, headers=_AUTH).json()
    assert curve["buckets"][0]["x"] == 100 and curve["buckets"][0]["p50"] == 12
    breakdown = api.get(f"{_BASE}/run/episodes/breakdown", params=_SCOPE, headers=_AUTH).json()
    assert breakdown["actors"][0]["actor"] == "actor 2" and breakdown["lag"]["counts"][2] == 1
    detail = api.get(f"{_BASE}/run", params=_SCOPE, headers=_AUTH).json()
    assert detail["hist_keys"] == ["advantages"] and detail["episodes"] == {"train": 1, "eval": 0}


@pytest.mark.parametrize(
    "route,extra",
    [
        ("histograms", {}),
        ("histograms", {"key": "bad key"}),
        ("histograms", {"key": "k", "bins": 257}),
        ("histograms", {"key": "k", "points": 0}),
        ("histograms", {"key": "k", "points": 1001}),
        ("histograms", {"key": "k", "rank": 10000}),
        ("histograms", {"key": "k", "x": "bad axis"}),
        ("episodes", {"limit": 1001}),
        ("episodes", {"offset": -1}),
        ("episodes", {"kind": "other"}),
        ("episodes", {"actor": "a" * 65}),
        ("episodes", {"env": "e" * 129}),
        ("episodes", {"include_replayed": "bad"}),
        ("episodes/curve", {"points": 2001}),
        ("episodes/curve", {"metric": "bad"}),
        ("episodes/curve", {"kind": "bad"}),
        ("episodes/curve", {"x": "bad axis"}),
        ("episodes/breakdown", {"last": 0}),
        ("episodes/breakdown", {"last": 5001}),
        ("episodes/breakdown", {"kind": "bad"}),
    ],
)
def test_bad_phase_two_parameters_return_400(api: TestClient, route: str, extra: dict[str, Any]) -> None:
    response = api.get(f"{_BASE}/run/{route}", params={**_SCOPE, **extra}, headers=_AUTH)
    assert response.status_code == 400
    assert "Invalid tracking request parameters" in response.json()["detail"]
