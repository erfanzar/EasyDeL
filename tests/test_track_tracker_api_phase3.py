# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Authenticated phase-three routes and same-run, integrity-checked blob serving."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eyvan.track._format import SegmentName, blob_key, encode_segment
from eyvan.track._store import LocalTrackingStore
from eyvan.track.tracker import TrackingIndex, create_tracker_app

_BASE = "/api/v1/tracking/runs"
_SCOPE = {"org_id": "org", "project_id": "project"}
_AUTH = {"Authorization": "Bearer secret"}
_DATA = b"real blob bytes"
_DIGEST = hashlib.sha256(_DATA).hexdigest()
_REF = {"sha256": _DIGEST, "content_type": "image/png", "bytes": len(_DATA)}
_ID = "a1-r0-bbbbbbbbbbbb-0-0"
_ROUTES = (
    "rollouts",
    f"rollouts/{_ID}",
    "completions",
    f"completions/{_ID}",
    "completions/stats",
    "tables",
    "media",
    f"blobs/{_DIGEST}",
)


def seed(
    store: LocalTrackingStore, stream: str, records: list[dict[str, Any]], *, run: str = "run", seq: int = 0
) -> str:
    """Append a real test segment."""

    key = f"org/project/{run}/{stream}/{SegmentName(1, 0, 'bbbbbbbbbbbb', seq).filename}"
    store.put(key, encode_segment(records))
    return key


@pytest.fixture
def api(tmp_path: Path) -> Any:
    store = LocalTrackingStore(tmp_path / "store")
    store.put(f"org/project/{blob_key('run', _DIGEST)}", _DATA)
    seed(
        store,
        "rollouts",
        [
            {
                "t": 100,
                "step": 1,
                "episode_id": "ep",
                "actor": None,
                "env": "Pong",
                "kind": "train",
                "policy_version": None,
                "length": 2,
                "tracks": {"reward": [1, 2]},
                "actions": [0, 1],
                "action_names": None,
                "ret": 3,
                "video": _REF,
                "fps": 24,
                "x": {},
            }
        ],
    )
    seed(
        store,
        "completions",
        [
            {
                "t": 101,
                "step": 2,
                "group": "g",
                "prompt": "question",
                "answer": None,
                "policy_version": None,
                "responses": [
                    {
                        "text": "answer",
                        "rewards": {"correct": 2},
                        "reward": 2,
                        "advantage": 0,
                        "tokens": 4,
                        "kl": None,
                        "truncated": False,
                    }
                ],
            }
        ],
    )
    seed(store, "tables", [{"t": 102, "step": 3, "key": "cases", "columns": ["image"], "rows": [[_REF]]}])
    seed(store, "media", [{"t": 103, "step": 4, "key": "image", "caption": None, "blob": _REF}])
    index = TrackingIndex(tmp_path / "index.sqlite3")
    with TestClient(create_tracker_app(store, index, token="secret", scan_interval_s=3600)) as client:
        yield client, store, index


@pytest.mark.parametrize("route", _ROUTES)
def test_phase_three_auth_scope_and_unknown_run(api: Any, route: str) -> None:
    client, _, _ = api
    path = f"{_BASE}/run/{route}"
    assert client.get(path, params=_SCOPE).status_code == 401
    assert client.get(path, params=_SCOPE, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(path, params=_SCOPE, headers=_AUTH).status_code == 200
    assert client.get(f"{_BASE}/missing/{route}", params=_SCOPE, headers=_AUTH).status_code == 404
    assert client.get(path, params={**_SCOPE, "org_id": "bad id"}, headers=_AUTH).status_code == 400
    assert client.get(path, params={}, headers=_AUTH).status_code == 400


def test_route_payloads_and_unknown_detail_ids(api: Any) -> None:
    client, _, _ = api

    def get(route: str, **params: Any) -> Any:
        return client.get(f"{_BASE}/run/{route}", params={**_SCOPE, **params}, headers=_AUTH)

    summary = get("rollouts").json()["rollouts"][0]
    assert summary["id"] == _ID and summary["track_names"] == ["reward"]
    assert "tracks" not in summary and "actions" not in summary
    detail = get(f"rollouts/{_ID}").json()
    assert detail["tracks"] == {"reward": [1, 2]} and detail["video"]["available"] is True
    group = get("completions").json()["groups"][0]
    assert group["zero_variance"] is True and group["mean_reward"] == 2
    assert get(f"completions/{_ID}").json()["responses"][0]["text"] == "answer"
    assert get("completions/stats").json()["steps"][0]["components"] == {"correct": 2}
    assert get("tables").json() == {"keys": [{"key": "cases", "count": 1, "last_step": 3}]}
    assert get("tables", key="cases", step=3).json()["rows"] == [[{**_REF, "available": True}]]
    assert get("tables", key="absent").status_code == 404
    assert get("tables", key="cases", step=2).status_code == 404
    assert get("media").json()["media"][0]["blob"]["available"] is True
    for route in ("rollouts", "completions"):
        assert get(f"{route}/a1-r0-bbbbbbbbbbbb-0-99").status_code == 404


def test_blob_bytes_and_every_security_header(api: Any) -> None:
    client, _, _ = api
    response = client.get(f"{_BASE}/run/blobs/{_DIGEST}", params=_SCOPE, headers=_AUTH)
    assert response.status_code == 200 and response.content == _DATA
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-security-policy"] == "default-src 'none'; sandbox"
    assert response.headers["cache-control"] == "private, max-age=31536000, immutable"
    assert response.headers["content-disposition"] == "inline"


def test_blob_unreferenced_other_run_missing_and_corrupt_record_are_not_authorized(api: Any) -> None:
    client, store, index = api
    unknown = hashlib.sha256(b"unreferenced").hexdigest()
    store.put(f"org/project/{blob_key('run', unknown)}", b"unreferenced")
    other = hashlib.sha256(b"other run").hexdigest()
    ref = {"sha256": other, "content_type": "image/png", "bytes": 9}
    store.put(f"org/project/{blob_key('other', other)}", b"other run")
    # Even physically placing another run's bytes in this prefix is insufficient.
    store.put(f"org/project/{blob_key('run', other)}", b"other run")
    seed(store, "media", [{"t": 100, "step": 1, "key": "m", "caption": None, "blob": ref}], run="other")
    corrupt_digest = hashlib.sha256(b"corrupt reference").hexdigest()
    corrupt_ref = {"sha256": corrupt_digest, "content_type": "image/png", "bytes": 17}
    store.put(f"org/project/{blob_key('run', corrupt_digest)}", b"corrupt reference")
    seed(
        store,
        "media",
        [
            {"t": 100, "step": 1, "key": "m", "caption": None, "blob": corrupt_ref},
            {"t": "invalid"},
        ],
        seq=2,
    )
    index.scan(store)
    for digest in (unknown, other, corrupt_digest):
        response = client.get(f"{_BASE}/run/blobs/{digest}", params=_SCOPE, headers=_AUTH)
        assert response.status_code == 404
    (store.directory / f"org/project/{blob_key('run', _DIGEST)}").unlink()
    missing = client.get(f"{_BASE}/run/blobs/{_DIGEST}", params=_SCOPE, headers=_AUTH)
    assert missing.status_code == 404 and missing.json()["detail"] == "blob not available"
    listing = client.get(f"{_BASE}/run/media", params=_SCOPE, headers=_AUTH).json()
    assert listing["media"][0]["blob"]["available"] is False


def test_blob_integrity_mismatch_never_serves_tampered_bytes(api: Any) -> None:
    client, store, _ = api
    (store.directory / f"org/project/{blob_key('run', _DIGEST)}").write_bytes(b"tampered bytes")
    response = client.get(f"{_BASE}/run/blobs/{_DIGEST}", params=_SCOPE, headers=_AUTH)
    assert response.status_code == 500
    assert response.json()["detail"] == "blob integrity check failed"
    assert b"tampered bytes" not in response.content


def test_blob_disallowed_type_is_rejected_during_ingest_and_rechecked_when_serving(api: Any) -> None:
    client, store, index = api
    bad_digest = hashlib.sha256(b"<script>bad</script>").hexdigest()
    store.put(f"org/project/{blob_key('run', bad_digest)}", b"<script>bad</script>")
    seed(
        store,
        "media",
        [
            {
                "t": 104,
                "step": 5,
                "key": "html",
                "caption": None,
                "blob": {"sha256": bad_digest, "content_type": "text/html", "bytes": 20},
            }
        ],
        seq=1,
    )
    assert index.scan(store).errors == 1
    assert client.get(f"{_BASE}/run/blobs/{bad_digest}", params=_SCOPE, headers=_AUTH).status_code == 404
    # Defence in depth for an old/tampered disposable index: never trust its MIME string.
    with sqlite3.connect(index.db_path) as db:
        db.execute("UPDATE content_records SET payload = replace(payload, 'image/png', 'text/html')")
    assert client.get(f"{_BASE}/run/blobs/{_DIGEST}", params=_SCOPE, headers=_AUTH).status_code == 404


@pytest.mark.parametrize("stream", ["rollouts", "tables"])
def test_blobs_authorized_by_rollout_video_or_table_cell_alone(api: Any, stream: str) -> None:
    client, _, index = api
    with sqlite3.connect(index.db_path) as db:
        db.execute(
            "DELETE FROM content_records WHERE segment NOT IN (SELECT segment FROM segments WHERE stream = ?)", (stream,)
        )
    response = client.get(f"{_BASE}/run/blobs/{_DIGEST}", params=_SCOPE, headers=_AUTH)
    assert response.status_code == 200 and response.content == _DATA


@pytest.mark.parametrize(
    "route,params",
    [
        ("rollouts", {"kind": "bad"}),
        ("rollouts", {"actor": "a" * 65}),
        ("rollouts", {"limit": 501}),
        ("rollouts", {"offset": -1}),
        ("rollouts", {"include_replayed": "bad"}),
        ("rollouts/bad", {}),
        ("completions/bad", {}),
        ("completions", {"sort": "bad"}),
        ("completions", {"step": -1}),
        ("completions", {"limit": 501}),
        ("completions", {"offset": -1}),
        ("completions/stats", {"points": 0}),
        ("completions/stats", {"points": 2001}),
        ("tables", {"key": "bad key"}),
        ("tables", {"step": -1}),
        ("media", {"key": "bad key"}),
        ("media", {"limit": 1001}),
        ("media", {"offset": -1}),
        ("blobs/not-a-digest", {}),
    ],
)
def test_invalid_phase_three_parameters(api: Any, route: str, params: dict[str, Any]) -> None:
    client, _, _ = api
    response = client.get(f"{_BASE}/run/{route}", params={**_SCOPE, **params}, headers=_AUTH)
    assert response.status_code == 400
