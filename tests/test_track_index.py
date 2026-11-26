# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Observable index contracts over real local immutable tracking segments."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from eyvan.track._format import SegmentName, encode_segment
from eyvan.track._store import LocalTrackingStore
from eyvan.track.tracker.index import TrackingIndex


def put(
    store: LocalTrackingStore,
    stream: str,
    records: list[dict[str, object]],
    *,
    run: str = "run",
    attempt: int = 1,
    rank: int = 0,
    seq: int = 0,
    writer: str = "bbbbbbbbbbbb",
) -> str:
    name = SegmentName(attempt, rank, writer, seq)
    key = f"org/project/{run}/{stream}/{name.filename}"
    store.put(key, encode_segment(records))
    return key


def scalar(step: int, value: object, *, t: float = 1000, axes: dict[str, float] | None = None) -> dict[str, object]:
    return {"step": step, "t": t, "v": {"loss": value}, "x": axes or {}}


def test_incremental_scan_rebuild_and_deleted_database_are_identical(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    db = tmp_path / "index.sqlite3"
    index = TrackingIndex(db)
    put(store, "meta", [{"op": "start", "t": 1000, "name": "training", "tags": ["ppo"], "group": "g"}])
    put(store, "meta", [{"op": "config", "t": 1001, "config": {"lr": 0.01}}], seq=1)
    put(store, "scalars", [scalar(0, 3.0), scalar(1, 2.0, axes={"frames": 50})])
    first = index.scan(store)
    assert first.ingested_segments == 3
    assert first.indexed_segments == 3
    assert first.errors == 0
    assert index.scan(store).ingested_segments == 0
    put(store, "meta", [{"op": "summary", "t": 1002, "summary": {"best": 2.0}}], seq=2)
    assert index.scan(store).ingested_segments == 1
    expected = (
        index.list_runs("org", "project"),
        index.run_detail("org", "project", "run"),
        index.series("org", "project", "run", ["loss"]),
    )
    db.unlink()
    assert index.rebuild(store).ingested_segments == 4
    assert (
        index.list_runs("org", "project"),
        index.run_detail("org", "project", "run"),
        index.series("org", "project", "run", ["loss"]),
    ) == expected
    fresh = TrackingIndex(tmp_path / "fresh.sqlite3")
    fresh.scan(store)
    assert fresh.run_detail("org", "project", "run") == expected[1]
    assert expected[1]["config"] == {"lr": 0.01}
    assert expected[1]["summary"] == {"best": 2.0}


def test_corrupt_segment_is_recorded_once_without_poisoning_good_segments(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    index = TrackingIndex(tmp_path / "index.sqlite3")
    key = f"org/project/run/scalars/{SegmentName(1, 0, 'bbbbbbbbbbbb', 0).filename}"
    store.put(key, b"not gzip")
    put(store, "scalars", [scalar(1, 2)], seq=1)
    # The entire malformed segment is skipped, including its otherwise valid first row.
    put(store, "scalars", [scalar(2, 8), scalar(-1, 7)], seq=2)
    put(store, "histograms", [{"future": True}])
    store.put("org/project/run/meta/not-a-segment.json", b"{}")
    assert index.scan(store).errors == 2
    detail = index.run_detail("org", "project", "run")
    assert len(detail["errors"]) == 2
    assert detail["errors"][0]["segment"] == key
    assert index.series("org", "project", "run", ["loss"])["series"][0]["points"] == [[1, 2, 2, 2]]
    assert index.scan(store).ingested_segments == 0
    assert index.scan(store).errors == 0


def test_attempt_replay_rank_defaults_and_sequence_order(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    index = TrackingIndex(tmp_path / "index.sqlite3")
    put(store, "scalars", [scalar(0, 10), scalar(1, 11), scalar(2, 12)])
    put(store, "scalars", [scalar(1, 21), scalar(1, 22), scalar(3, 23)], attempt=2)
    put(store, "scalars", [scalar(1, 25)], attempt=2, seq=1)
    put(store, "scalars", [scalar(1, 90)], attempt=2, rank=3)
    index.scan(store)
    result = index.series("org", "project", "run", ["loss"], include_replayed=True)["series"][0]
    assert result["rank"] == 0
    assert result["points"] == [[0, 10, 10, 10], [1, 25, 25, 25], [2, 12, 12, 12], [3, 23, 23, 23]]
    assert result["attempts"] == [{"attempt": 1, "first_x": 0}, {"attempt": 2, "first_x": 1}]
    assert result["replayed"] == [{"attempt": 1, "points": [[1, 11]]}]
    assert index.series("org", "project", "run", ["loss"])["series"][0]["replayed"] == []
    assert index.series("org", "project", "run", ["loss"], rank=3)["series"][0]["points"] == [[1, 90, 90, 90]]


def test_metadata_order_and_per_writer_cumulative_stats(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    index = TrackingIndex(tmp_path / "index.sqlite3")
    put(
        store,
        "meta",
        [
            {"op": "start", "t": 100, "name": "first", "workload_id": "wl-1"},
            {"op": "config", "t": 1000, "config": {"version": 1}},
            {"op": "summary", "t": 100, "summary": {"a": 1, "b": 2}},
            {"op": "finish", "t": 101, "status": "failed"},
        ],
    )
    put(
        store,
        "meta",
        [
            {"op": "start", "t": 200, "name": "resumed"},
            {"op": "config", "t": 200, "config": {"version": 2}},
            {"op": "summary", "t": 201, "summary": {"a": 3}},
            {"op": "finish", "t": 202, "status": "finished"},
        ],
        attempt=2,
    )
    stats = {
        "op": "stats",
        "t": 200,
        "records": 8,
        "dropped": {"keys": 1, "rate": 2, "invalid": 3, "queue": 4},
        "spilled": 2,
    }
    put(store, "meta", [stats], seq=1)
    put(store, "meta", [dict(stats, t=201, records=10)], seq=2)
    put(store, "meta", [dict(stats, records=5)], writer="aaaaaaaaaaaa")
    index.scan(store)
    detail = index.run_detail("org", "project", "run")
    assert detail["name"] == "resumed"
    assert detail["workload_id"] == "wl-1"
    assert detail["attempts"] == 2
    assert detail["config"] == {"version": 2}
    assert detail["summary"] == {"a": 3, "b": 2}
    assert detail["status"] == "finished"
    assert detail["stats"] == {"records": 15, "dropped": {"keys": 2, "rate": 4, "invalid": 6, "queue": 8}, "spilled": 4}
    assert detail["attempt_list"][0]["status"] == "failed"


@pytest.mark.parametrize("status", ["finished", "failed", "aborted", "running", "unknown"])
def test_status_uses_latest_attempt(tmp_path: Path, status: str) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    index = TrackingIndex(tmp_path / "index.sqlite3")
    put(store, "meta", [{"op": "finish", "t": 100, "status": "finished"}])
    t = time.time() if status == "running" else 200
    records = [{"op": "start", "t": t}]
    if status in {"finished", "failed", "aborted"}:
        records.append({"op": "finish", "t": t + 1, "status": status})
    put(store, "meta", records, attempt=2)
    index.scan(store)
    assert index.run_detail("org", "project", "run")["status"] == status
    assert len(index.list_runs("org", "project", status=status)) == 1
    assert not index.list_runs("org", "project", status="not-a-status")


def test_downsampling_nonfinite_and_named_axes(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    index = TrackingIndex(tmp_path / "index.sqlite3")
    values = [1, "nan", 3, "inf", 5, "-inf"]
    records = [scalar(i, value, t=100 + i, axes={"frames": i * 10} if i != 2 else {}) for i, value in enumerate(values)]
    put(store, "scalars", records)
    index.scan(store)
    result = index.series("org", "project", "run", ["loss"], points=3)["series"][0]
    assert result["downsampled"]
    assert result["total"] == 6
    assert result["nonfinite"] == 3
    assert result["points"] == [[0, 1, 1, 1], [2, 3, 3, 3], [4, 5, 5, 5]]
    raw = index.series("org", "project", "run", ["loss"])["series"][0]
    assert not raw["downsampled"]
    assert raw["points"][1] == [1, None, None, None]
    timed = index.series("org", "project", "run", ["loss"], x="time")["series"][0]
    assert timed["points"][0] == [100, 1, 1, 1]
    custom = index.series("org", "project", "run", ["loss"], x="frames")["series"][0]
    assert [point[0] for point in custom["points"]] == [0, 10, 30, 40, 50]
    assert custom["total"] == 5
    assert index.run_detail("org", "project", "run")["axes"] == ["frames"]


def test_uneven_bucket_math_and_all_nonfinite_bucket(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    index = TrackingIndex(tmp_path / "index.sqlite3")
    put(store, "scalars", [scalar(i, v) for i, v in enumerate([1, 3, 5, 7, 9, "nan", "inf"])])
    index.scan(store)
    result = index.series("org", "project", "run", ["loss"], points=3)["series"][0]
    assert result["points"] == [[0, 2, 1, 3], [2, 6, 5, 7], [4, 9, 9, 9]]
    all_bad = index.series("org", "project", "run", ["loss"], points=6)["series"][0]
    assert all_bad["points"][-1] == [5, None, None, None]


def test_scope_sorting_limits_and_concurrent_reads(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    index = TrackingIndex(tmp_path / "index.sqlite3")
    for i in range(3):
        put(store, "scalars", [scalar(0, i, t=100 + i)], run=f"run{i}")
    index.scan(store)
    assert [r["run_id"] for r in index.list_runs("org", "project", limit=2)] == ["run2", "run1"]
    assert index.list_runs("other", "project") == []
    assert index.run_detail("org", "project", "absent") is None
    assert index.series("org", "project", "absent", ["loss"]) is None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: index.run_detail("org", "project", "run1"), range(20)))
    assert all(r["run_id"] == "run1" for r in results)


def test_invalid_deflate_payload_does_not_interrupt_scan(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    index = TrackingIndex(tmp_path / "index.sqlite3")
    broken = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff" + b"\xff" * 20
    store.put(f"org/project/run/scalars/{SegmentName(1, 0, 'bbbbbbbbbbbb', 0).filename}", broken)
    put(store, "scalars", [scalar(2, 3)], seq=1)
    assert index.scan(store).errors == 1
    assert index.series("org", "project", "run", ["loss"])["series"][0]["points"] == [[2, 3, 3, 3]]


def test_empty_scalar_record_still_updates_run_time_step_and_axes(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    index = TrackingIndex(tmp_path / "index.sqlite3")
    now = time.time()
    put(store, "scalars", [{"step": 4, "t": now, "v": {}, "x": {"frames": 100}}])
    index.scan(store)
    detail = index.run_detail("org", "project", "run")
    assert detail["status"] == "running"
    assert detail["last_step"] == 4
    assert detail["updated_at"] == now
    assert detail["axes"] == ["frames"]
    assert detail["keys_count"] == 0


def test_later_attempt_identity_survives_clock_rollback(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    index = TrackingIndex(tmp_path / "index.sqlite3")
    put(store, "meta", [{"op": "start", "t": 1000, "name": "old", "workload_id": "wl-1"}])
    put(store, "meta", [{"op": "start", "t": 200, "name": "new"}], attempt=2)
    index.scan(store)
    detail = index.run_detail("org", "project", "run")
    assert detail["name"] == "new"
    assert detail["workload_id"] == "wl-1"


@pytest.mark.parametrize("points", [0, -1, 5001])
def test_series_rejects_invalid_limits(tmp_path: Path, points: int) -> None:
    index = TrackingIndex(tmp_path / "index.sqlite3")
    with pytest.raises(ValueError):
        index.series("org", "project", "run", ["loss"], points=points)
